import os
import os.path as osp
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import clip
from .base_engine import BaseEngine
import torch.nn.functional as F
import numpy as np
from open_clip.loss import ClipLoss
import copy
import time
from tqdm import tqdm
import gc
from torch.utils.checkpoint import checkpoint


def optimize_gpu_memory():
    """
    Automatically cleans up unused GPU memory to optimize usage.
    """
    # Collect garbage
    gc.collect()

    # Release all unused cached memory held by the caching allocator
    torch.cuda.empty_cache()

    # Optionally, force PyTorch to release GPU memory that it holds
    torch.cuda.ipc_collect()

    print("GPU memory optimized: ")
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


# From https://github.com/mlfoundations/wise-ft/blob/master/src/models/utils.py
def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


class ZSCLEngine(BaseEngine):
    def __init__(self, args, model) -> None:
        # Each engine should have the following attributes: model, optimizer, logger, criterion
        super().__init__(args)
        model, preprocess_fn = clip.load(args.backbone, device=args.device)
        model.float()
        model.eval()
        self.model = model
        self.ref_model = copy.deepcopy(model)
        # l2 model is updated after calling fit.
        self.l2_model = copy.deepcopy(model)
        self.we_model = copy.deepcopy(model)
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        self.weight_ensemble_count = 0
        self.iter_in_total = 0

    @staticmethod
    def modify_commandline_options(parser):
        parser.parse_known_args()
        parser.add_argument("--ref_caption_root", type=str, default="data/conceptual_captions")
        parser.add_argument("--ref_image_dataset", type=str, default="ImageNet")
        parser.add_argument("--warmup_length", type=int, default=100)
        parser.add_argument("--model_average_freq", type=int, default=100)
        parser.add_argument("--label_smoothing", type=float, default=0.2)
        parser.add_argument("--iter_per_stage", type=int, default=1000)
        return parser

    def configure_learnable_params(
        self, param_keys="all", requires_grad=True, **kwargs
    ):
        # freeze all the parameters in self.model, self.ref_model, self.l2_model, and self.we_model
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for name, param in self.ref_model.named_parameters():
            param.requires_grad = False
        for name, param in self.l2_model.named_parameters():
            param.requires_grad = False
        for name, param in self.we_model.named_parameters():
            param.requires_grad = False

        params = []
        for name, param in self.model.named_parameters():
            # only the logit_scale parameter is not learnable
            if "logit_scale" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                params.append(param)
        self.learnable_parameters = params

    @torch.no_grad()
    def encode_class_features(self, model, label_texts):
        if not label_texts:
            return None
        total_class_features = []
        cur_idx = 0
        while cur_idx < len(label_texts):
            text = label_texts[cur_idx: cur_idx + self.args.batch_size]
            labels = clip.tokenize(text).to(self.device)
            with torch.no_grad():
                class_features = model.encode_text(labels)
                class_features = class_features / class_features.norm(dim=-1, keepdim=True)
            total_class_features.append(class_features)
            cur_idx += self.args.batch_size
        total_class_features = torch.cat(total_class_features, dim=0)
        return total_class_features

    def encode_class_features_with_grad(self, model, label_texts):
        if not label_texts:
            return None
        total_class_features = []
        cur_idx = 0
        incremental_size = self.args.batch_size
        while cur_idx < len(label_texts):
            text = label_texts[cur_idx: cur_idx + incremental_size]
            labels = clip.tokenize(text).to(self.device)
            # use checkpoint to save memory
            class_features = checkpoint(model.encode_text, labels)
            # class_features = model.encode_text(labels)
            class_features = class_features / class_features.norm(dim=-1, keepdim=True)
            total_class_features.append(class_features)
            cur_idx += incremental_size
        total_class_features = torch.cat(total_class_features, dim=0)
        return total_class_features

    def configure_train_dataloader(self, train_dataset, **kwargs):
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        if hasattr(train_dataset, "current_classes_with_prompt"):
            self.train_classes = train_dataset.current_classes_with_prompt
            self.train_indices = train_dataset.current_classes_indices
            self.train_label_to_idx = {label.item(): idx for idx, label in enumerate(self.train_indices)}
        else:
            self.train_classes = train_dataset.classes
            self.train_indices = train_dataset.class_indices
            self.train_label_to_idx = {label.item(): idx for idx, label in enumerate(self.train_indices)}
        self.target_class_features = self.encode_class_features(self.model, self.train_classes)

    def configure_test_dataloader(self, test_dataset, **kwargs):
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )
        # for testing, we use we_model
        if hasattr(test_dataset, "classes_with_prompt"):
            self.target_class_features = self.encode_class_features(self.we_model, test_dataset.classes_with_prompt)
        else:
            self.target_class_features = self.encode_class_features(self.we_model, test_dataset.classes)

    def train_transform_data(self, inputs, targets, text_targets=None):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        if text_targets is not None and not isinstance(text_targets, tuple):
            text_targets = text_targets.to(self.device)
        # transform the target using self.train_label_to_idx, but targets might be on GPU
        targets = torch.tensor([self.train_label_to_idx[t.item()] for t in targets], device=self.device)
        return inputs, targets, text_targets

    def transform_data(self, inputs, targets, text_targets=None):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        if text_targets is not None and not isinstance(text_targets, tuple):
            text_targets = text_targets.to(self.device)
        return inputs, targets, text_targets

    @staticmethod
    def weight_ensemble(model_0, model_1, sma_count):
        for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
            param_k.data = (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
        return model_1

    def train_epoch(self, stage, epoch, set_train=True, reference_image_dataset=None, **kwargs):
        if set_train:
            self.model.train()
        else:
            self.model.eval()

        batch_num = 0
        train_loss = 0
        correct = 0
        total = 0

        num_batches = len(self.train_loader)

        reference_image_dataloader = DataLoader(
            reference_image_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

        ref_data_iter = iter(reference_image_dataloader)

        for batch_idx, (inputs, targets, text_targets) in enumerate(self.train_loader):
            start_time = time.time()
            inputs, targets, text_targets = self.train_transform_data(
                inputs, targets, text_targets=text_targets
            )
            ref_images, _, _ = next(ref_data_iter)
            ref_images = ref_images.to(self.device)
            with torch.autograd.set_detect_anomaly(True):
                self.optimizer.zero_grad()
                step = batch_idx + epoch * num_batches
                self.scheduler(step)
                loss, similarity = self.train_step(
                    inputs, targets, text_targets, ref_images, **kwargs
                )
                loss.backward()
                # # add gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.learnable_parameters, 1.0)
                self.optimizer.step()

            if self.iter_in_total % self.args.model_average_freq == 0:
                self.weight_ensemble_count += 1
                print("Averaging the model weights on iter {}".format(self.iter_in_total))
                self.weight_ensemble(
                    self.model, self.we_model, self.weight_ensemble_count
                )

            train_loss += loss.item()
            _, predicted = similarity.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_num += 1

            if batch_idx % self.args.log_interval == 0:
                if self.logger is not None:
                    self.logger.add_scalar(
                        f"train/stage_{stage}/loss", loss.item(), step
                    )
                    self.logger.add_scalar(
                        f"train/stage_{stage}/acc", 100.0 * correct / total, step
                    )
                print(
                    "Stage {} -- Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.0f}%)".format(
                        stage,
                        epoch,
                        batch_idx * len(inputs),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        loss.item(),
                        correct,
                        total,
                        100.0 * correct / total,
                    )
                )

            self.iter_in_total += 1

            if self.iter_in_total > self.args.iter_per_stage:
                print("Total number of iterations reached. Exiting the training loop.")
                self.in_training_budget = False
                break

            self.training_time += time.time() - start_time

        if self.logger is not None:
            self.logger.add_scalar(
                f"train/stage_{stage}/loss", train_loss / batch_num, epoch
            )
            self.logger.add_scalar(
                f"train/stage_{stage}/acc", 100.0 * correct / total, epoch
            )
        print(
            "Stage {} -- Train Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                stage,
                epoch,
                train_loss / batch_num,
                correct,
                total,
                100.0 * correct / total,
            )
        )

    def criterion(self, outputs, targets, text_targets):
        logits = outputs @ text_targets.T
        return self.loss_func(logits, targets)

    def train_step(self, inputs, targets, text_targets, reference_images, **kwargs):
        current_image_embeddings = self.model.encode_image(inputs)
        current_image_embeddings = current_image_embeddings / current_image_embeddings.norm(dim=-1, keepdim=True)

        # instead of using text_targets directly, we need to encode the self.classes
        text_embeddings = self.encode_class_features_with_grad(self.model, self.train_classes)

        # ce loss
        loss = self.criterion(self.model.logit_scale.exp() * current_image_embeddings, targets, text_embeddings)

        reference_images = reference_images.to(self.device)
        # encode the reference images using ref_model
        with torch.no_grad():
            ref_image_embeddings = self.ref_model.encode_image(reference_images)
            ref_image_embeddings = ref_image_embeddings / ref_image_embeddings.norm(dim=-1, keepdim=True)

        # encode the reference images using current model
        current_ref_image_embeddings = self.model.encode_image(reference_images)
        current_ref_image_embeddings = current_ref_image_embeddings / current_ref_image_embeddings.norm(dim=-1, keepdim=True)

        # ZSCL image loss
        logits_current = self.model.logit_scale.exp() * current_ref_image_embeddings @ self.ref_text_features.T
        logits_ref = self.model.logit_scale.exp() * ref_image_embeddings @ self.ref_text_features.T
        loss_zscl_image = self.distillation(logits_ref, logits_current)

        # ZSCL text loss
        loss_zscl_text = self.distillation(logits_ref.t(), logits_current.t())

        # l2 loss
        loss_l2 = self.l2_loss(self.model, self.l2_model)

        loss += loss_zscl_image + loss_zscl_text + loss_l2

        similarity = (100 * current_image_embeddings @ text_embeddings.T).softmax(dim=-1)
        return loss, similarity

    @staticmethod
    def distillation(t, s, T=2):
        p = F.softmax(t / T, dim=1)
        loss = F.cross_entropy(s / T, p, reduction="mean") * (T ** 2)
        return loss

    @staticmethod
    def l2_loss(model, model_ref):
        loss = 0.0
        for param_q, param_k in zip(model.parameters(), model_ref.parameters()):
            loss += F.mse_loss(param_q, param_k.detach(), reduction="sum")
        return loss

    def test_step(self, inputs, **kwargs):
        outputs = self.we_model.encode_image(inputs)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        similarity = (100 * outputs @ self.target_class_features.T).softmax(dim=-1)
        _, predicted = similarity.max(1)
        return predicted

    def _before_train_stage(self, train_dataset, **kwargs):
        self.model_adaptation(train_dataset, **kwargs)
        self.configure_learnable_params(**kwargs)
        self.configure_optimizers(**kwargs)
        self.configure_train_dataloader(train_dataset, **kwargs)
        # TODO: configure scheduler
        # num_batches = len(self.train_loader)
        self.scheduler = cosine_lr(
            self.optimizer,
            self.args.lr,
            self.args.warmup_length,
            self.args.iter_per_stage,
        )

        # reinitialize the weight_ensemble_count to 0
        self.weight_ensemble_count = 0
        # reinitialize the iter_in_total to 0
        self.iter_in_total = 0
        # set in_training_budget to True
        self.in_training_budget = True
        # calculate training time
        self.training_time = 0

    def fit(
        self, train_dataset, test_datasets=None, evaluation_tags=None, stage=0, reference_captions=None,
            reference_image_dataset=None, **kwargs
    ):
        # optimize GPU memory at the beginning of training
        optimize_gpu_memory()
        self._before_train_stage(train_dataset, **kwargs)

        # check the parameters requires_grad
        # for name, param in self.model.named_parameters():
        #     print("self.model", name, param.requires_grad)
        # input()
        # for name, param in self.ref_model.named_parameters():
        #     print("self.ref_model", name, param.requires_grad)
        # input()
        # for name, param in self.l2_model.named_parameters():
        #     print("self.l2_model", name, param.requires_grad)
        # input()
        # for name, param in self.we_model.named_parameters():
        #     print("self.we_model", name, param.requires_grad)
        # input()

        # prior to training, we can encode the reference text for good.
        if reference_captions is not None:
            # the feature is already normalized
            self.ref_text_features = self.encode_class_features(self.ref_model, reference_captions)

        acc = None
        for epoch_idx in range(self.args.start_epoch, self.args.n_epochs):
            # if epoch_idx % self.args.eval_interval == 0 and test_datasets is not None:
            #     assert len(test_datasets) == len(
            #         evaluation_tags
            #     ), "The number of test datasets should be equal to the number of evaluation tags"
            #     acc = self.evaluate(
            #         test_datasets,
            #         stage=stage,
            #         epoch=epoch_idx,
            #         evaluation_tags=evaluation_tags,
            #         **kwargs,
            #     )

            self._before_train_epoch(train_dataset, **kwargs)
            self.train_epoch(stage, epoch_idx, reference_image_dataset=reference_image_dataset, **kwargs)
            self._after_train_epoch(**kwargs)

            # controls early stop when the budget is exhausted
            if self.in_training_budget is False:
                break

        print("Training time: ", self.training_time)
        print("Average training time per iteration: ", self.training_time / self.weight_ensemble_count)

        # write time to file
        with open(os.path.join(self.args.results_dir, "zscl_training_time.txt"), "a") as outfile:
            outfile.write(
                f"{stage}, Total time: {self.training_time}, average time per iteration: {self.training_time / self.iter_in_total}\n")

        # if test_datasets is not None:
        #     acc = self.evaluate(
        #         test_datasets,
        #         stage=stage,
        #         epoch=self.args.n_epochs,
        #         evaluation_tags=evaluation_tags,
        #         **kwargs,
        #     )
        # save the model
        self.save_checkpoint(stage, epoch_idx, acc, **kwargs)
        # average the model weights here
        self._after_train_stage(stage, test_datasets, evaluation_tags)

    def save_checkpoint(self, stage, epoch, acc=None, **kwargs):
        if self.args.save:
            state = {
                "net": self.we_model.state_dict(),
                "acc": acc,
                "epoch": epoch,
                "stage": stage,
            }
            ckpt_dir = os.path.join(self.args.results_dir, "checkpoint")
            if not osp.isdir(ckpt_dir):
                os.mkdir(ckpt_dir)
            torch.save(
                state,
                os.path.join(ckpt_dir, "stage_%02d_ckpt_%04d.pth" % (stage, epoch)),
            )

    def _after_train_stage(self, stage, test_datasets, evaluation_tags=None, **kwargs):
        # update the L2 model with the current model
        self.l2_model = copy.deepcopy(self.model)
        self.l2_model.eval()
        # Evaluate the performances on the test datasets
        print(
            "After training. Evaluating the performances of the ensembled weights on the test datasets."
        )

        # optimize GPU memory
        torch.cuda.empty_cache()

        # self.evaluate(
        #     test_datasets,
        #     stage=stage,
        #     epoch=self.args.n_epochs,
        #     evaluation_tags=evaluation_tags,
        #     **kwargs,
        # )

    @staticmethod
    def similarity_calculation(features, target_label_features):
        # target_label_features can be so huge that we need to split it into chunks
        # to avoid OOM error
        chunk_size = 1000
        similarity = [
            100 * features @ target_label_features[i: i + chunk_size].T
            for i in range(0, target_label_features.shape[0], chunk_size)
        ]
        similarity = torch.cat(similarity, dim=-1).softmax(dim=-1)
        return similarity

    def evaluate_single_dataset(
        self, test_dataset, stage, tag, epoch=0, evaluate_seen_unseen=False, **kwargs
    ):
        # sourcery skip: low-code-quality
        self._before_evaluate_single_dataset(test_dataset, **kwargs)

        correct = 0
        total = 0
        learned_correct = 0
        learned_total = 0
        unseen_correct = 0
        unseen_total = 0
        total_loss = 0.0

        if evaluate_seen_unseen:
            if hasattr(test_dataset, "learned_classes_indices"):
                learned_classes_indices = test_dataset.learned_classes_indices
            else:
                assert hasattr(test_dataset, "labels"), "Dataset does not have labels"
                learned_classes_indices = torch.unique(
                    torch.from_numpy(test_dataset.labels), sorted=True
                )
        # print("learned_classes_indices:", learned_classes_indices)
        # input("Press Enter to continue...")

        with torch.no_grad():
            for images, labels, label_texts in tqdm(self.test_loader):
                images, labels, label_texts = self.transform_data(
                    images, labels, label_texts
                )
                # use previous model for inference
                features = self.we_model.encode_image(images)
                # similarity = (100 * features @ self.model.class_features.T).softmax(dim=-1)
                similarity = self.similarity_calculation(
                    features.float(), self.target_class_features
                )
                _, predicted = similarity.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                total_loss += torch.nn.functional.cross_entropy(
                    similarity, labels, reduction="sum"
                ).item()
                # find indices of samples from current classes and past classes
                if evaluate_seen_unseen:
                    for idx, ll in enumerate(labels):
                        if ll.item() in learned_classes_indices:
                            learned_total += 1
                            if predicted[idx] == ll:
                                learned_correct += 1
                        else:
                            unseen_total += 1
                            if predicted[idx] == ll:
                                unseen_correct += 1

        if self.logger is not None:
            self.logger.add_scalar(
                f"test/stage_{stage}/{tag}_acc", 100.0 * correct / total, epoch
            )

        print(
            "Test : on {} ({}) \tAcc: {:.3f} ({}/{})".format(
                tag, test_dataset.name, 100.0 * correct / total, correct, total
            )
        )
        print("Loss: {:.3f}".format(total_loss / total))

        if evaluate_seen_unseen:
            print(
                "Seen classes: {:.2f} ({}/{})".format(
                    100.0 * learned_correct / learned_total,
                    learned_correct,
                    learned_total,
                )
            )
            print(
                "Unseen classes: {:.2f} ({}/{})".format(
                    100.0 * unseen_correct / unseen_total if unseen_total > 0 else 0,
                    unseen_correct,
                    unseen_total,
                )
            )
        # optimize GPU memory
        torch.cuda.empty_cache()

        # instead of returning the overall accuracy, we return the accuracy of current classes, past classes, and overall, should be a dict
        if evaluate_seen_unseen:
            return {
                "overall": 100.0 * correct / total,
                "seen": 100.0 * learned_correct / learned_total,
                "unseen": 100.0 * unseen_correct / unseen_total
                if unseen_total > 0
                else 0,
                "overall_loss": total_loss / total,
            }
        else:
            return {
                "overall": 100.0 * correct / total,
                "overall_loss": total_loss / total,
            }