import torch
from tqdm import tqdm
from .base_engine import BaseEngine
from torch.utils.data import Subset, DataLoader
import numpy as np
import os.path as osp
import sys
import os

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

class MainEngine(BaseEngine):
    def __init__(self, args, zero_shot_model) -> None:
        super().__init__(args)
        self.model = zero_shot_model

    def resume(self, ckpt_path, **kwargs):
        if osp.isfile(ckpt_path):
            print(f"=> loading checkpoint '{ckpt_path}'")
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint["net"])
            self.args.start_epoch = checkpoint["epoch"]

            if hasattr(self.model.retrieval_branch, "linear_classifier"):
                self.model.retrieval_branch.linear_classifier = checkpoint[
                    "linear_classifier"
                ]
            if hasattr(self.model.retrieval_branch, "linear_classifiers"):
                self.model.retrieval_branch.linear_classifiers = checkpoint[
                    "linear_classifiers"
                ]
            print(
                f"""=> loaded checkpoint '{ckpt_path}' (epoch {checkpoint["epoch"]})"""
            )
        else:
            print(f"=> no checkpoint found at '{ckpt_path}'")

    def save_checkpoint(self, stage, epoch, acc=None, **kwargs):
        if self.args.save:
            state = {
                "net": self.model.state_dict(),
                "acc": acc,
                "epoch": epoch,
                "stage": stage,
            }
            if hasattr(self.model.retrieval_branch, "linear_classifier"):
                state[
                    "linear_classifier"
                ] = self.model.retrieval_branch.linear_classifier
            if hasattr(self.model.retrieval_branch, "linear_classifiers"):
                state[
                    "linear_classifiers"
                ] = self.model.retrieval_branch.linear_classifiers

            ckpt_dir = os.path.join(self.args.results_dir, "checkpoint")
            if not osp.isdir(ckpt_dir):
                os.mkdir(ckpt_dir)
            torch.save(
                state,
                os.path.join(ckpt_dir, "stage_%02d_ckpt_%04d.pth" % (stage, epoch)),
            )

    def evaluate(
        self,
        test_datasets,
        stage=0,
        epoch=0,
        evaluation_tags=None,
        evaluate_seen_unseen=False,
        **kwargs,
    ):
        if evaluation_tags is None:
            evaluation_tags = ["train_test"]
        if not isinstance(test_datasets, list):
            test_datasets = [test_datasets]
        if not isinstance(evaluation_tags, list):
            evaluation_tags = [evaluation_tags]
        return {
            evaluation_tag: self.evaluate_single_dataset(
                test_dataset,
                stage=stage,
                epoch=epoch,
                tag=evaluation_tag,
                evaluate_seen_unseen=evaluate_seen_unseen,
                **kwargs,
            )
            for test_dataset, evaluation_tag in zip(test_datasets, evaluation_tags)
        }

    def _before_evaluate_single_dataset(self, test_dataset, **kwargs):
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

        if hasattr(test_dataset, "classes_with_prompt"):
            target_classes = test_dataset.classes
            self.model.encode_class_features(test_dataset.classes_with_prompt)
            if type(test_dataset.labels) == list:
                unique_labels = np.unique(np.concatenate(test_dataset.labels))
            elif type(test_dataset.labels) == np.ndarray:
                unique_labels = np.unique(test_dataset.labels)
            else:
                unique_labels = np.unique(np.array(test_dataset.labels))
        else:
            raise ValueError(
                "Dataset does not have classes_with_prompt attribute nor learned_classes_with_prompt attribute"
            )

        unique_class_features = self.model.class_features
        self.target_class_features = unique_class_features

        self.model.retrieval_branch.set_calibration_features(
            target_classes, unique_class_features, unique_labels
        )
        self.model.retrieval_branch.find_mapping_from_exemplar_to_target()
        self.model.retrieval_branch.compute_overlapping_indices()

        self.model.eval()

    def transform_data(self, inputs, targets, text_targets=None):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        if text_targets is not None and not isinstance(text_targets, tuple):
            text_targets = text_targets.to(self.device)
        return inputs, targets, text_targets

    @staticmethod
    def similarity_calculation(features, target_label_features):
        # target_label_features can be so huge that we need to split it into chunks
        # to avoid OOM error
        chunk_size = 1000
        similarity = [
            100 * features @ target_label_features[i : i + chunk_size].T
            for i in range(0, target_label_features.shape[0], chunk_size)
        ]
        similarity = torch.cat(similarity, dim=-1).softmax(dim=-1)
        return similarity
    
    def evaluate_single_dataset(
        self, test_dataset, stage, tag, epoch=0, evaluate_seen_unseen=False, evaluate_current_past=False, **kwargs
    ):
        if evaluate_seen_unseen:
            return self.evaluate_single_dataset_seen_unseen(test_dataset, stage, tag, epoch, evaluate_seen_unseen=True, **kwargs)
        elif evaluate_current_past:
            return self.evaluate_single_dataset_current_past(test_dataset, stage, tag, epoch, evaluate_current_past=True, **kwargs)
        else:
            return self.evaluate_single_dataset_seen_unseen(test_dataset, stage, tag, epoch, evaluate_seen_unseen=False, **kwargs)

    def evaluate_single_dataset_seen_unseen(
        self, test_dataset, stage, tag, epoch=0, evaluate_seen_unseen=False, **kwargs
    ):
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

        with torch.no_grad():
            for images, labels, label_texts in tqdm(self.test_loader):
                images, labels, label_texts = self.transform_data(
                    images, labels, label_texts
                )
                features = self.model(images)
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

    def evaluate_single_dataset_current_past(
        self, test_dataset, stage, tag, epoch=0, evaluate_current_past=False, **kwargs
    ):
        self._before_evaluate_single_dataset(test_dataset, **kwargs)

        correct = 0
        total = 0
        current_correct = 0
        current_total = 0
        past_correct = 0
        past_total = 0

        if evaluate_current_past:
            if hasattr(test_dataset, "current_classes_indices"):
                current_classes_indices = test_dataset.current_classes_indices
                if hasattr(test_dataset, "past_classes_indices"):
                    past_classes_indices = test_dataset.past_classes_indices
                else:
                    past_classes_indices = test_dataset.current_classes_indices
            else:
                assert hasattr(test_dataset, "labels"), "Dataset does not have labels"
                current_classes_indices = torch.unique(
                    torch.from_numpy(test_dataset.labels), sorted=True
                )
                past_classes_indices = torch.unique(
                    torch.from_numpy(test_dataset.labels), sorted=True
                )

        with torch.no_grad():
            for images, labels, label_texts in tqdm(self.test_loader):
                images, labels, label_texts = self.transform_data(
                    images, labels, label_texts
                )
                features = self.model(images)
                similarity = self.similarity_calculation(
                    features.float(), self.target_class_features
                )
                _, predicted = similarity.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                # find indices of samples from current classes and past classes
                if evaluate_current_past:
                    for idx, ll in enumerate(labels):
                        if ll.item() in current_classes_indices:
                            current_total += 1
                            if predicted[idx] == ll:
                                current_correct += 1

                        if ll.item() in past_classes_indices:
                            past_total += 1
                            if predicted[idx] == ll:
                                past_correct += 1

        if self.logger is not None:
            self.logger.add_scalar(
                f"test/stage_{stage}/{tag}_acc", 100.0 * correct / total, epoch
            )

        print(
            "Test : on {} ({}) \tAcc: {:.3f} ({}/{})".format(
                tag, test_dataset.name, 100.0 * correct / total, correct, total
            )
        )
        if evaluate_current_past:
            print(
                "Current classes: {:.2f} ({}/{})".format(
                    100.0 * current_correct / current_total,
                    current_correct,
                    current_total,
                )
            )
            print(
                "Past classes: {:.2f} ({}/{})".format(
                    100.0 * past_correct / past_total, past_correct, past_total
                )
            )

        if evaluate_current_past:
            return {
                "overall": 100.0 * correct / total,
                "current": 100.0 * current_correct / current_total,
                "past": 100.0 * past_correct / past_total,
            }
        else:
            return {"overall": 100.0 * correct / total}