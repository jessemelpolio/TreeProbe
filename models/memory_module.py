import torch
from .base_network import BaseNetwork
import clip
from torch.utils.data import DataLoader
import faiss
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import copy
from torch.utils.data import TensorDataset
from models.modules import ClassificationHead
import time
import os
import open_clip

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class MemoryModule(BaseNetwork):
    def __init__(self, args, encoder=None) -> None:
        super().__init__(args)
        self.args = args
        self.buffer_type = args.buffer_type
        self.device = args.device
        if self.args.backbone == "ViT-H/14":
            self.clip_encoder = (
                open_clip.create_model_and_transforms(
                    "ViT-H-14", pretrained="laion2b_s32b_b79k", device=self.device
                )[0]
                .eval()
                .requires_grad_(False)
            )
        else:
            self.clip_encoder = (
                clip.load(args.backbone, jit=False, device=self.device)[0]
                .eval()
                .requires_grad_(False)
                if encoder is None
                else encoder
            )
        self.clip_encoder = self.clip_encoder.float()
        self.runtime_get_dim()

        # Exemplar set information
        self.sample_buffer = None
        self.exemplar_idx_to_class = {}
        self.exemplar_idx_to_class_with_prompt = {}
        self.exemplar_idx_to_label = {}
        self.exemplar_labels = None
        self.exemplar_classes_with_prompt = None
        self.exemplar_classes = None
        self.exemplar_classes_features = None

        self.target_labels = None
        self.target_classes = None
        self.target_classes_with_prompt = None
        self.target_classes_features = None

        self.k = args.k
        self.num_encoded_samples = 0
        self.fit_recorded_times = []
        self.inference_recorded_times = []
        self.set_steps = None

        if args.retriever == "knn":
            self.knn_searcher = None
        elif args.retriever == "tree_probe":
            from models.memory_helpers.simple_tree import SimpleTree

            self.knn_searcher = None
            self.tree_clustering = SimpleTree(
                max_instances=self.args.tree_probe_max_instances,
                min_samples_for_update=self.args.tree_probe_min_samples,
                args=self.args,
            )
            self.linear_classifiers = []
            self.classifier_labels = []
            self.centroids = []
        else:
            raise NotImplementedError

    @torch.no_grad()
    def runtime_get_dim(self):
        if "336px" in self.args.backbone:
            tensor = torch.randn(1, 3, 336, 336).to(self.device)
        else:
            tensor = torch.randn(1, 3, 224, 224).to(self.device)
        out = self.clip_encoder.encode_image(tensor)
        self.enc_dim = out.shape[-1]

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument(
            "--retriever",
            type=str,
            default="knn",
            help="Method for getting results from memory",
        )
        parser.add_argument("--kernel", type=str, default="rbf", help="Kernel for SVM")
        parser.add_argument("--k", type=int, default=20)
        parser.add_argument("--buffer_type", type=str, default="both")
        parser.add_argument("--num_samples_per_class", type=int, default=5)
        parser.add_argument("--sample_portion", type=float, default=0.2)
        parser.add_argument("--tau", type=float, default=0.1)
        parser.add_argument("--num_clusters", type=int, default=10)
        parser.add_argument("--lazy_update", action="store_true")
        parser.add_argument("--train_memory", action="store_true")
        parser.add_argument(
            "--calibration_out_type",
            type=str,
            default="max",
            choices=["max", "weighted"],
        )
        parser.add_argument("--tree_probe_max_instances", type=int, default=10000)
        parser.add_argument("--tree_probe_min_samples", type=int, default=1)
        parser.add_argument("--evaluation_iter_ratio", type=float, default=0.25)
        parser.add_argument("--use_pytorch_linear_for_tree_probe", action="store_true")
        parser.parse_known_args()
        return parser

    def build_retriever(self):
        # print('Building retriever...')
        if self.args.retriever in [
            "knn",
            "tree_probe",
        ]:
            # self.knn_searcher = faiss.IndexHNSWFlat(self.enc_dim, 32)
            self.knn_searcher = faiss.IndexFlatIP(self.enc_dim)
            self.knn_searcher.add(self.sample_buffer["image_features"].numpy())
        else:
            raise NotImplementedError

    def get_idx_to_class(self, idx_to_class):
        cursor = len(self.exemplar_idx_to_class)
        for idx, class_name in idx_to_class.items():
            if class_name not in self.exemplar_idx_to_class.values():
                self.exemplar_idx_to_class[cursor] = class_name
                cursor += 1

        # get mapping from the input idx_to_class to the idx_to_class in the buffer
        idx_to_class_in_buffer = {}
        for idx, class_name in idx_to_class.items():
            for idx_buffer, class_name_buffer in self.exemplar_idx_to_class.items():
                if class_name == class_name_buffer:
                    idx_to_class_in_buffer[idx] = idx_buffer
                    break
        return idx_to_class_in_buffer

    # for data incremental learning, some classes may not appear in the buffer but their labels are mapped using the whole dataset. This cause out of index error for some forward functions.
    def extract_dataset_info(self, dataset):
        self.unique_class_features = dataset.unique_class_features

    def extract_exemplar_info(self, dataset):
        if self.exemplar_classes is None:
            self.exemplar_classes = dataset.current_classes
            self.exemplar_classes_with_prompt = dataset.current_classes_with_prompt
            self.exemplar_classes_features = torch.from_numpy(
                dataset.current_classes_features
            ).float()

            for idx, class_label in enumerate(dataset.current_classes_indices):
                self.exemplar_idx_to_class[idx] = dataset.current_classes[idx]
                self.exemplar_idx_to_class_with_prompt[
                    idx
                ] = dataset.current_classes_with_prompt[idx]
                self.exemplar_idx_to_label[idx] = class_label.item()
        else:
            new_classes = [
                class_label
                for class_label in dataset.current_classes
                if class_label not in self.exemplar_classes
            ]
            new_classes_with_prompt = [
                class_label_with_prompt
                for class_label, class_label_with_prompt in zip(
                    dataset.current_classes, dataset.current_classes_with_prompt
                )
                if class_label not in self.exemplar_classes
            ]
            new_labels = [
                label.item()
                for label in dataset.current_classes_indices
                if label.item() not in self.exemplar_idx_to_label.values()
            ]
            new_classes_features = [
                class_features
                for class_features, class_label in zip(
                    dataset.current_classes_features, dataset.current_classes
                )
                if class_label not in self.exemplar_classes
            ]

            if len(new_classes) > 0:
                self.exemplar_classes += new_classes
                self.exemplar_classes_with_prompt += new_classes_with_prompt
                self.exemplar_classes_features = torch.cat(
                    [
                        self.exemplar_classes_features,
                        torch.from_numpy(np.array(new_classes_features)).float(),
                    ],
                    dim=0,
                )
                cursor = len(self.exemplar_idx_to_class)
                for idx, class_name in enumerate(new_classes):
                    self.exemplar_idx_to_class[cursor] = class_name
                    self.exemplar_idx_to_class_with_prompt[
                        cursor
                    ] = new_classes_with_prompt[idx]
                    self.exemplar_idx_to_label[cursor] = new_labels[idx]
                    cursor += 1

        def convert_idx_to_label(idx):
            return self.exemplar_idx_to_label[idx]

        self.exemplar_idx_to_label_map_func = np.vectorize(convert_idx_to_label)

        self.exemplar_label_to_idx = {
            value: key for key, value in self.exemplar_idx_to_label.items()
        }

        def convert_label_to_idx(label):
            return self.exemplar_label_to_idx[label]

        self.exemplar_label_to_idx_map_func = np.vectorize(convert_label_to_idx)

        self.exemplar_label_to_class_feature = {
            value: self.exemplar_classes_features[key].cpu().numpy()
            for key, value in self.exemplar_idx_to_label.items()
        }

        def convert_label_to_class_feature(label):
            return self.exemplar_label_to_class_feature[label]

        self.exemplar_label_to_class_feature_map_func = np.vectorize(
            convert_label_to_class_feature
        )

    # @torch.no_grad()
    def extend_memory(self, dataset, duplicate=10):
        if self.args.use_encoded_dataset:
            self.extend_memory_by_encoded_features(dataset)
        else:
            self.extend_memory_by_raw_images(dataset, duplicate)

    # @torch.no_grad()
    def extend_memory_by_encoded_features(self, dataset):
        dl = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

        # self.get_idx_to_class(dataset.idx_to_class_with_prompt)
        self.extract_exemplar_info(dataset)
        self.extract_dataset_info(dataset)

        raw_images_buffer = []
        labels_buffer = []
        normalized_images_buffer = []
        normalized_label_features_buffer = []
        for images, labels, texts in tqdm(dl):
            images, labels, texts = (
                images.to(self.device),
                labels.to(self.device),
                texts.to(self.device),
            )
            raw_images_buffer.append(images)
            labels_buffer.append(labels)
            normalized_images = images / images.norm(dim=-1, keepdim=True)
            normalized_texts = texts / texts.norm(dim=-1, keepdim=True)
            normalized_images_buffer.append(normalized_images)
            normalized_label_features_buffer.append(normalized_texts)

        raw_images_buffer = torch.cat(raw_images_buffer, dim=0)
        labels_buffer = torch.cat(labels_buffer, dim=0)
        normalized_images_buffer = torch.cat(normalized_images_buffer, dim=0)
        normalized_label_features_buffer = torch.cat(
            normalized_label_features_buffer, dim=0
        )

        if self.args.retriever == "tree_probe":
            train_features_lr = raw_images_buffer.cpu().numpy()
            train_labels_lr = labels_buffer.cpu().numpy()
            # Fit the tree clustering
            print("Fitting the tree clustering...")
            self.tree_clustering.fit(train_features_lr, train_labels_lr)
            print("Training the classifiers...")
            # start_time = time.time()
            self.tree_clustering.train_classifiers()
            # total_time = time.time() - start_time
            # with open(os.path.join(self.args.results_dir, "training_time.txt"), "a") as outfile:
            #     outfile.write(f"Total time: {total_time}\n")
            print("Getting the classifiers...")
            (
                self.linear_classifiers,
                self.classifier_labels,
                self.centroids,
            ) = self.tree_clustering.get_classifiers()
            self.centroids = np.array(self.centroids)
            self.centroids = torch.from_numpy(self.centroids).to(self.device).float()
            # Normalize the centroids
            self.centroids = self.centroids / self.centroids.norm(dim=-1, keepdim=True)

        if self.sample_buffer is None:
            self.sample_buffer = {
                "raw_image_features": raw_images_buffer.cpu(),
                "image_features": normalized_images_buffer.cpu(),
                "label_features": normalized_label_features_buffer.cpu(),
                "labels": labels_buffer.cpu(),
            }
        else:
            self.sample_buffer["raw_image_features"] = torch.cat(
                [self.sample_buffer["raw_image_features"], raw_images_buffer.cpu()],
                dim=0,
            )
            self.sample_buffer["image_features"] = torch.cat(
                [self.sample_buffer["image_features"], normalized_images_buffer.cpu()],
                dim=0,
            )
            self.sample_buffer["label_features"] = torch.cat(
                [
                    self.sample_buffer["label_features"],
                    normalized_label_features_buffer.cpu(),
                ],
                dim=0,
            )
            self.sample_buffer["labels"] = torch.cat(
                [self.sample_buffer["labels"], labels_buffer.cpu()], dim=0
            )

        if self.args.retriever == "kmeans_linear":
            # we need to assign none to self.linear_classifiers and retrain it
            self.linear_classifiers = []
            self.kmeans_model = KMeans(
                n_clusters=self.args.num_clusters, random_state=0
            )
            self.kmeans_model.fit(self.sample_buffer["image_features"].cpu().numpy())
            # for each cluster, we would want to train a linear classifier
            # print("total number of samples", len(self.sample_buffer["labels"]), "total number of clusters", self.args.num_clusters)
            for i in range(self.args.num_clusters):
                # np.where returns a tuple, so we select [0] as indices
                cluster_indices = np.where(self.kmeans_model.labels_ == i)[0]
                cluster_labels = self.sample_buffer["labels"][cluster_indices]
                cluster_features = self.sample_buffer["raw_image_features"][
                    cluster_indices
                ]
                # print("cluster", i, "has", len(cluster_indices), "samples")
                classifier = LogisticRegression(random_state=0, C=0.316, max_iter=5000)
                classifier.fit(
                    cluster_features.cpu().numpy(), cluster_labels.cpu().numpy()
                )
                self.linear_classifiers.append(classifier)

        # Find mapping from labels to label features
        self.label_to_label_features = {}
        self.num_of_samples_per_class_in_exemplar = []
        for label in self.exemplar_label_to_idx.keys():
            identical_label_features = self.sample_buffer["label_features"][
                self.sample_buffer["labels"] == label
            ]
            if identical_label_features.shape[0] > 0:
                self.label_to_label_features[label] = identical_label_features[0]
            self.num_of_samples_per_class_in_exemplar.append(
                torch.sum(self.sample_buffer["labels"] == label).item()
            )
        self.exemplar_classes_features = torch.stack(
            list(self.label_to_label_features.values()), dim=0
        ).float()
        self.exemplar_classes = list(self.exemplar_idx_to_class.values())

        if self.args.retriever == "linear":
            self.linear_classifier = LogisticRegression(
                random_state=0, C=0.316, max_iter=1000
            )
            self.linear_classifier.fit(
                self.sample_buffer["raw_image_features"].cpu().numpy(),
                self.sample_buffer["labels"].cpu().numpy(),
            )

        self.build_retriever()

    def fit_memory(self, raw_images_buffer, labels_buffer):
        start_time = time.time()
        if self.args.retriever == "tree_probe":
            train_features_lr = raw_images_buffer.cpu().numpy()
            train_labels_lr = labels_buffer.cpu().numpy()
            # Fit the tree clustering
            print("Fitting the tree clustering...")
            self.tree_clustering.fit(train_features_lr, train_labels_lr)
            print("Training the classifiers...")
            self.tree_clustering.train_classifiers()
            print("Getting the classifiers...")
            (
                self.linear_classifiers,
                self.classifier_labels,
                self.centroids,
            ) = self.tree_clustering.get_classifiers()
            self.centroids = np.array(self.centroids)
            self.centroids = torch.from_numpy(self.centroids).to(self.device).float()
            # Normalize the centroids
            self.centroids = self.centroids / self.centroids.norm(dim=-1, keepdim=True)

        end_time = time.time()
        fit_time = end_time - start_time

        # Find mapping from labels to label features
        self.label_to_label_features = {}
        self.num_of_samples_per_class_in_exemplar = []
        for label in self.exemplar_label_to_idx.keys():
            identical_label_features = self.sample_buffer["label_features"][
                self.sample_buffer["labels"] == label
            ]
            if identical_label_features.shape[0] > 0:
                self.label_to_label_features[label] = identical_label_features[0]
            self.num_of_samples_per_class_in_exemplar.append(
                torch.sum(self.sample_buffer["labels"] == label).item()
            )
        self.exemplar_classes_features = torch.stack(
            list(self.label_to_label_features.values()), dim=0
        ).float()
        self.exemplar_classes = list(self.exemplar_idx_to_class.values())

        return fit_time

    # For memory_module, we would want to maintain the unified class mappings in the buffer
    # @torch.no_grad()
    def extend_memory_by_raw_images(self, knn_dataset, duplicate=10):
        self.runtime_get_dim()
        labels_buffer = []
        image_features_buffer = []
        label_features_buffer = []
        raw_train_features = []
        raw_train_labels = []

        # drop last should be disabled since dropping last batch would cause severe performance drop on small datasets
        dl = DataLoader(
            knn_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

        # idx_to_class_in_buffer = self.get_idx_to_class(knn_dataset.idx_to_class)

        self.extract_exemplar_info(knn_dataset)

        cur_label_features = copy.deepcopy(
            self.encode_class_features(knn_dataset.classes)
        )

        for images, labels, texts in tqdm(dl):
            images, labels = images.to(self.device), labels.to(self.device)
            # map the labels to the buffer
            labels = torch.tensor(
                [self.exemplar_idx_to_label[idx.item()] for idx in labels]
            ).to(self.device)
            texts = clip.tokenize(list(texts)).to(self.device)
            image_features_raw = self.clip_encoder.encode_image(images)
            label_features_raw = self.clip_encoder.encode_text(texts)
            image_features = image_features_raw / image_features_raw.norm(
                dim=-1, keepdim=True
            )
            label_features = label_features_raw / label_features_raw.norm(
                dim=-1, keepdim=True
            )
            image_features_buffer.append(image_features)
            labels_buffer.append(labels)
            label_features_buffer.append(label_features)
            raw_train_features.append(image_features_raw)
            raw_train_labels.append(labels)

        # get unnormalized label features
        unnormalized_label_features = []
        with torch.no_grad():
            unnormalized_label_features.extend(
                self.clip_encoder.encode_text(clip.tokenize([c]).to(self.device))
                for c in knn_dataset.classes
            )
        unnormalized_label_features = torch.cat(unnormalized_label_features, dim=0)

        # Add label features to the buffer, duplicate multiple times
        current_label_features = torch.cat([cur_label_features] * duplicate, dim=0)
        raw_train_features.append(
            torch.cat(duplicate * [unnormalized_label_features], dim=0)
        )
        raw_train_labels.append(
            torch.cat(
                duplicate * [torch.arange(len(self.class_features)).to(self.device)]
            )
        )
        image_features_buffer.append(current_label_features)
        labels_buffer.append(
            torch.cat(
                [torch.arange(len(knn_dataset.classes)).to(self.device)] * duplicate,
                dim=0,
            )
        )
        label_features_buffer.append(current_label_features)

        if self.args.retriever == "tree_probe":
            train_features_lr = torch.cat(raw_train_features).cpu().numpy()
            train_labels_lr = torch.cat(raw_train_labels).cpu().numpy()
            self.tree_clustering.fit(train_features_lr, train_labels_lr)
            self.tree_clustering.train_classifiers()
            (
                self.linear_classifiers,
                self.classifier_labels,
                self.centroids,
            ) = self.tree_clustering.get_classifiers()
            self.centroids = np.array(self.centroids)
            self.centroids = torch.from_numpy(self.centroids).to(self.device).float()
            # Normalize the centroids
            self.centroids = self.centroids / self.centroids.norm(dim=-1, keepdim=True)

        if self.sample_buffer is None:
            self.sample_buffer = {
                "image_features": torch.cat(image_features_buffer, dim=0).cpu(),
                "raw_image_features": torch.cat(raw_train_features, dim=0).cpu(),
                "labels": torch.cat(labels_buffer, dim=0).cpu(),
                "label_features": torch.cat(label_features_buffer, dim=0).cpu(),
            }
        else:
            image_features = torch.cat(
                [
                    self.sample_buffer["image_features"],
                    torch.cat(image_features_buffer, dim=0).cpu(),
                ],
                dim=0,
            )
            labels = torch.cat(
                [self.sample_buffer["labels"], torch.cat(labels_buffer, dim=0).cpu()],
                dim=0,
            )
            label_features = torch.cat(
                [
                    self.sample_buffer["label_features"],
                    torch.cat(label_features_buffer, dim=0).cpu(),
                ],
                dim=0,
            )
            raw_image_features = torch.cat(
                [
                    self.sample_buffer["raw_image_features"],
                    torch.cat(raw_train_features, dim=0).cpu(),
                ],
                dim=0,
            )

            # self.sample_buffer = {"image_features": image_features, "labels": labels, "label_features": label_features}
            self.sample_buffer.update({"image_features": image_features})
            self.sample_buffer.update({"raw_image_features": raw_image_features})
            self.sample_buffer.update({"labels": labels})
            self.sample_buffer.update({"label_features": label_features})

        # update the label features
        cur_label_features = []
        for class_name in self.exemplar_idx_to_class.values():
            clip_class_token = self.clip_encoder.encode_text(
                clip.tokenize([class_name]).to(self.device)
            )
            clip_class_token /= clip_class_token.norm(dim=-1, keepdim=True)
            cur_label_features.append(clip_class_token)
        self.exemplar_classes_features = torch.cat(cur_label_features, dim=0)
        self.exemplar_classes = list(self.exemplar_idx_to_class.values())

        self.build_retriever()

    def find_mapping_from_exemplar_to_target(self):
        # Find the mapping from exemplar classes to target classes and store it in self.exemplar_to_target, which is a dict, key presents exemplar class label, value represents target class label, -1 represents no mapping
        assert self.exemplar_classes is not None
        assert self.target_classes is not None
        self.exemplar_to_target = {}
        for i, exemplar_class in enumerate(self.exemplar_classes):
            if exemplar_class in self.target_classes:
                self.exemplar_to_target[i] = self.target_classes.index(exemplar_class)
            else:
                self.exemplar_to_target[i] = -1

    def compute_overlapping_indices(self):
        if overlap_indices := [
            (i, j)
            for i, item in enumerate(self.exemplar_classes)
            for j, other_item in enumerate(self.target_classes)
            if item == other_item
        ]:
            self.overlapping_exemplar_indices, self.overlapping_target_indices = zip(
                *overlap_indices
            )
        else:
            self.overlapping_exemplar_indices = []
            self.overlapping_target_indices = []

    # This should be called in engine
    def set_calibration_features(self, target_classes, target_features, unique_labels):
        self.target_classes_features = target_features.float()
        self.target_classes = target_classes
        # self.target_unique_labels is a numpy array
        self.target_unique_labels = unique_labels

    def compute_p_cases(self, query):
        # print("In compute_p_cases...")
        # print(self.target_classes_features.shape)
        query_target_label_sim = (
            100
            * self.target_classes_features.unsqueeze(0).repeat(query.shape[0], 1, 1)
            @ query.unsqueeze(-1)
        ).squeeze()  # B x N
        mask_qt = torch.zeros_like(query_target_label_sim).to(self.device)
        mask_qt[:, self.overlapping_target_indices] = 1
        p_case_1 = (mask_qt * query_target_label_sim.softmax(-1)).sum(
            dim=-1, keepdim=True
        )  # B x 1
        p_case_2 = 1 - p_case_1
        return p_case_1, p_case_2, query_target_label_sim, mask_qt

    def compute_knn_p_e_case_1(self, label_indices):
        # convert label indices to exemplar indices using self.exemplar_label_to_idx
        exemplar_indices = self.exemplar_label_to_idx_map_func(
            label_indices.cpu().numpy()
        )
        bin_counts = []
        for batch_idx in range(label_indices.shape[0]):
            bin_counts.append(
                np.bincount(
                    exemplar_indices[batch_idx], minlength=len(self.exemplar_classes)
                )
            )
        bin_counts = np.stack(bin_counts)
        # bin_counts = np.bincount(exemplar_indices, minlength=len(self.exemplar_classes))
        bin_counts_target = np.zeros((label_indices.shape[0], len(self.target_classes)))
        bin_counts_target[:, self.overlapping_target_indices] = bin_counts[
            :, self.overlapping_exemplar_indices
        ]
        bin_counts_target /= bin_counts_target.sum(axis=-1, keepdims=True) + 1e-8
        return torch.from_numpy(bin_counts_target).to(self.device).float()

    def compute_p_z_cases(self, query_target_label_sim, mask_qt):
        p_z_case_1 = query_target_label_sim * mask_qt
        p_z_case_1 /= p_z_case_1.sum(dim=-1, keepdim=True) + 1e-8  # B x N
        p_z_case_2 = query_target_label_sim * (1 - mask_qt)
        p_z_case_2 /= p_z_case_2.sum(dim=-1, keepdim=True) + 1e-8  # B x N
        return p_z_case_1, p_z_case_2

    def knn_forward(self, clip_feature_raw):
        clip_feature_norm = (
            clip_feature_raw / clip_feature_raw.norm(dim=-1, keepdim=True).float()
        )
        _, indices = self.knn_searcher.search(clip_feature_norm.cpu().numpy(), self.k)
        indices_batched = torch.from_numpy(indices)
        indices = indices_batched.flatten()

        if self.buffer_type == "exemplar_only":
            label_out = (
                self.sample_buffer["labels"][indices]
                .reshape(clip_feature_raw.shape[0], self.k)
                .to(self.device)
            )
            assert (
                self.exemplar_classes is not None
            ), "Need exemplar classes for this mode"
            p_e_case_1 = self.compute_knn_p_e_case_1(label_out)
            class_index = p_e_case_1.argmax(dim=-1)
            text_out = self.target_classes_features[class_index]
            return text_out
        elif self.buffer_type == "avg_prob":
            label_out = (
                self.sample_buffer["labels"][indices]
                .reshape(clip_feature_raw.shape[0], self.k)
                .to(self.device)
            )
            assert (
                self.exemplar_classes is not None
            ), "Need exemplar classes for calibration"
            p_case_1, p_case_2, query_target_label_sim, mask_qt = self.compute_p_cases(
                clip_feature_norm
            )
            p_e_case_1 = self.compute_knn_p_e_case_1(label_out)
            p_z = query_target_label_sim.softmax(dim=-1)
            final_p = (p_e_case_1 + p_z) / 2
            class_index = final_p.argmax(dim=-1)
            # return the label features of the class with the highest probability
            text_out = self.target_classes_features[class_index]
            return text_out
        elif self.buffer_type == "aim_prob":
            label_out = (
                self.sample_buffer["labels"][indices]
                .reshape(clip_feature_raw.shape[0], self.k)
                .to(self.device)
            )
            assert (
                self.exemplar_classes is not None
            ), "Need exemplar classes for calibration"
            p_case_1, p_case_2, query_target_label_sim, mask_qt = self.compute_p_cases(
                clip_feature_norm
            )
            p_z_case_1_orig, p_z_case_2 = self.compute_p_z_cases(
                query_target_label_sim, mask_qt
            )
            p_e_case_1 = self.compute_knn_p_e_case_1(label_out)

            p_z_case_1 = p_e_case_1 * p_z_case_1_orig
            p_z_case_1 = p_z_case_1 / (p_z_case_1.sum(dim=-1, keepdim=True) + 1e-8)

            p_x_case_1 = p_case_1 * p_z_case_1  # B x N
            p_x_case_2 = p_case_2 * p_z_case_2  # B x N

            final_p = p_x_case_1 + p_x_case_2
            class_index = final_p.argmax(dim=-1)
            # return the label features of the class with the highest probability
            text_out = self.target_classes_features[class_index]
            return text_out
        elif self.buffer_type == "aim_emb":
            text_out = (
                self.sample_buffer["label_features"][indices]
                .reshape(clip_feature_raw.shape[0], self.k, -1)
                .to(self.device)
            )
            image_out = (
                self.sample_buffer["image_features"][indices]
                .reshape(clip_feature_raw.shape[0], self.k, -1)
                .to(self.device)
            )
            label_out = (
                self.sample_buffer["labels"][indices]
                .reshape(clip_feature_raw.shape[0], self.k)
                .to(self.device)
            )
            assert (
                self.exemplar_classes is not None
            ), "Need exemplar classes for calibration"
            attention_weight = torch.matmul(
                clip_feature_norm.unsqueeze(1).float().to(self.device),
                image_out.transpose(1, 2).float(),
            )  # B x 1 x N
            attention_weight = (attention_weight).softmax(dim=-1)  # B x 1 x N
            text_out = torch.bmm(attention_weight, text_out.float()).squeeze()  # B x D
            text_out /= text_out.norm(dim=-1, keepdim=True)

            p_case_1, p_case_2, query_target_label_sim, mask_qt = self.compute_p_cases(
                clip_feature_norm.to(self.device)
            )
            out = p_case_1 * text_out + p_case_2 * clip_feature_norm.to(self.device)
            return out
        else:
            raise NotImplementedError

    def compute_cluster_p_e_case_1(self, query, clusters, classifiers):
        # query: B x C, C is the feature dimension, query should be unnormalized
        # clusters: B x K, K is the number of retrieved samples
        # classifiers: list of classifiers
        # return: B x N, where N is the number of target classes
        if self.overlapping_exemplar_indices == []:
            return torch.zeros(query.shape[0], len(self.target_classes_features)).to(
                self.device
            )
        total_probs = []
        total_masks = []
        overlapping_exemplar_indices = np.array(self.overlapping_exemplar_indices)
        overlapping_target_indices = np.array(self.overlapping_target_indices)
        num_of_samples_per_class_in_exemplar = np.array(
            self.num_of_samples_per_class_in_exemplar
        )
        min_num_of_samples_per_class_in_exemplar = np.minimum(
            num_of_samples_per_class_in_exemplar, self.k
        )

        for clf in classifiers:
            assert clf is not None
            # create masks and probs
            exemplar_prob = np.zeros((query.shape[0], len(self.exemplar_classes)))
            exemplar_mask = np.zeros((len(self.exemplar_classes)))
            target_mask = np.zeros(len(self.target_classes_features))
            target_prob = np.zeros((query.shape[0], len(self.target_classes_features)))

            prob = clf.predict_proba(query.cpu().numpy())
            classes_of_clf = clf.classes_
            classes_of_clf_numpy = np.array(classes_of_clf)
            classes_of_clf_in_exemplar = self.exemplar_label_to_idx_map_func(
                classes_of_clf_numpy
            )

            exemplar_mask[classes_of_clf_in_exemplar] = 1
            target_mask[overlapping_target_indices] = exemplar_mask[
                overlapping_exemplar_indices
            ]
            total_masks.append(target_mask)

            exemplar_prob[:, classes_of_clf_in_exemplar] = prob
            exemplar_prob = exemplar_prob[:, overlapping_exemplar_indices]
            # TODO: renormalize the probabilities to sum to 1, this is to ensure reliable confidence over overlapping classes
            exemplar_prob /= (exemplar_prob.sum(axis=-1, keepdims=True)) + 1e-8
            target_prob[:, overlapping_target_indices] = exemplar_prob
            target_prob[
                :, overlapping_target_indices
            ] /= min_num_of_samples_per_class_in_exemplar[overlapping_exemplar_indices]
            total_probs.append(target_prob)

        total_probs = np.stack(total_probs, axis=0)  # N_CLS x B x C
        total_masks = np.stack(total_masks, axis=0)  # N_CLS x C
        total_masks = total_masks.sum(axis=0) + 1e-8  # C
        # for each sample in the batch, we calculate the number of occurrences of each class
        occurrences = []
        for b in range(query.shape[0]):
            occurrence = np.bincount(
                clusters[b].cpu().numpy(), minlength=len(classifiers)
            )
            occurrences.append(occurrence)
        occurrences = np.stack(occurrences, axis=0)  # B x N_CLS
        # multiply the occurrences with the probabilities
        total_probs = total_probs.transpose(1, 0, 2)  # B x N_CLS x C
        final_probs = total_probs * occurrences[:, :, np.newaxis]  # B x N_CLS x C
        final_probs = final_probs.sum(axis=1)  # B x C
        # TODO: This is to ensure class probabilities are balanced across classes since some classes may have more classifiers than others
        final_probs /= total_masks[np.newaxis, :]
        return torch.from_numpy(final_probs).to(self.device).float()

    def tree_probe_forward(self, clip_feature_raw):
        clip_feature_norm = (
            clip_feature_raw / clip_feature_raw.norm(dim=-1, keepdim=True).float()
        )

        _, indices = self.knn_searcher.search(clip_feature_norm.cpu().numpy(), self.k)
        indices_batched = torch.from_numpy(indices)
        # shuffle the indices
        # indices_batched[:, :] = indices_batched[:, :][:, torch.randperm(indices_batched.shape[1])]
        indices = indices_batched.flatten()

        image_out = (
            self.sample_buffer["image_features"][indices]
            .reshape(clip_feature_raw.shape[0], self.k, -1)
            .to(self.device)
        )
        # calculate the closest cluster to self.cluster_centers
        dist = torch.cdist(
            image_out.float(),
            self.centroids.unsqueeze(0).repeat(image_out.shape[0], 1, 1).float(),
        )
        clusters = torch.argmin(dist, dim=-1)  # B x K

        if self.buffer_type == "exemplar_only":
            image_out, text_out = self.cluster_probe_forward_segment(
                clip_feature_raw, clusters, image_out
            )
            text_out = torch.mean(text_out, dim=1)
            text_out /= text_out.norm(dim=-1, keepdim=True).float()
            return text_out
        elif self.buffer_type == "avg_prob":
            # This mode averages probabilities from exemplars and zero-shot predictions
            # It uses a fixed 50-50 split between exemplar-based and zero-shot probabilities
            p_case_1, p_case_2, query_target_label_sim, mask_qt = self.compute_p_cases(
                clip_feature_norm
            )
            p_z_case_1, p_z_case_2 = self.compute_p_z_cases(
                query_target_label_sim, mask_qt
            )
            p_e_case_1 = self.compute_cluster_p_e_case_1(
                clip_feature_raw, clusters, self.linear_classifiers
            )

            p_z_case_1 = p_e_case_1 * p_z_case_1
            p_z_case_1 = p_z_case_1 / (p_z_case_1.sum(dim=-1, keepdim=True) + 1e-8)

            p_x_case_1 = 0.5 * p_z_case_1  # B x N
            p_x_case_2 = 0.5 * p_z_case_2  # B x N

            final_p = p_x_case_1 + p_x_case_2

            class_index = final_p.argmax(dim=-1)
            return self.target_classes_features[class_index]
        elif self.buffer_type == "avg_emb":
            # This mode averages embeddings from retrieved exemplars and the input feature
            # It uses attention mechanism to weight the retrieved exemplars
            text_out = (
                self.sample_buffer["label_features"][indices]
                .reshape(clip_feature_raw.shape[0], self.k, -1)
                .to(self.device)
            )
            image_out = (
                self.sample_buffer["image_features"][indices]
                .reshape(clip_feature_raw.shape[0], self.k, -1)
                .to(self.device)
            )
            assert (
                self.exemplar_classes is not None
            ), "Need exemplar classes for calibration"

            _, text_out = self.cluster_probe_forward_segment(
                clip_feature_raw, clusters, image_out
            )
            attention_weight = torch.matmul(
                clip_feature_norm.unsqueeze(1).float(),
                image_out.transpose(1, 2).float(),
            )  # B x 1 x N
            attention_weight = (attention_weight).softmax(dim=-1)  # B x 1 x N
            text_out = torch.bmm(attention_weight, text_out.float()).squeeze()  # B x D
            text_out /= text_out.norm(dim=-1, keepdim=True).float()
            return 0.5 * text_out + 0.5 * clip_feature_norm
        elif self.buffer_type == "aim_prob":
            # This mode is similar to "avg_prob" but uses adaptive importance mixing (AIM)
            # It dynamically adjusts the balance between exemplar-based and zero-shot probabilities
            p_case_1, p_case_2, query_target_label_sim, mask_qt = self.compute_p_cases(
                clip_feature_norm
            )
            p_z_case_1, p_z_case_2 = self.compute_p_z_cases(
                query_target_label_sim, mask_qt
            )
            p_e_case_1 = self.compute_cluster_p_e_case_1(
                clip_feature_raw, clusters, self.linear_classifiers
            )

            p_z_case_1 = p_e_case_1 * p_z_case_1
            p_z_case_1 = p_z_case_1 / (p_z_case_1.sum(dim=-1, keepdim=True) + 1e-8)

            p_x_case_1 = p_case_1 * p_z_case_1  # B x N
            p_x_case_2 = p_case_2 * p_z_case_2  # B x N

            final_p = p_x_case_1 + p_x_case_2

            class_index = final_p.argmax(dim=-1)
            return self.target_classes_features[class_index]
        elif self.buffer_type == "aim_emb":
            # This mode is similar to "avg_emb" but uses adaptive importance mixing (AIM)
            # It dynamically adjusts the balance between retrieved exemplar embeddings and input feature
            text_out = (
                self.sample_buffer["label_features"][indices]
                .reshape(clip_feature_raw.shape[0], self.k, -1)
                .to(self.device)
            )
            image_out = (
                self.sample_buffer["image_features"][indices]
                .reshape(clip_feature_raw.shape[0], self.k, -1)
                .to(self.device)
            )
            assert (
                self.exemplar_classes is not None
            ), "Need exemplar classes for calibration"
            p_case_1, p_case_2, query_target_label_sim, mask_qt = self.compute_p_cases(
                clip_feature_norm
            )
            _, text_out = self.cluster_probe_forward_segment(
                clip_feature_raw, clusters, image_out
            )
            attention_weight = torch.matmul(
                clip_feature_norm.unsqueeze(1).float(),
                image_out.transpose(1, 2).float(),
            )  # B x 1 x N
            attention_weight = (attention_weight).softmax(dim=-1)  # B x 1 x N
            text_out = torch.bmm(attention_weight, text_out.float()).squeeze()  # B x D
            text_out /= text_out.norm(dim=-1, keepdim=True).float()
            return p_case_1 * text_out + p_case_2 * clip_feature_norm
        else:
            raise NotImplementedError

    def cluster_probe_forward_segment(self, clip_feature_raw, clusters, image_out):
        # This function performs classification using linear classifiers and maps the results to class features

        pred_text = []
        for i, clf in enumerate(self.linear_classifiers):
            if clf is not None:
                # Use the classifier to predict labels for the input features
                prediction = clf.predict(clip_feature_raw.cpu().numpy())
                # Map the predicted labels to their corresponding class features
                x = [self.exemplar_label_to_class_feature_map_func(p) for p in prediction]
                x = np.array(x)
                pred_text.append(x)
            else:
                # If no classifier is available, use a default label
                x = self.exemplar_label_to_class_feature_map_func(self.classifier_labels[i][0])
                # Create a batch of identical features
                x = np.array([x])
                x = x.repeat(clip_feature_raw.shape[0], axis=0)
                pred_text.append(x)

        assert len(pred_text) > 0  # Ensure we have predictions

        try:
            # Convert predictions to a tensor and reshape
            new_pred_text = torch.from_numpy(np.array(pred_text)).transpose(0, 1).to(self.device)
        except:
            # Debug information in case of error
            for pt in pred_text:
                print("pt", pt.shape, pt.dtype)
            raise TypeError

        # Select predictions based on cluster assignments
        pred_text_out = [new_pred_text[b][clusters[b]] for b in range(new_pred_text.shape[0])]
        
        # Return both image features and text predictions
        return (
            image_out.float(),
            torch.stack(pred_text_out, dim=0).to(self.device).float(),
        )

    def cluster_calibration_forward_segment(
        self, clip_feature_norm, clip_feature_raw, clusters
    ):
        # This function performs calibrated classification by combining exemplar-based and zero-shot predictions

        # Compute probabilities for different cases
        p_case_1, p_case_2, query_target_label_sim, mask_qt = self.compute_p_cases(clip_feature_norm)
        
        # Compute zero-shot probabilities
        p_z_case_1, p_z_case_2 = self.compute_p_z_cases(query_target_label_sim, mask_qt)
        
        # Compute exemplar-based probabilities
        p_e_case_1 = self.compute_cluster_p_e_case_1(clip_feature_raw, clusters, self.linear_classifiers)

        # Combine exemplar-based and zero-shot probabilities
        p_z_case_1 = p_e_case_1 * p_z_case_1
        p_z_case_1 = p_z_case_1 / (p_z_case_1.sum(dim=-1, keepdim=True) + 1e-8)  # Normalize

        # Compute final probabilities for each case
        p_x_case_1 = p_case_1 * p_z_case_1  # B x N
        p_x_case_2 = p_case_2 * p_z_case_2  # B x N

        # Combine probabilities from both cases
        final_p = p_x_case_1 + p_x_case_2
        
        # Select the class with highest probability
        class_index = final_p.argmax(dim=-1)
        
        # Return the feature vectors for the selected classes
        return self.target_classes_features[class_index]

    def linear_probe_with_zero_shot_forward(self, clip_feature_raw):
        prob = self.linear_classifier.predict_proba(clip_feature_raw.cpu().numpy())
        # combining with zero-shot
        classes_in_clf = self.linear_classifier.classes_
        mask = np.isin(self.target_unique_labels, classes_in_clf)
        out_prob = np.zeros((prob.shape[0], len(self.target_classes)))
        out_prob[:, mask] = prob
        clip_feature_norm = clip_feature_raw / clip_feature_raw.norm(
            dim=-1, keepdim=True
        )
        zero_shot_similarity = (
            100
            * self.target_classes_features.unsqueeze(0).repeat(
                clip_feature_norm.shape[0], 1, 1
            )
            @ clip_feature_norm.unsqueeze(-1)
        ).squeeze()  # B x N
        zero_shot_prob = torch.softmax(zero_shot_similarity, dim=-1)
        out_prob[:, ~mask] = zero_shot_prob.cpu().numpy()[:, ~mask]
        out = torch.argmax(torch.from_numpy(out_prob), dim=-1)  # B
        out = self.target_classes_features[out]
        return out.to(self.device)

    def forward(self, clip_feature):
        if self.args.retriever == "knn":
            return self.knn_forward(clip_feature)
        elif self.args.retriever == "tree_probe":
            return self.tree_probe_forward(clip_feature)
        else:
            raise NotImplementedError
