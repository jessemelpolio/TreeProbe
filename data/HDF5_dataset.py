import torch
import numpy as np
import random
from .combined_dataset import deal_with_dataset
import h5py
import os
import bisect
import itertools


class ContinualHDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, datasets, name_list):
        assert isinstance(datasets, list) and isinstance(datasets[0], dict), "datasets should be a list of dicts"

        self.name = "_".join(name_list)
        self.datasets = datasets
        self.classes = []
        self.classes_with_prompt = []
        self.image_features = []
        self.label_features = []
        self.labels = []
        self.dataset_sizes = []
        # self.breakpoints = [0]

        class_to_global_idx = {}

        for dataset in datasets:
            original_classes = [cls.decode('utf-8') for cls in dataset['original_classes']]
            classes_with_prompt = [cls.decode('utf-8') for cls in dataset['classes']]

            dataset_classes = []
            dataset_classes_with_prompt = []
            class_remap = {}

            for idx, (cls, cls_with_prompt) in enumerate(zip(original_classes, classes_with_prompt)):
                if cls not in class_to_global_idx:
                    global_idx = len(self.classes)
                    class_to_global_idx[cls] = global_idx
                    self.classes.append(cls)
                    self.classes_with_prompt.append(cls_with_prompt)
                    dataset_classes.append(cls)
                    dataset_classes_with_prompt.append(cls_with_prompt)

                class_remap[idx] = class_to_global_idx[cls]

            # self.breakpoints.append(len(self.classes))

            self.image_features.append(dataset['image_features'])
            self.label_features.append(dataset['label_features'])

            dataset_labels = np.array([class_remap[label] for label in dataset['labels']])
            self.labels.append(dataset_labels)
            self.dataset_sizes.append(len(dataset_labels))

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        self.idx_to_class_with_prompt = {idx: cls for idx, cls in enumerate(self.classes_with_prompt)}

        self.extract_unique_class_features()
        self.cumulative_sizes = self.cumsum(self.dataset_sizes)

    def extract_unique_class_features(self):
        label_features = np.concatenate(self.label_features)
        labels = np.concatenate(self.labels)
        self.unique_class_features = np.zeros((len(self.classes), self.label_features[0].shape[1]))
        for idx in range(len(self.classes)):
            indices = np.where(labels == idx)[0]
            if len(indices) > 0:
                self.unique_class_features[idx] = label_features[indices[0]]

    @staticmethod
    def cumsum(sequence):
        return list(itertools.accumulate(sequence))

    def __len__(self):
        return sum(self.dataset_sizes)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return (
            torch.from_numpy(self.image_features[dataset_idx][sample_idx]),
            torch.tensor(self.labels[dataset_idx][sample_idx]).long(),
            torch.from_numpy(self.label_features[dataset_idx][sample_idx]),
        )


class ConcatHDF5Dataset(ContinualHDF5Dataset):
    def __init__(self, datasets, name_list):
        super().__init__(datasets, name_list)
        self.concat_dataset()
        self.subset_item_indices = list(range(self.dataset_size))

    def concat_dataset(self):
        self.concat_image_features = np.concatenate(self.image_features)
        self.concat_label_features = np.concatenate(self.label_features)
        self.concat_labels = np.concatenate(self.labels)
        self.dataset_size = len(self.concat_labels)

    def __len__(self):
        return len(self.subset_item_indices)

    def __getitem__(self, index):
        index = self.subset_item_indices[index]
        return (
            torch.from_numpy(self.concat_image_features[index]),
            torch.tensor(self.concat_labels[index]).long(),
            torch.from_numpy(self.concat_label_features[index]),
        )


class ZeroShotHDF5Dataset(ContinualHDF5Dataset):
    def __init__(self, datasets, name_list, num_classes=100, padding=0):
        super().__init__(datasets, name_list)

        assert num_classes <= len(
            self.classes
        ), "num_classes should be smaller than the number of classes in the dataset"
        random.seed(0)
        self.padding = padding
        self.random_classes_indices = random.sample(
            list(range(len(self.classes))), num_classes
        )
        self.classes = [self.idx_to_class[i] for i in self.random_classes_indices]
        self.zero_shot_classes = self.classes
        self.classes_with_prompt = [
            self.idx_to_class_with_prompt[i] for i in self.random_classes_indices
        ]
        self.zero_shot_classes_with_prompt = self.classes_with_prompt
        concat_labels = np.concatenate(self.labels)
        self.subset_item_indices = [
            idx
            for idx in range(self.cumulative_sizes[-1])
            if concat_labels[idx] in self.random_classes_indices
        ]
        self.remapping_label = {
            c: i
            for i, c in enumerate(self.random_classes_indices)
        }

    def update_padding(self, padding):
        self.padding = padding

    def __len__(self):
        return len(self.subset_item_indices)

    def __getitem__(self, index):
        subset_index = self.subset_item_indices[index]
        image, label, label_feature = super().__getitem__(subset_index)
        return (
            image,
            torch.tensor(self.remapping_label[label.item()] + self.padding).long(),
            label_feature,
        )
        # subset_index = self.subset_item_indices[index]
        # assert (
        #     self.concat_labels[subset_index] in self.random_classes_indices
        # ), "label should be in random_classes"
        # return (
        #     torch.from_numpy(self.concat_image_features[subset_index]),
        #     torch.tensor(
        #         self.remapping_label[self.concat_labels[subset_index]] + self.padding
        #     ).long(),
        #     torch.from_numpy(self.concat_label_features[subset_index]),
        # )


class MixHDF5Dataset(ConcatHDF5Dataset):
    def __init__(self, datasets, name_list, split=5, num_samples=100):
        super().__init__(datasets, name_list)

        random.seed(0)
        self.repermutation = random.sample(
            list(range(len(self.classes))), len(self.classes)
        )
        self.reverse_repermutation = {
            self.repermutation[i]: i for i in range(len(self.repermutation))
        }
        self.all_classes = [self.idx_to_class[i] for i in self.repermutation]
        self.all_classes_with_prompt = [
            self.idx_to_class_with_prompt[i] for i in self.repermutation
        ]
        self.subset_classes_length = len(self.classes) // split

        self.subset_class_indices = {}
        for task in range(split):
            self.subset_class_indices[task] = self.repermutation[
                task
                * self.subset_classes_length : (task + 1)
                * self.subset_classes_length
            ]

        sample_count = {}

        self.subset_item_indices = {}
        for task in range(split):
            self.subset_item_indices[task] = []
        for idx in range(self.dataset_size):
            task = (
                self.reverse_repermutation[self.concat_labels[idx]]
                // self.subset_classes_length
            )
            if task < split:
                assert (
                    self.concat_labels[idx] in self.subset_class_indices[task]
                ), "label should be in subset_class_indices"
                if self.concat_labels[idx] not in sample_count:
                    sample_count[self.concat_labels[idx]] = 0
                if sample_count[self.concat_labels[idx]] < num_samples:
                    sample_count[self.concat_labels[idx]] += 1
                    self.subset_item_indices[task].append(idx)

        self.stage = 0
        self.num_stages = split

        self.classes = [
            self.idx_to_class[i] for i in self.subset_class_indices[self.stage]
        ]
        self.classes_with_prompt = [
            self.idx_to_class_with_prompt[i]
            for i in self.subset_class_indices[self.stage]
        ]
        self.remapping_label = {
            self.subset_class_indices[self.stage][i]: i
            for i in range(len(self.subset_class_indices[self.stage]))
        }

        self.additional_classes = None
        self.additional_classes_with_prompt = None

    def add_additional_classes(self, additional_classes, additional_classes_with_prompt):
        self.additional_classes = additional_classes
        self.additional_classes_with_prompt = additional_classes_with_prompt

        if self.stage == 0:
            self.classes = self.classes + self.additional_classes
            self.classes_with_prompt = self.classes_with_prompt + self.additional_classes_with_prompt

    def forward_stage(self):
        self.stage += 1
        self.classes = self.all_classes[
                       self.stage
                       * self.subset_classes_length: (self.stage + 1)
                                                     * self.subset_classes_length
                       ]
        self.classes_with_prompt = self.all_classes_with_prompt[
                                   self.stage
                                   * self.subset_classes_length: (self.stage + 1)
                                                                 * self.subset_classes_length
                                   ]
        self.repermutation_subset = self.repermutation[
                                    self.stage
                                    * self.subset_classes_length: (self.stage + 1)
                                                                  * self.subset_classes_length
                                    ]
        self.remapping_label = {
            self.repermutation_subset[i]: i
            for i in range(len(self.repermutation_subset))
        }

        if self.additional_classes is not None:
            self.classes = self.classes + self.additional_classes
            self.classes_with_prompt = self.classes_with_prompt + self.additional_classes_with_prompt

        if self.stage == self.num_stages:
            print("Finish all stages")
            return

    def __len__(self):
        return len(self.subset_item_indices[self.stage])

    def __getitem__(self, index):
        subset_index = self.subset_item_indices[self.stage][index]
        assert (
            self.concat_labels[subset_index] in self.subset_class_indices[self.stage]
        ), "label should be in subset_class_indices"
        return (
            torch.from_numpy(self.concat_image_features[subset_index]),
            torch.tensor(self.remapping_label[self.concat_labels[subset_index]]).long(),
            torch.from_numpy(self.concat_label_features[subset_index]),
        )


class DataIncrementalHDF5Dataset(ConcatHDF5Dataset):
    def __init__(self, datasets, name_list, perc=0.02, incremental_type="double"):
        super().__init__(datasets, name_list)

        # build incremental list
        self.incremental_type = incremental_type
        self.perc_list = []
        if incremental_type == "double":
            incremental_perc = perc
            if perc > 1:
                perc = perc / 100
            self.perc_list.append(perc)
            while incremental_perc < 1:
                self.perc_list.append(perc)
                perc = perc * 2
                incremental_perc += perc
            self.perc_list.append(1 - sum(self.perc_list))
        elif incremental_type == "fixed":
            if perc > 1:
                perc = perc / 100
            self.perc_list = [perc] * int(1 / perc)
            if sum(self.perc_list) < 1:
                self.perc_list.append(1 - sum(self.perc_list))
        else:
            raise ValueError("incremental_type should be double or fixed")
        self.num_stages = len(self.perc_list)
        random.seed(0)
        self.random_indices = random.sample(range(self.dataset_size), self.dataset_size)
        self.learned_classes_indices = torch.tensor([], dtype=torch.long)
        self.init_stage()

    def init_stage(self):
        self.stage = 0
        self.__build_subset()

    def forward_stage(self):
        self.stage += 1
        if self.stage == self.num_stages:
            print("Finish all stages")
            return
        self.__build_subset()

    def __build_subset(self):
        start_idx = int(self.dataset_size * sum(self.perc_list[: self.stage]))
        end_idx = int(self.dataset_size * sum(self.perc_list[: self.stage + 1]))
        if self.stage == self.num_stages - 1:
            end_idx = self.dataset_size
        self.subset_item_indices = self.random_indices[start_idx:end_idx]
        self.__get_current_stage_classes()
        if self.stage != 0:
            self.__get_past_classes()
        self.__get_learned_classes()

    def __get_current_stage_classes(self):
        self.current_classes_indices = torch.unique(
            torch.from_numpy(self.concat_labels[self.subset_item_indices]), sorted=True
        )
        self.current_classes = [
            self.idx_to_class[i.item()] for i in self.current_classes_indices
        ]
        self.current_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.current_classes_indices
        ]
        self.current_classes_features = self.unique_class_features[
            self.current_classes_indices.numpy()
        ]

    def __get_past_classes(self):
        # we call __get_past_classes() before __get_learned_classes()
        self.past_classes_indices = self.learned_classes_indices
        self.past_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.past_labels = [
            self.class_to_idx[self.idx_to_class[i.item()]]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_features = self.unique_class_features[
            self.learned_classes_indices.numpy()
        ]

    def __get_learned_classes(self):
        learned_classes_indices_current_stage = torch.unique(
            torch.from_numpy(self.concat_labels[self.subset_item_indices]), sorted=True
        )
        self.learned_classes_indices = torch.cat(
            (self.learned_classes_indices, learned_classes_indices_current_stage)
        )
        self.learned_classes_indices = torch.unique(
            self.learned_classes_indices, sorted=True
        )
        self.learned_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.learned_labels = [
            self.class_to_idx[self.idx_to_class[i.item()]]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_features = self.unique_class_features[
            self.learned_classes_indices.numpy()
        ]


class ClassIncrementalHDF5Dataset(ConcatHDF5Dataset):
    def __init__(
        self, datasets, name_list, num_classes=100, inclusive=False, include_whole=False
    ):
        super().__init__(datasets, name_list)
        self.num_classes = num_classes
        self.num_stages = (
            len(self.classes) // num_classes
            if len(self.classes) % num_classes < num_classes / 2
            else len(self.classes) // num_classes + 1
            # if len(self.classes) % num_classes == 0
            # else len(self.classes) // num_classes + 1
        )
        # print("num_stages", self.num_stages)
        self.classes_mapping = list(range(len(self.classes)))
        random.seed(0)
        random.shuffle(self.classes_mapping)
        self.inclusive = inclusive
        self.include_whole = include_whole
        self.init_stage()

    def init_stage(self):
        self.stage = 0
        self.subset_classes = []
        self.__build_subset()

    def forward_stage(self):
        self.stage += 1
        if self.stage == self.num_stages:
            print("Finish all stages")
            return
        self.__build_subset()

    def __build_subset(self):
        start_idx = self.stage * self.num_classes
        end_idx = min((self.stage + 1) * self.num_classes, len(self.classes))
        if self.stage == self.num_stages - 1:
            end_idx = len(self.classes)

        if self.inclusive:
            self.subset_classes.extend(self.classes_mapping[start_idx:end_idx])
        else:
            self.subset_classes = self.classes_mapping[start_idx:end_idx]

        if self.include_whole:
            self.subset_item_indices = list(range(self.dataset_size))
        else:
            self.subset_item_indices = [
                idx
                for idx in range(self.dataset_size)
                if self.concat_labels[idx] in self.subset_classes
            ]
        self.__get_current_stage_classes(start_idx, end_idx)
        if self.stage != 0:
            self.__get_past_classes()
        self.__get_learned_classes(end_idx)

    def __get_current_stage_classes(self, start_idx, end_idx):
        self.current_classes_indices = torch.sort(
            torch.tensor(self.classes_mapping[start_idx:end_idx])
        )[0]
        self.current_classes = [
            self.idx_to_class[i.item()] for i in self.current_classes_indices
        ]
        self.current_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.current_classes_indices
        ]
        self.current_classes_features = self.unique_class_features[
            self.current_classes_indices.numpy()
        ]

    def __get_past_classes(self):
        # we call __get_past_classes() before __get_learned_classes()
        self.past_classes_indices = self.learned_classes_indices
        self.past_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.past_labels = [
            self.class_to_idx[self.idx_to_class[i.item()]]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_features = self.unique_class_features[
            self.learned_classes_indices.numpy()
        ]

    def __get_learned_classes(self, end_idx):
        self.learned_classes_indices = torch.sort(
            torch.tensor(self.classes_mapping[:end_idx])
        )[0]
        self.learned_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.learned_labels = [
            self.class_to_idx[self.idx_to_class[i.item()]]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_features = self.unique_class_features[
            self.learned_classes_indices.numpy()
        ]


class DatasetIncrementalHDF5Dataset(ConcatHDF5Dataset):
    def __init__(self, datasets, name_list):
        super().__init__(datasets, name_list)
        self.num_stages = len(self.datasets)
        self.learned_classes_indices = torch.tensor([], dtype=torch.long)
        self.init_stage()

    def init_stage(self):
        self.stage = 0
        self.__build_subset()

    def forward_stage(self):
        self.stage += 1
        if self.stage == self.num_stages:
            print("Finish all stages")
            return
        self.__build_subset()

    def __build_subset(self):
        self.__get_current_stage_classes()
        if self.stage != 0:
            self.__get_past_classes()
        self.__get_learned_classes()

    def __get_current_stage_classes(self):
        self.current_classes_indices = torch.unique(
            torch.from_numpy(self.labels[self.stage]), sorted=True
        )
        # print("current_classes_indices", self.current_classes_indices)
        # print("idx_to_class", self.idx_to_class)
        self.current_classes = [
            self.idx_to_class[i.item()] for i in self.current_classes_indices
        ]
        self.current_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.current_classes_indices
        ]
        self.current_classes_features = self.unique_class_features[
            self.current_classes_indices.numpy()
        ]

    def __get_past_classes(self):
        # we call __get_past_classes() before __get_learned_classes()
        self.past_classes_indices = self.learned_classes_indices
        self.past_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.past_labels = [
            self.class_to_idx[self.idx_to_class[i.item()]]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_features = self.unique_class_features[
            self.learned_classes_indices.numpy()
        ]

    def __get_learned_classes(self):
        learned_classes_indices_current_stage = torch.unique(
            torch.from_numpy(self.labels[self.stage]), sorted=True
        )
        self.learned_classes_indices = torch.cat(
            (self.learned_classes_indices, learned_classes_indices_current_stage)
        )
        # num_classes_before_sorted = len(self.learned_classes_indices)
        # self.learned_classes_indices = torch.unique(
        #     self.learned_classes_indices, sorted=True
        # )
        # assert len(self.learned_classes_indices) == num_classes_before_sorted
        self.learned_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_features = self.unique_class_features[
            self.learned_classes_indices.numpy()
        ]

    def __len__(self):
        return len(self.labels[self.stage])

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.image_features[self.stage][index]),
            torch.tensor(self.labels[self.stage][index]).long(),
            torch.from_numpy(self.label_features[self.stage][index]),
        )


def load_hdf5_files_to_datasets(dataset_list, hdf5_folder, load_train=True):
    # Find all hdf5 files
    available_datasets = os.listdir(hdf5_folder)
    # Filter hdf5 files
    for hd in available_datasets:
        if not hd.endswith(".hdf5"):
            available_datasets.remove(hd)

    train_datasets = []
    test_datasets = []

    for dataset in dataset_list:
        print(f"Loading {dataset}...")
        # deal with train dataset
        if load_train:
            if f"{dataset}_train.hdf5" not in available_datasets:
                raise ValueError(f"hdf5 file for {dataset} not found")
            hdf5_file_path = os.path.join(hdf5_folder, f"{dataset}_train.hdf5")
            vector_dict = {}
            with h5py.File(hdf5_file_path, "r") as hf:
                for key in hf.keys():
                    vector_dict[key] = np.array(hf[key])
            train_datasets.append(vector_dict)

        # deal with test dataset
        if f"{dataset}_test.hdf5" not in available_datasets:
            raise ValueError(f"hdf5 file for {dataset} not found")
        hdf5_file_path = os.path.join(hdf5_folder, f"{dataset}_test.hdf5")
        vector_dict = {}
        with h5py.File(hdf5_file_path, "r") as hf:
            for key in hf.keys():
                vector_dict[key] = np.array(hf[key])
        test_datasets.append(vector_dict)

    return train_datasets, test_datasets


def get_hdf5_continual_learning_dataset(args):
    # deal with dataset
    dataset_list, n_datasets = deal_with_dataset(args)
    # hdf5_folder = os.path.join(args.data_root, "hdf5")
    hdf5_folder = args.data_root

    train_datasets, test_datasets = load_hdf5_files_to_datasets(dataset_list, hdf5_folder)

    # fix: update: to unify the interface, the returned test_dataset is always a list
    if args.incremental == "data":
        train_dataset = DataIncrementalHDF5Dataset(train_datasets, dataset_list)
        # test_dataset = DataIncrementalDataset(test_datasets, dataset_list)
        test_dataset = [ConcatHDF5Dataset(test_datasets, dataset_list)]
    elif args.incremental == "class":
        train_dataset = ClassIncrementalHDF5Dataset(
            train_datasets, dataset_list, args.num_classes
        )
        test_dataset = ClassIncrementalHDF5Dataset(
            test_datasets,
            dataset_list,
            args.num_classes,
            inclusive=True,
            include_whole=True,
        )
        # test_dataset = [ConcatDataset(test_datasets, dataset_list)]
    elif args.incremental == "dataset":
        train_dataset = DatasetIncrementalHDF5Dataset(train_datasets, dataset_list)
        # test_dataset = DatasetIncrementalDataset(test_datasets, dataset_list)
        # train_dataset = []
        test_dataset = []
        for i in range(len(test_datasets)):
            # train_dataset.append(ConcatDataset([train_datasets[i]], [dataset_list[i]]))
            test_dataset.append(ConcatHDF5Dataset([test_datasets[i]], [dataset_list[i]]))
    else:
        raise ValueError(f"Unknown incremental type: {args.incremental}")

    return train_dataset, test_dataset


def get_hdf5_held_out_dataset(args):
    # deal with dataset
    dataset_str = args.held_out_dataset
    dataset_list = dataset_str.split(",")

    # dataset_list, n_datasets = deal_with_dataset(args)
    # hdf5_folder = os.path.join(args.data_root, "hdf5")
    hdf5_folder = args.data_root
    train_datasets = []
    test_datasets = []
    for dataset in dataset_list:
        train_dataset, test_dataset = load_hdf5_files_to_datasets(
            [dataset], hdf5_folder, load_train=False
        )
        # train_datasets.append(ConcatDataset(train_dataset, [dataset]))
        test_datasets.append(ConcatHDF5Dataset(test_dataset, [dataset]))
    return train_datasets, test_datasets


# def get_continual_learning_dataset_1(args):
#     # deal with dataset
#     dataset_list, n_datasets = deal_with_dataset(args)
#     hdf5_folder = os.path.join(args.data_root, "hdf5")
#
#     train_datasets, test_datasets = load_hdf5_datasets(dataset_list, hdf5_folder)
#
#     if args.incremental == "data":
#         train_dataset = DataIncrementalHDF5Dataset(train_datasets, dataset_list)
#         # test_dataset = DataIncrementalDataset(test_datasets, dataset_list)
#         test_dataset = ConcatHDF5Dataset(test_datasets, dataset_list)
#     elif args.incremental == "class":
#         train_dataset = ClassIncrementalHDF5Dataset(
#             train_datasets, dataset_list, args.num_classes
#         )
#         test_dataset = ConcatHDF5Dataset(test_datasets, dataset_list)
#     elif args.incremental == "dataset":
#         train_dataset = DatasetIncrementalHDF5Dataset(train_datasets, dataset_list)
#         test_dataset = ConcatHDF5Dataset(test_datasets, dataset_list)
#     else:
#         raise ValueError(f"Unknown incremental type: {args.incremental}")
#
#     return train_dataset, test_dataset
#
#
# def get_continual_learning_dataset_1(args):
#     # deal with dataset
#     dataset_list, n_datasets = deal_with_dataset(args)
#     hdf5_folder = os.path.join(args.data_root, "hdf5")
#
#     train_datasets, test_datasets = load_hdf5_datasets(dataset_list, hdf5_folder)
#
#     if args.incremental == "data":
#         train_dataset = DataIncrementalHDF5Dataset(train_datasets, dataset_list)
#         # test_dataset = DataIncrementalDataset(test_datasets, dataset_list)
#         test_dataset = ConcatHDF5Dataset(test_datasets, dataset_list)
#     elif args.incremental == "class":
#         train_dataset = ClassIncrementalHDF5Dataset(
#             train_datasets, dataset_list, args.num_classes
#         )
#         test_dataset = ConcatHDF5Dataset(test_datasets, dataset_list)
#     elif args.incremental == "dataset":
#         train_dataset = DatasetIncrementalHDF5Dataset(train_datasets, dataset_list)
#         test_dataset = ConcatHDF5Dataset(test_datasets, dataset_list)
#     else:
#         raise ValueError(f"Unknown incremental type: {args.incremental}")
#
#     return train_dataset, test_dataset


# def get_held_out_dataset_1(args):
#     # deal with dataset
#     dataset_str = args.held_out_dataset
#     dataset_list = dataset_str.split(",")
#
#     # dataset_list, n_datasets = deal_with_dataset(args)
#     hdf5_folder = os.path.join(args.data_root, "hdf5")
#     train_datasets, test_datasets = load_hdf5_datasets(dataset_list, hdf5_folder)
#     train_dataset = ConcatHDF5Dataset(train_datasets, dataset_list)
#     test_dataset = ConcatHDF5Dataset(test_datasets, dataset_list)
#     return train_dataset, test_dataset
#
#
# def get_held_out_dataset_1(args):
#     # deal with dataset
#     dataset_str = args.held_out_dataset
#     dataset_list = dataset_str.split(",")
#
#     # dataset_list, n_datasets = deal_with_dataset(args)
#     hdf5_folder = os.path.join(args.data_root, "hdf5")
#     train_datasets, test_datasets = load_hdf5_datasets(dataset_list, hdf5_folder)
#     train_dataset = ConcatHDF5Dataset(train_datasets, dataset_list)
#     test_dataset = ConcatHDF5Dataset(test_datasets, dataset_list)
#     return train_dataset, test_dataset


# def get_ablation_datasets(args):
#     hdf5_folder = os.path.join(args.data_root, "hdf5")
#     # deal with dataset
#     ablation_datasets = args.datasets
#     ablation_datasets_list = ablation_datasets.split(",")
#     train_ablation_datasets = []
#     test_ablation_datasets = []
#     for dataset in ablation_datasets_list:
#         train_hdf5_datasets, test_hdf5_datasets = load_hdf5_datasets(
#             [dataset], hdf5_folder
#         )
#         train_dataset = ConcatHDF5Dataset(train_hdf5_datasets, [dataset])
#         test_dataset = ConcatHDF5Dataset(test_hdf5_datasets, [dataset])
#         train_ablation_datasets.append(train_dataset)
#         test_ablation_datasets.append(test_dataset)
#     return train_ablation_datasets, test_ablation_datasets
#
#
# def get_train_and_test_dataset(args):
#     # deal with dataset
#     memory_dataset_str = args.datasets
#     memory_dataset_list = memory_dataset_str.split(",")
#     held_out_dataset_str = args.held_out_dataset
#     held_out_dataset_list = held_out_dataset_str.split(",")
#     train_dataset_list = memory_dataset_list + held_out_dataset_list
#
#     # dataset_list, n_datasets = deal_with_dataset(args)
#     hdf5_folder = os.path.join(args.data_root, "hdf5")
#
#     memory_hdf5_datasets, _ = load_hdf5_datasets(memory_dataset_list, hdf5_folder)
#     train_hdf5_datasets, test_hdf5_datasets = load_hdf5_datasets(
#         train_dataset_list, hdf5_folder
#     )
#
#     memory_dataset = ConcatHDF5Dataset(memory_hdf5_datasets, memory_dataset_list)
#     train_dataset = ConcatHDF5Dataset(train_hdf5_datasets, train_dataset_list)
#     test_dataset = ConcatHDF5Dataset(test_hdf5_datasets, train_dataset_list)
#
#     return memory_dataset, train_dataset, test_dataset
