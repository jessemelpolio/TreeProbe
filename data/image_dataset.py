import torch
import numpy as np
import random
import bisect
from .combined_dataset import deal_with_dataset, single_dataset_build
import os
import copy


class ContinualImageDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        datasets = copy.deepcopy(datasets)
        self.name = "UnitedDataset"
        self.datasets = list(datasets)
        self.num_datasets = len(self.datasets)
        assert self.num_datasets > 0, "datasets should not be an empty iterable"
        self.per_dataset_size = [len(d) for d in self.datasets]
        self.per_dataset_classes = []
        self.per_dataset_classes_with_prompt = []
        self.per_dataset_classes_indices = []
        self.per_dataset_targets = []
        self.classes = []
        self.classes_with_prompt = []
        self.breakpoints = [0]
        self.targets = []
        self.text_targets = []

        class_to_global_idx = {}
        global_idx = 0

        for d in self.datasets:
            name = d.__name__ if hasattr(d, "__name__") else d.__class__.__name__
            self.name += f"_{name}"
            assert hasattr(d, "class_to_idx"), f"Dataset {name} should have class_to_idx attribute"
            class_to_idx = d.class_to_idx
            assert hasattr(d, "prompt_template"), f"Dataset {name} should have prompt_template attribute"
            prompt = random.choice(d.prompt_template)
            current_classes = list(class_to_idx.keys())

            current_classes_with_prompt = [prompt.format(c) for c in current_classes]
            self.per_dataset_classes_with_prompt.append(current_classes_with_prompt)

            # Handle overlapping classes
            current_global_indices = []
            for cls in current_classes:
                if cls not in class_to_global_idx:
                    class_to_global_idx[cls] = global_idx
                    self.classes.append(cls)
                    self.classes_with_prompt.append(prompt.format(cls))
                    global_idx += 1
                current_global_indices.append(class_to_global_idx[cls])

            self.per_dataset_classes.append(current_classes)
            self.breakpoints.append(len(self.classes))

            assert hasattr(d, "targets"), "Dataset should have targets attribute"

            text_targets = [current_classes_with_prompt[t] for t in d.targets]
            remapped_targets = [current_global_indices[t] for t in d.targets]

            self.per_dataset_targets.append(d.targets)
            self.per_dataset_classes_indices.append(current_global_indices)
            self.targets += remapped_targets
            self.text_targets += text_targets

        # class_to_idx might not be a dict due to overlapping classes
        # self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = dict(enumerate(self.classes))
        self.idx_to_class_with_prompt = dict(enumerate(self.classes_with_prompt))
        self.cumulative_sizes = self.cumsum(self.datasets)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def init_stage(self):
        pass

    def forward_stage(self):
        pass

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return (
            self.datasets[dataset_idx][sample_idx][0],
            torch.tensor(self.targets[idx], dtype=torch.long),
            self.text_targets[idx],
        )


class ConcatImageDataset(ContinualImageDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.name = "ConcatImageDataset"
        self.subset_item_indices = list(range(len(self.targets)))

    def __build_subset(self):
        pass

    def init_stage(self):
        self.stage = 0
        self.__build_subset()

    def forward_stage(self):
        self.stage += 1
        if self.stage >= self.num_stages:
            print("Finish all stages")
            return
        self.__build_subset()

    def __len__(self):
        return len(self.subset_item_indices)

    def __getitem__(self, idx):
        return super().__getitem__(self.subset_item_indices[idx])

    def get_task_indentifier(self):
        return self.stage


class ZeroShotImageDataset(ContinualImageDataset):
    def __init__(self, datasets, num_classes=100, padding=0):
        super().__init__(datasets)

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
        self.subset_item_indices = [
            idx
            for idx in range(self.cumulative_sizes[-1])
            if self.targets[idx] in self.random_classes_indices
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
        image, _, text = super().__getitem__(subset_index)
        return (
            image,
            torch.tensor(
                self.remapping_label[self.targets[subset_index]] + self.padding
            ).long(),
            text
        )


class MixImageDataset(ContinualImageDataset):
    def __init__(self, datasets, split=5, num_samples=100):
        super().__init__(datasets)

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
        for idx in range(self.cumulative_sizes[-1]):
            task = (
                self.reverse_repermutation[self.targets[idx]]
                // self.subset_classes_length
            )
            if task < split:
                assert (
                    self.targets[idx] in self.subset_class_indices[task]
                ), "label should be in subset_class_indices"
                if self.targets[idx] not in sample_count:
                    sample_count[self.targets[idx]] = 0
                if sample_count[self.targets[idx]] < num_samples:
                    sample_count[self.targets[idx]] += 1
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
            self.classes += self.additional_classes
            self.classes_with_prompt += self.additional_classes_with_prompt

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
        image, _, text = super().__getitem__(subset_index)
        return (
            image,
            torch.tensor(self.remapping_label[self.targets[subset_index]]).long(),
            text,
        )


class DataIncrementalImageDataset(ConcatImageDataset):
    def __init__(self, datasets, perc=0.01, perc_list=None):
        super().__init__(datasets)
        self.name = "DataIncrementalDataset"
        if perc_list is not None:
            self.perc_list = perc_list
            assert isinstance(self.perc_list, list), "perc_list should be a list"
            assert sum(self.perc_list) in [1, 100], "perc_list should sum to 1 or 100"
            if sum(self.perc_list) == 100:
                self.perc_list = [p / 100 for p in self.perc_list]
            self.num_stages = len(self.perc_list)
            self.dataset_size = len(self.dataset)
        else:
            self.perc = perc
            self.subset_size = int(len(self.dataset) * self.perc)
            self.num_stages = (
                len(self.dataset) // self.subset_size
                if len(self.dataset) % self.subset_size == 0
                else len(self.dataset) // self.subset_size + 1
            )

        random.seed(0)
        self.random_indices = random.sample(range(len(self.dataset)), len(self.dataset))
        self.learned_classes_indices = torch.tensor([], dtype=torch.long)
        # ############################################################
        # self.classes = self.classes_with_prompt
        # self.idx_to_class = self.dataset.idx_to_class_with_prompt
        # ############################################################
        self.init_stage()

    def init_stage(self):
        self.stage = 0
        self.__build_subset()

    def forward_stage(self):
        self.stage += 1
        if self.stage >= self.num_stages:
            print("Finish all stages")
            return
        self.__build_subset()

    def __build_subset(self):
        # if self.stage == "mix":
        #     self.subset_item_indices = list(range(len(self.dataset)))
        #     return
        if hasattr(self, "perc_list"):
            start_idx = int(self.dataset_size * sum(self.perc_list[: self.stage]))
            end_idx = int(self.dataset_size * sum(self.perc_list[: self.stage + 1]))
            if self.stage == self.num_stages - 1:
                end_idx = len(self.dataset)
        else:
            start_idx = self.stage * self.subset_size
            end_idx = min((self.stage + 1) * self.subset_size, len(self.dataset))
        self.subset_item_indices = self.random_indices[start_idx:end_idx]
        # self.subset_item_indices.extend(self.random_indices[start_idx:end_idx])
        self.__get_current_stage_classes()
        if self.stage != 0:
            self.__get_past_classes()
        self.__get_learned_classes()

    def __get_current_stage_classes(self):
        self.current_classes_indices = torch.unique(
            torch.from_numpy(self.targets[self.subset_item_indices]), sorted=True
        )
        self.current_classes = [
            self.idx_to_class[i.item()] for i in self.current_classes_indices
        ]
        self.current_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.current_classes_indices
        ]

    def __get_past_classes(self):
        self.past_classes_indices = self.learned_classes_indices
        self.past_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        # self.past_labels = [
        #     self.class_to_idx[self.idx_to_class[i.item()]]
        #     for i in self.learned_classes_indices
        # ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]

    def __get_learned_classes(self):
        learned_classes_indices_current_stage = torch.unique(
            torch.from_numpy(self.targets[self.subset_item_indices]), sorted=True
        )
        self.learned_classes_indices = torch.cat(
            (self.learned_classes_indices, learned_classes_indices_current_stage)
        )
        self.learned_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]


class ClassIncrementalImageDataset(ConcatImageDataset):
    def __init__(self, dataset, num_classes=10, randomize=False, perc_list=None):
        super().__init__(dataset)
        self.name = "ClassIncrementalDataset"
        if perc_list is not None:
            self.perc_list = perc_list
            assert sum(perc_list) in [1, 100], "perc_list should sum to 1"
            if sum(self.perc_list) == 100:
                self.perc_list = [p / 100 for p in self.perc_list]
            self.num_stages = len(perc_list)
        else:
            self.num_stages = (
                len(self.classes) // num_classes
                if len(self.classes) % num_classes < num_classes // 2
                else len(self.classes) // num_classes + 1
            )
        self.classes_mapping = list(range(len(self.classes)))
        self.num_classes = num_classes
        # ############################################################
        # self.classes = self.dataset.classes_with_prompt
        # self.idx_to_class = self.dataset.idx_to_class_with_prompt
        # ############################################################
        self.classes_mapping = list(range(len(self.classes)))
        random.seed(0)
        random.shuffle(self.classes_mapping)

        # inverse_classes_mapping = {v: k for k, v in enumerate(self.classes_mapping)}
        # self.inverse_classes_mapping = np.array(
        #     [inverse_classes_mapping[i] for i in range(len(self.classes))]
        # )
        # self.breakpoints = [i * self.num_classes for i in range(self.num_stages + 1)]
        self.init_stage()

    def init_stage(self):
        self.stage = 0
        self.subset_classes = []
        # self.subset_item_indices = []
        self.__build_subset()

    def forward_stage(self):
        self.stage += 1
        if self.stage >= self.num_stages:
            print("Finish all stages")
            return
        self.__build_subset()

    def __build_subset(self):
        # if self.stage == "mix":
        #     self.subset_item_indices = list(range(len(self.dataset)))
        #     return
        # self.subset_item_indices = []
        if hasattr(self, "perc_list"):
            start_idx = int(len(self.classes) * sum(self.perc_list[: self.stage]))
            end_idx = int(len(self.classes) * sum(self.perc_list[: self.stage + 1]))
            if start_idx == end_idx:
                end_idx += 1
            if self.stage == self.num_stages - 1:
                end_idx = len(self.classes)
        else:
            start_idx = self.stage * self.num_classes
            end_idx = min((self.stage + 1) * self.num_classes, len(self.classes))

        if self.stage == self.num_stages - 1:
            end_idx = len(self.classes)

        self.subset_classes = self.classes_mapping[start_idx:end_idx]

        self.subset_item_indices = [
            idx
            for idx in range(len(self.dataset))
            if int(self.dataset.targets[idx]) in self.subset_classes
        ]

        self.__get_current_stage_classes(start_idx, end_idx)
        if self.stage != 0:
            self.__get_past_classes()
        self.__get_learned_classes(end_idx)

    def __get_current_stage_classes(self, start_idx, end_idx):
        self.current_classes_indices = torch.tensor(
            list(self.classes_mapping[start_idx:end_idx]), dtype=torch.long
        )
        self.current_classes = [
            self.idx_to_class[i.item()] for i in self.current_classes_indices
        ]
        self.current_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.current_classes_indices
        ]

    def __get_past_classes(self):
        self.past_classes_indices = self.learned_classes_indices
        self.past_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        # self.past_labels = [
        #     self.class_to_idx[self.idx_to_class[i.item()]]
        #     for i in self.learned_classes_indices
        # ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]

    def __get_learned_classes(self, end_idx):
        self.learned_classes_indices = torch.sort(
            torch.tensor(self.classes_mapping[:end_idx])
        )[0]
        self.learned_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        # self.learned_labels = [
        #     self.class_to_idx[self.idx_to_class[i.item()]]
        #     for i in self.learned_classes_indices
        # ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]

class DatasetIncrementalImageDataset(ConcatImageDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.name = "DatasetIncrementalDataset"
        self.num_stages = len(self.datasets)
        self.learned_classes_indices = torch.tensor([], dtype=torch.long)
        # ############################################################
        # self.classes = self.dataset.classes_with_prompt
        # self.idx_to_class = self.dataset.idx_to_class_with_prompt
        # ############################################################
        self.init_stage()

    def init_stage(self):
        self.stage = 0
        self.__build_subset()

    def forward_stage(self):
        self.stage += 1
        if self.stage >= self.num_stages:
            print("Finish all stages")
            return
        self.__build_subset()

    def __build_subset(self):
        if self.stage == 0:
            self.subset_item_indices = list(range(len(self.datasets[self.stage])))
        else:
            self.subset_item_indices = list(
                range(
                    self.cumulative_sizes[self.stage - 1],
                    self.cumulative_sizes[self.stage],
                )
            )
        self.__get_current_stage_classes()
        if self.stage != 0:
            self.__get_past_classes()
        self.__get_learned_classes()

    def __get_current_stage_classes(self):
        self.current_classes = self.per_dataset_classes[self.stage]
        self.current_classes_with_prompt = self.per_dataset_classes_with_prompt[self.stage]
        self.current_classes_indices = torch.tensor(
            self.per_dataset_classes_indices[self.stage], dtype=torch.long
        )

    def __get_past_classes(self):
        # we call __get_past_classes() before __get_learned_classes()
        self.past_classes_indices = self.learned_classes_indices
        self.past_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]

    def __get_learned_classes(self):
        current_classes_indices = torch.tensor(self.per_dataset_classes_indices[self.stage], dtype=torch.long)
        self.learned_classes_indices = torch.cat(
            (self.learned_classes_indices, current_classes_indices)
        )
        self.learned_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]


def get_image_continual_testset_dataset(args):
    # deal with dataset
    dataset_list, n_datasets = deal_with_dataset(args)

    # deal with each dataset
    train_datasets = []
    test_datasets = []
    for dataset in dataset_list:
        train_dataset, test_dataset = single_dataset_build(args, dataset)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    # unite datasets
    united_train_dataset = ConcatImageDataset(train_datasets)
    united_test_dataset = ConcatImageDataset(test_datasets)

    if args.incremental == "data":
        incremental_train_dataset = DataIncrementalImageDataset(
            united_train_dataset, args.perc, perc_list=[2, 2, 4, 8, 16, 32, 36]
        )
        incremental_test_dataset = DataIncrementalImageDataset(
            united_test_dataset, args.perc, perc_list=[2, 2, 4, 8, 16, 32, 36]
        )
    elif args.incremental == "class":
        incremental_train_dataset = ClassIncrementalImageDataset(
            united_train_dataset, args.num_classes, args.randomize
        )
        incremental_test_dataset = ClassIncrementalImageDataset(
            united_test_dataset, args.num_classes, args.randomize
        )
    elif args.incremental == "dataset":
        incremental_train_dataset = DatasetIncrementalImageDataset(united_train_dataset)
        incremental_test_dataset = DatasetIncrementalImageDataset(united_test_dataset)
    else:
        raise NotImplementedError("incremental type not implemented")

    return incremental_train_dataset, incremental_test_dataset


def get_image_continual_learning_dataset(args):
    # deal with dataset
    dataset_list, n_datasets = deal_with_dataset(args)

    # deal with each dataset
    train_datasets = []
    test_datasets = []
    for dataset in dataset_list:
        train_dataset, test_dataset = single_dataset_build(args, dataset)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    # unite datasets
    # united_train_dataset = ConcatImageDataset(train_datasets)
    # united_test_dataset = ConcatImageDataset(test_datasets)
    united_train_dataset = train_datasets
    united_test_dataset = test_datasets

    if args.incremental == "data":
        incremental_train_dataset = DataIncrementalImageDataset(
            united_train_dataset, args.perc, perc_list=[2, 2, 4, 8, 16, 32, 36]
        )
        incremental_test_dataset = united_test_dataset
    elif args.incremental == "class":
        incremental_train_dataset = ClassIncrementalImageDataset(
            united_train_dataset, args.num_classes, args.randomize
        )
        incremental_test_dataset = united_test_dataset
    elif args.incremental == "dataset":
        incremental_train_dataset = DatasetIncrementalImageDataset(united_train_dataset)
        incremental_test_dataset = [ConcatImageDataset([test_d]) for test_d in test_datasets]
    else:
        raise NotImplementedError("incremental type not implemented")

    return incremental_train_dataset, incremental_test_dataset


def get_image_held_out_dataset(args):
    # deal with dataset
    dataset_str = args.held_out_dataset
    dataset_list = dataset_str.split(",")

    # deal with each dataset
    train_datasets = []
    test_datasets = []
    for dataset in dataset_list:
        train_dataset, test_dataset = single_dataset_build(args, dataset)
        train_datasets.append(ConcatImageDataset([train_dataset]))
        test_datasets.append(ConcatImageDataset([test_dataset]))

    return train_datasets, test_datasets


def get_conceptual_captions_dataset(args):
    import pandas as pd
    root = args.ref_caption_root
    file_name = "Validation_GCC-1.1.0-Validation.tsv"
    file_path = os.path.join(root, file_name)
    df = pd.read_csv(file_path, sep='\t')
    captions = df.iloc[:, 0].tolist()
    return captions


def get_image_dataset_with_name(args, dataset_name):
    # deal with dataset
    train_dataset, test_dataset = single_dataset_build(args, dataset_name)
    train_datasets = ConcatImageDataset([train_dataset])
    test_datasets = ConcatImageDataset([test_dataset])
    return train_datasets, test_datasets


def get_united_dataset(args):
    # deal with dataset
    dataset_list, n_datasets = deal_with_dataset(args)

    # deal with each dataset
    train_datasets = []
    test_datasets = []
    for dataset in dataset_list:
        train_dataset, test_dataset = single_dataset_build(args, dataset)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    # unite datasets
    united_train_dataset = ConcatImageDataset(train_datasets)
    united_test_dataset = ConcatImageDataset(test_datasets)

    return united_train_dataset, united_test_dataset
