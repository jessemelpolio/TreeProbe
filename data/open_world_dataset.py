import torch
import bisect
import random
from torch.utils.data import Subset
import copy


# This class inherits mostly from ConcatDataset but is modified to allow for open world recognition
class OpenWorldDataset(torch.utils.data.Dataset):
    # By default, novel class should be in the first position
    def __init__(self, datasets, novel_datasets=None):
        if novel_datasets is None:
            novel_datasets = []
        self.datasets = list(datasets)
        assert self.datasets, "datasets should not be an empty iterable"
        self.targets = []
        self.text_targets = []

        self.classes = ["novel"]

        for d in self.datasets:
            assert not isinstance(
                d, torch.utils.data.IterableDataset
            ), "ConcatDataset does not support IterableDataset"
            assert hasattr(
                d, "class_to_idx"
            ), "Dataset should have class_to_idx attribute"
            assert isinstance(d.class_to_idx, dict), "class_to_idx should be a dict"
            idx_to_class = {v: k for k, v in d.class_to_idx.items()}
            self.classes += [idx_to_class[i] for i in range(len(idx_to_class))]
            cum_max_class = max(self.targets, default=1)
            for i in range(len(d)):
                sample, target = d[i]
                self.text_targets.append(idx_to_class[target])
                self.targets.append(target + cum_max_class)

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if len(novel_datasets) > 0:
            self.novel_datasets = list(novel_datasets)
            assert (
                len(self.novel_datasets) > 0
            ), "novel_datasets should not be an empty iterable"
            for n_d, d in enumerate(self.novel_datasets):
                print("Dealing with dataset {}".format(d))
                for _ in range(len(d)):
                    self.targets.append(0)
                    self.text_targets.append("novel")

        self.cumulative_sizes = (
            self.cumsum(self.datasets + self.novel_datasets)
            if len(novel_datasets) > 0
            else self.cumsum(self.datasets)
        )

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

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
            self.targets[idx],
            self.text_targets[idx],
        )

    def __len__(self):
        return self.cumulative_sizes[-1]


# This class inherits mostly from ConcatDataset but is modified to allow for open world recognition
class ConcatWithTextDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"
        self.targets = []
        self.text_targets = []
        self.classes = []
        self.original_classes = []
        self.breakpoints = [0]
        self.name = "ConcatWithTextDataset"

        for n_d, d in enumerate(self.datasets):
            name = d.__name__ if hasattr(d, "__name__") else d.__class__.__name__
            assert not isinstance(
                d, torch.utils.data.IterableDataset
            ), "ConcatDataset does not support IterableDataset"
            assert hasattr(
                d, "class_to_idx"
            ), "Dataset should have class_to_idx attribute"
            assert isinstance(d.class_to_idx, dict), "class_to_idx should be a dict"
            assert hasattr(d, "targets"), "Dataset should have targets attribute"
            assert hasattr(
                d, "prompt_template"
            ), "Dataset should have prompt_template attribute"

            cum_max_class = len(self.classes)
            self.name += "_" + name
            prompt = random.choice(d.prompt_template)
            class_to_idx = {prompt.format(c): i for i, c in enumerate(d.classes)}
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            # TODO: check if these classes are already in the list
            self.original_classes += d.classes
            self.classes += [idx_to_class[i] for i in range(len(idx_to_class))]
            self.breakpoints.append(len(self.classes))
            # cum_max_class = max(self.targets) + 1 if len(self.targets) > 0 else 0
            # cum_max_class = len(self.classes) - len(d.classes)
            text_targets = [idx_to_class[t] for t in d.targets]
            data_targets = [t + cum_max_class for t in d.targets]
            self.targets += data_targets
            self.text_targets += text_targets

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        self.cumulative_sizes = self.cumsum(self.datasets)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

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

    def __len__(self):
        return self.cumulative_sizes[-1]


class RemappedWithTextDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, class_index, classes, original_classes):
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"
        self.targets = []
        self.text_targets = []
        assert type(classes) == list, "class_index should be a list"
        assert (
            len(classes) == class_index
        ), "class_index should be a list of length len(classes)"
        self.classes = classes
        self.original_classes = original_classes
        self.breakpoints = [0]
        self.name = "RemappedWithTextDataset"

        for n_d, d in enumerate(self.datasets):
            name = d.__name__ if hasattr(d, "__name__") else d.__class__.__name__
            assert not isinstance(
                d, torch.utils.data.IterableDataset
            ), "ConcatDataset does not support IterableDataset"
            assert hasattr(
                d, "class_to_idx"
            ), "Dataset should have class_to_idx attribute"
            assert isinstance(d.class_to_idx, dict), "class_to_idx should be a dict"
            assert hasattr(d, "targets"), "Dataset should have targets attribute"
            assert hasattr(
                d, "prompt_template"
            ), "Dataset should have prompt_template attribute"
            cum_max_class = len(self.classes)
            self.name += "_" + name
            prompt = random.choice(d.prompt_template)
            class_to_idx = {prompt.format(c): i for i, c in enumerate(d.classes)}
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            # TODO: check if these classes are already in the list
            self.original_classes += d.classes
            self.classes += [idx_to_class[i] for i in range(len(idx_to_class))]
            self.breakpoints.append(len(self.classes))
            # cum_max_class = class_index + max(self.targets) if len(self.targets) > 0 else class_index
            # cum_max_class = len(self.classes) - len(d.classes)
            text_targets = [idx_to_class[t] for t in d.targets]
            data_targets = [t + cum_max_class for t in d.targets]
            self.targets += data_targets
            self.text_targets += text_targets

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        self.cumulative_sizes = self.cumsum(self.datasets)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

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

    def __len__(self):
        return self.cumulative_sizes[-1]


class SubsampleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.name = self.dataset.name + "_subsampled"
        self.datasets = self.dataset.datasets
        # self.targets = [self.dataset.targets[i] for i in self.indices]
        # self.text_targets = [self.dataset.text_targets[i] for i in self.indices]
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = self.dataset.idx_to_class
        self.breakpoints = self.dataset.breakpoints
        self.cumulative_sizes = self.dataset.cumulative_sizes

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, sample_buffer):
        self.dataset = copy.deepcopy(sample_buffer)
        self.image_features = self.dataset["image_features"].cpu().detach()
        self.label_features = self.dataset["label_features"].cpu().detach()
        self.labels = self.dataset["labels"].cpu().detach()

    def __getitem__(self, index):
        return (
            self.image_features[index],
            self.labels[index],
            self.label_features[index],
        )

    def __len__(self):
        return self.image_features.shape[0]
