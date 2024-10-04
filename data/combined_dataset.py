# from avalanche.benchmarks.utils import concat_classification_datasets
import os
import copy
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import random
from PIL import Image
import numpy as np
from data.open_world_dataset import (
    OpenWorldDataset,
    ConcatWithTextDataset,
    RemappedWithTextDataset,
    SubsampleDataset,
)
from .standardize_dataset import standardize_dataset


def get_augment_transforms(inp_sz):
    """
    Returns appropriate augmentation given dataset size and name
    Arguments:
        indices (sequence): a sequence of indices
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_augment = [torchvision.transforms.RandomResizedCrop(inp_sz)]
    test_augment = [
        torchvision.transforms.Resize(inp_sz + 32),
        torchvision.transforms.CenterCrop(inp_sz),
    ]

    # if dataset not in ['MNIST', 'SVHN', 'KMNIST']:
    train_augment.append(torchvision.transforms.RandomHorizontalFlip())

    train_augment_transform = torchvision.transforms.Compose(
        train_augment
        + [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]
    )
    test_augment_transform = torchvision.transforms.Compose(
        test_augment
        + [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]
    )

    return train_augment_transform, test_augment_transform


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_clip_transform(input_size):
    try:
        from torchvision.transforms import InterpolationMode

        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    return Compose(
        [
            # torchvision.transforms.RandomHorizontalFlip(),
            Resize(input_size, interpolation=BICUBIC),
            CenterCrop(input_size),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def deal_with_dataset(args):
    dataset_str = args.datasets
    dataset_list = dataset_str.split(",")
    n_datasets = len(dataset_list)
    return dataset_list, n_datasets


def dataset_func(dataset):  # sourcery skip: low-code-quality
    if dataset == "CIFAR10":
        return torchvision.datasets.CIFAR10
    elif dataset == "CIFAR100":
        return torchvision.datasets.CIFAR100
    elif dataset == "SVHN":
        return torchvision.datasets.SVHN
    elif dataset == "MNIST":
        return torchvision.datasets.MNIST
    elif dataset == "KMNIST":
        return torchvision.datasets.KMNIST
    elif dataset == "FashionMNIST":
        return torchvision.datasets.FashionMNIST
    elif dataset == "STL10":
        return torchvision.datasets.STL10
    elif dataset == "CUB200":
        return torchvision.datasets.CUB200
    elif dataset == "TinyImageNet":
        return torchvision.datasets.TinyImageNet
    elif dataset == "Caltech256":
        return torchvision.datasets.Caltech256
    elif dataset == "Omniglot":
        return torchvision.datasets.Omniglot
    elif dataset == "Flowers102":
        return torchvision.datasets.Flowers102
    elif dataset == "FGVCAircraft":
        return torchvision.datasets.FGVCAircraft
    elif dataset == "Food101":
        return torchvision.datasets.Food101
    elif dataset == "StanfordCars":
        return torchvision.datasets.StanfordCars
    elif dataset == "OxfordIIITPet":
        return torchvision.datasets.OxfordIIITPet
    elif dataset == "DTD":
        return torchvision.datasets.DTD
    elif dataset == "PCAM":
        return torchvision.datasets.PCAM
    elif dataset == "iNaturalist":
        from data.dataset_classes.inaturalist import iNaturalist

        return iNaturalist
    elif dataset == "Places365LT":
        from data.dataset_classes.places365 import Places365LT

        return Places365LT
    elif dataset == "Caltech101":
        from data.dataset_classes.caltech101 import Caltech101

        return Caltech101
    elif dataset == "SUN397":
        from data.dataset_classes.sun397 import SUN397

        return SUN397
    elif dataset == "EuroSAT":
        from data.dataset_classes.eurosat import EuroSAT

        return EuroSAT
    elif dataset == "ImageNet":
        from data.dataset_classes.imagenet import ImageNet

        return ImageNet
    elif dataset == "ImageNetA":
        from data.dataset_classes.imagenet import ImageNetA

        return ImageNetA
    elif dataset == "ImageNetR":
        from data.dataset_classes.imagenet import ImageNetR

        return ImageNetR
    elif dataset == "ImageNetSketch":
        from data.dataset_classes.imagenet import ImageNetSketch

        return ImageNetSketch
    elif dataset == "UCF101":
        from data.dataset_classes.ucf101 import UCF101

        return UCF101
    elif dataset == "Resisc45":
        from data.dataset_classes.resisc45 import Resisc45

        return Resisc45
    elif dataset == "CLEVRCount":
        from data.dataset_classes.clevr import CLEVRCount

        return CLEVRCount
    elif dataset == "CLEVRDist":
        from data.dataset_classes.clevr import CLEVRDist

        return CLEVRDist
    else:
        raise ValueError("Dataset not supported")


def get_open_world_dataset(list_of_datasets, cumulative=True):
    datasets = []
    length = len(list_of_datasets)

    for i in range(length):
        if cumulative:
            datasets.append(
                OpenWorldDataset(
                    list_of_datasets[: i + 1], novel_datasets=list_of_datasets[i + 1 :]
                )
            )
        else:
            datasets.append(
                OpenWorldDataset(
                    list_of_datasets[i], novel_datasets=list_of_datasets[i + 1 :]
                )
            )

    return datasets


def get_seen_unseen_dataset(list_of_datasets, scenario="cumulative_cumulative"):
    seen_config, unseen_config = scenario.split("_")

    if seen_config == "cumulative":
        seen_datasets = compile_datasets(list_of_datasets, mode="pre_cumulative")
    elif seen_config == "single":
        seen_datasets = compile_datasets(list_of_datasets, mode="single")
    elif seen_config == "exclusive":
        seen_datasets = compile_datasets(list_of_datasets, mode="exclusive")
    else:
        raise ValueError("Invalid seen config")

    if unseen_config == "cumulative":
        unseen_datasets = compile_datasets(list_of_datasets, mode="post_cumulative")
        del unseen_datasets[0]
        last = unseen_datasets[-1]
        unseen_datasets.append(last)
    elif unseen_config == "single":
        unseen_datasets = compile_datasets(list_of_datasets, mode="single")
    elif unseen_config == "exclusive":
        unseen_datasets = compile_datasets(list_of_datasets, mode="exclusive")
    else:
        raise ValueError("Invalid unseen config")

    return seen_datasets, unseen_datasets


def compile_datasets(list_of_datasets, mode="cumulative"):
    datasets = []
    length = len(list_of_datasets)

    for i in range(length):
        if mode == "pre_cumulative":
            datasets.append(ConcatWithTextDataset(list_of_datasets[: i + 1]))
        elif mode == "post_cumulative":
            datasets.append(ConcatWithTextDataset(list_of_datasets[i:]))
        elif mode == "single":
            datasets.append(ConcatWithTextDataset([list_of_datasets[i]]))
        elif mode == "exclusive":
            dataset_list = copy.deepcopy(list_of_datasets)
            del dataset_list[i]
            datasets.append(ConcatWithTextDataset(dataset_list))
        else:
            raise ValueError("Mode not supported")

    return datasets


def remap_datasets(list_of_datasets):
    datasets = []
    length = len(list_of_datasets)
    class_index = 0
    classes = []
    original_classes = []

    for i in range(length):
        datasets.append(
            RemappedWithTextDataset(
                [list_of_datasets[i]],
                class_index,
                copy.deepcopy(classes),
                copy.deepcopy(original_classes),
            )
        )
        # A potential bug here: if there are multiple prompt templates, the dataset and the current classes may choose different prompt templates
        class_to_idx = list_of_datasets[i].class_to_idx
        prompt = random.choice(list_of_datasets[i].prompt_template)
        current_classes = list(class_to_idx.keys())
        original_classes += current_classes
        current_classes = [prompt.format(c) for c in current_classes]
        classes += current_classes
        class_index += len(current_classes)

    return datasets


def get_combined_open_world_dataset(args):
    # deal with dataset
    dataset_list, n_datasets = deal_with_dataset(args)
    # deal with each dataset
    train_datasets = []
    test_datasets = []
    for i, dataset in enumerate(dataset_list):
        if args.network_arc == "clip":
            train_transform = get_clip_transform(args.input_size)
            test_transform = get_clip_transform(args.input_size)
        else:
            train_transform, test_transform = get_augment_transforms(args.input_size)

        df = dataset_func(dataset)
        train_datasets.append(
            df(
                root=args.data_root,
                split="train",
                download=True,
                transform=train_transform,
            )
        )
        test_datasets.append(
            df(
                root=args.data_root,
                split="val",
                download=True,
                transform=test_transform,
            )
        )

    open_world_train_dataset = get_open_world_dataset(train_datasets, cumulative=False)
    open_world_test_dataset = get_open_world_dataset(test_datasets, cumulative=True)

    return open_world_train_dataset, open_world_test_dataset, n_datasets


def get_combined_open_world_concat_dataset(args):
    # deal with dataset
    dataset_list, _ = deal_with_dataset(args)
    # deal with each dataset
    train_datasets = []
    test_datasets = []
    for i, dataset in enumerate(dataset_list):
        if args.network_arc == "clip" or args.network_arc == "rac":
            train_transform = get_clip_transform(args.input_size)
            test_transform = get_clip_transform(args.input_size)
        else:
            train_transform, test_transform = get_augment_transforms(args.input_size)

        df = dataset_func(dataset)
        train_set = df(
            root=args.data_root, split="train", download=True, transform=train_transform
        )
        train_set = standardize_dataset(train_set)
        train_datasets.append(train_set)
        if dataset in ["Food101", "StanfordCars"]:
            test_set = df(
                root=args.data_root,
                split="test",
                download=True,
                transform=test_transform,
            )
            test_set = standardize_dataset(test_set)
            test_datasets.append(test_set)
        else:
            test_set = df(
                root=args.data_root,
                split="val",
                download=True,
                transform=test_transform,
            )
            test_set = standardize_dataset(test_set)
            test_datasets.append(test_set)

    (
        open_world_train_dataset_seen,
        open_world_train_dataset_unseen,
    ) = get_seen_unseen_dataset(train_datasets, scenario=args.train_scenario)
    (
        open_world_test_dataset_seen,
        open_world_test_dataset_unseen,
    ) = get_seen_unseen_dataset(test_datasets, scenario=args.eval_scenario)

    accumulated_train_datasets = compile_datasets(train_datasets, mode="pre_cumulative")

    return (
        open_world_train_dataset_seen,
        open_world_train_dataset_unseen,
        open_world_test_dataset_seen,
        open_world_test_dataset_unseen,
        accumulated_train_datasets,
    )


def get_open_world_single_datasets(args, transform=None):
    # deal with dataset
    dataset_list, _ = deal_with_dataset(args)
    # deal with each dataset
    train_datasets = []
    test_datasets = []
    for i, dataset in enumerate(dataset_list):
        train_set, test_set = single_dataset_build(args, dataset, transform=transform)
        train_set = ConcatWithTextDataset([train_set])
        test_set = ConcatWithTextDataset([test_set])
        train_datasets.append(train_set)
        test_datasets.append(test_set)

    return train_datasets, test_datasets


def get_combined_continual_learning_datasets(args):
    # deal with dataset
    dataset_list, _ = deal_with_dataset(args)
    # deal with each dataset
    train_datasets = []
    test_datasets = []
    for i, dataset in enumerate(dataset_list):
        train_dataset, test_dataset = single_dataset_build(args, dataset)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    training_datasets_of_current_stage = remap_datasets(train_datasets)
    test_datasets_of_current_stage = remap_datasets(test_datasets)
    test_datasets_of_previous_stage = compile_datasets(
        test_datasets, mode="pre_cumulative"
    )
    test_datasets_of_previous_stage = [None] + test_datasets_of_previous_stage[:-1]

    accumulated_train_datasets = compile_datasets(train_datasets, mode="pre_cumulative")

    return (
        training_datasets_of_current_stage,
        test_datasets_of_current_stage,
        test_datasets_of_previous_stage,
        accumulated_train_datasets,
    )


def get_mixed_evaluation_dataset(args):
    # deal with dataset
    dataset_str = args.mixed_datasets
    dataset_list = dataset_str.split(",")
    dataset_lengths = []
    test_datasets = []
    # deal with each dataset
    for i, dataset in enumerate(dataset_list):
        train_set, test_set = single_dataset_build(args, dataset)
        dataset_lengths.append(len(test_set))
        test_datasets.append(test_set)

    min_dataset_length = min(dataset_lengths)
    total_dataset_length = sum(dataset_lengths)
    average_dataset_length = min(
        int(total_dataset_length / len(dataset_list)), min_dataset_length
    )

    np.random.seed(0)
    cur_length = 0
    total_indices = []
    for i, test_set in enumerate(test_datasets):
        indices = np.random.choice(
            len(test_set), size=average_dataset_length, replace=False
        )
        indices += cur_length
        total_indices.extend(indices.tolist())
        cur_length += len(test_datasets[i])

    mixed_dataset = ConcatWithTextDataset(test_datasets)
    mixed_dataset = SubsampleDataset(mixed_dataset, total_indices)
    return mixed_dataset


def single_dataset_build(args, dataset, transform=None):
    if args.network_arc == "clip" or args.network_arc == "rac":
        if transform is None:
            train_transform = get_clip_transform(args.input_size)
            test_transform = get_clip_transform(args.input_size)
        else:
            train_transform = transform
            test_transform = transform
    else:
        train_transform, test_transform = get_augment_transforms(args.input_size)

    df = dataset_func(dataset)
    if dataset in ["CIFAR10", "CIFAR100"]:
        train_set = df(
            root=args.data_root, train=True, download=True, transform=train_transform
        )
        test_set = df(
            root=args.data_root, train=False, download=True, transform=test_transform
        )
    elif dataset in ["Food101", "StanfordCars", "SVHN"]:
        train_set = df(
            root=args.data_root, split="train", download=True, transform=train_transform
        )
        test_set = df(
            root=args.data_root, split="test", download=True, transform=test_transform
        )
    elif dataset in ["Flowers102"]:
        train_set = df(
            root=args.data_root, split="train", download=True, transform=train_transform
        )
        test_set = df(
            root=args.data_root, split="val", download=True, transform=test_transform
        )
    elif dataset in ["OxfordIIITPet"]:
        train_set = df(
            root=args.data_root,
            split="trainval",
            download=True,
            transform=train_transform,
        )
        test_set = df(
            root=args.data_root, split="test", download=True, transform=test_transform
        )
    elif dataset in [
        "Places365LT",
        "iNaturalist",
        "ImageNet",
        "ImageNetA",
        "ImageNetR",
        "ImageNetSketch",
        "CLEVRCount",
        "CLEVRDist",
        "Resisc45",
        "UCF101",
    ]:
        train_set = df(root=args.data_root, split="train", transform=train_transform)
        test_set = df(root=args.data_root, split="val", transform=test_transform)
    else:
        train_set = df(
            root=args.data_root, split="train", download=True, transform=train_transform
        )
        test_set = df(
            root=args.data_root, split="val", download=True, transform=test_transform
        )

    train_set = standardize_dataset(train_set)
    test_set = standardize_dataset(test_set)
    return train_set, test_set


def get_combined_continual_learning_datasets_with_extra_held_out_set(args):
    # deal with dataset
    dataset_list, _ = deal_with_dataset(args)

    # deal with each dataset
    train_datasets = []
    test_datasets = []
    for i, dataset in enumerate(dataset_list):
        train_set, test_set = single_dataset_build(args, dataset)
        train_datasets.append(train_set)
        test_datasets.append(test_set)

    held_out_dataset = args.held_out_dataset
    held_out_train_set, held_out_test_set = single_dataset_build(args, held_out_dataset)

    np.random.seed(0)
    held_out_train_sets_of_current_stage = []
    for i, dataset in enumerate(train_datasets):
        dataset_length = len(dataset)
        current_indices = list(range(dataset_length))
        held_out_dataset_length = len(held_out_train_set)
        held_out_indices = (
            np.random.choice(held_out_dataset_length, size=dataset_length)
            + dataset_length
        )
        concated_dataset = ConcatWithTextDataset([dataset, held_out_train_set])
        held_out_train_sets_of_current_stage.append(
            SubsampleDataset(
                concated_dataset, current_indices + held_out_indices.tolist()
            )
        )

    training_datasets_of_current_stage = remap_datasets(train_datasets)
    test_datasets_of_current_stage = remap_datasets(test_datasets)
    # test_datasets_of_previous_stage = compile_datasets(test_datasets, mode='pre_cumulative')
    # test_datasets_of_previous_stage = [None] + test_datasets_of_previous_stage[:-1]

    accumulated_train_datasets = compile_datasets(train_datasets, mode="pre_cumulative")

    return (
        training_datasets_of_current_stage,
        test_datasets_of_current_stage,
        held_out_train_sets_of_current_stage,
        accumulated_train_datasets,
    )
