from data.image_dataset import (
    ConcatImageDataset,
    ZeroShotImageDataset,
    MixImageDataset,
    DatasetIncrementalImageDataset,
    single_dataset_build
)
from data.combined_dataset import deal_with_dataset


# returned is a list of datasets, each dataset is a single dataset without be wrapped with ConcatImageDataset
def get_naive_task(args, dataset_list):
    # deal with each dataset
    test_datasets = []
    for dataset in dataset_list:
        _, test_dataset = single_dataset_build(args, dataset)
        test_datasets.append(test_dataset)

    return test_datasets


def get_target_task(args):
    dataset_str = args.datasets
    dataset_list = dataset_str.split(",")
    test_datasets = get_naive_task(args, dataset_list)
    test_datasets = [ConcatImageDataset([d]) for d in test_datasets]
    return test_datasets


def get_zero_shot_task(args):
    dataset_str = args.held_out_dataset
    dataset_list = dataset_str.split(",")
    test_datasets = get_naive_task(args, dataset_list)
    test_datasets = [ConcatImageDataset([d]) for d in test_datasets]
    return test_datasets


def get_union_task(args):
    dataset_list, _ = deal_with_dataset(args)
    test_datasets = get_naive_task(args, dataset_list)
    return DatasetIncrementalImageDataset(test_datasets)


def get_union_zero_shot_task(args):
    target_dataset_str = args.datasets
    tatget_dataset_list = target_dataset_str.split(",")
    zero_shot_dataset_str = args.held_out_dataset
    zero_shot_dataset_list = zero_shot_dataset_str.split(",")

    target_test_datasets = get_naive_task(args, tatget_dataset_list)
    zero_shot_test_datasets = get_naive_task(args, zero_shot_dataset_list)

    union_task = DatasetIncrementalImageDataset(target_test_datasets)
    zero_shot_task = ZeroShotImageDataset(
        zero_shot_test_datasets,
        # num_classes=len(union_task.classes),
        num_classes=100,
        padding=len(union_task.classes),
    )

    all_classes = union_task.classes + zero_shot_task.classes
    all_classes_with_prompt = (
        union_task.classes_with_prompt + zero_shot_task.classes_with_prompt
    )
    union_task.classes = all_classes
    union_task.classes_with_prompt = all_classes_with_prompt
    zero_shot_task.classes = all_classes
    zero_shot_task.classes_with_prompt = all_classes_with_prompt

    # print("all_classes: ", all_classes)
    # print("all_classes_with_prompt: ", all_classes_with_prompt)

    return union_task, zero_shot_task


def get_mix_task(args):
    dataset_list, _ = deal_with_dataset(args)
    test_datasets = get_naive_task(args, dataset_list)
    return MixImageDataset(test_datasets)


# TODO: this is not consistent with our implication
def get_mix_zero_shot_task_original(args):
    target_dataset_str = args.datasets
    tatget_dataset_list = target_dataset_str.split(",")
    zero_shot_dataset_str = args.held_out_dataset
    zero_shot_dataset_list = zero_shot_dataset_str.split(",")

    target_test_datasets = get_naive_task(args, tatget_dataset_list)
    zero_shot_test_datasets = get_naive_task(args, zero_shot_dataset_list)

    mix_task = MixImageDataset(target_test_datasets)
    num_classes = len(mix_task.classes)
    zero_shot_task = ZeroShotImageDataset(
        zero_shot_test_datasets, num_classes, num_classes
    )

    all_classes = mix_task.classes + zero_shot_task.zero_shot_classes
    all_classes_with_prompt = (
        mix_task.classes_with_prompt + zero_shot_task.zero_shot_classes_with_prompt
    )
    mix_task.classes = all_classes
    mix_task.classes_with_prompt = all_classes_with_prompt
    zero_shot_task.classes = all_classes
    zero_shot_task.classes_with_prompt = all_classes_with_prompt

    return mix_task, zero_shot_task


def get_mix_zero_shot_task(args):
    target_dataset_str = args.datasets
    tatget_dataset_list = target_dataset_str.split(",")
    zero_shot_dataset_str = args.held_out_dataset
    zero_shot_dataset_list = zero_shot_dataset_str.split(",")

    target_test_datasets = get_naive_task(args, tatget_dataset_list)
    zero_shot_test_datasets = get_naive_task(args, zero_shot_dataset_list)

    mix_task = MixImageDataset(target_test_datasets)
    num_classes = len(mix_task.classes)
    zero_shot_task = ZeroShotImageDataset(
        zero_shot_test_datasets, num_classes, num_classes
    )

    all_classes = mix_task.classes + zero_shot_task.zero_shot_classes
    all_classes_with_prompt = (
        mix_task.classes_with_prompt + zero_shot_task.zero_shot_classes_with_prompt
    )
    # mix_task.classes = all_classes
    # mix_task.classes_with_prompt = all_classes_with_prompt
    # If this is the first stage, we already add classes and classes_with_prompt in add_additional_classes
    mix_task.add_additional_classes(zero_shot_task.zero_shot_classes, zero_shot_task.zero_shot_classes_with_prompt)

    zero_shot_task.classes = all_classes
    zero_shot_task.classes_with_prompt = all_classes_with_prompt

    return mix_task, zero_shot_task
