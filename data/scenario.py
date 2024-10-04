from data.HDF5_dataset import (
    load_hdf5_files_to_datasets,
    ConcatHDF5Dataset,
    ZeroShotHDF5Dataset,
    MixHDF5Dataset,
    DatasetIncrementalHDF5Dataset,
)
from data.combined_dataset import deal_with_dataset


def get_target_task(args):
    dataset_list, _ = deal_with_dataset(args)
    # hdf5_folder = os.path.join(args.data_root, "hdf5")
    hdf5_folder = args.data_root
    test_datasets = []
    for dataset in dataset_list:
        _, test_dataset = load_hdf5_files_to_datasets([dataset], hdf5_folder)
        test_datasets.append(ConcatHDF5Dataset(test_dataset, [dataset]))

    return test_datasets


def get_zero_shot_task(args):
    dataset_str = args.held_out_dataset
    dataset_list = dataset_str.split(",")
    # hdf5_folder = os.path.join(args.data_root, "hdf5")
    hdf5_folder = args.data_root
    test_datasets = []
    for dataset in dataset_list:
        _, test_dataset = load_hdf5_files_to_datasets([dataset], hdf5_folder)
        test_datasets.append(ConcatHDF5Dataset(test_dataset, [dataset]))

    return test_datasets


def get_union_task(args):
    dataset_list, _ = deal_with_dataset(args)
    # hdf5_folder = os.path.join(args.data_root, "hdf5")
    hdf5_folder = args.data_root
    _, test_datasets = load_hdf5_files_to_datasets(dataset_list, hdf5_folder)

    return DatasetIncrementalHDF5Dataset(test_datasets, dataset_list)


def get_union_zero_shot_task(args):
    tatget_dataset_list, _ = deal_with_dataset(args)
    zero_shot_dataset_str = args.held_out_dataset
    zero_shot_dataset_list = zero_shot_dataset_str.split(",")
    # hdf5_folder = os.path.join(args.data_root, "hdf5")
    hdf5_folder = args.data_root
    _, target_test_datasets = load_hdf5_files_to_datasets(tatget_dataset_list, hdf5_folder)
    _, zero_shot_test_datasets = load_hdf5_files_to_datasets(zero_shot_dataset_list, hdf5_folder)

    union_task = DatasetIncrementalHDF5Dataset(target_test_datasets, tatget_dataset_list)
    zero_shot_task = ZeroShotHDF5Dataset(
        zero_shot_test_datasets,
        zero_shot_dataset_list,
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
    # hdf5_folder = os.path.join(args.data_root, "hdf5")
    hdf5_folder = args.data_root
    _, test_datasets = load_hdf5_files_to_datasets(dataset_list, hdf5_folder)

    return MixHDF5Dataset(test_datasets, dataset_list)


# TODO: This is incosistent with our implication
def get_mix_zero_shot_task_original(args):
    tatget_dataset_list, _ = deal_with_dataset(args)
    zero_shot_dataset_str = args.held_out_dataset
    zero_shot_dataset_list = zero_shot_dataset_str.split(",")
    # hdf5_folder = os.path.join(args.data_root, "hdf5")
    hdf5_folder = args.data_root
    _, target_test_datasets = load_hdf5_files_to_datasets(tatget_dataset_list, hdf5_folder)
    _, zero_shot_test_datasets = load_hdf5_files_to_datasets(zero_shot_dataset_list, hdf5_folder)

    mix_task = MixHDF5Dataset(target_test_datasets, tatget_dataset_list)
    num_classes = len(mix_task.classes)
    zero_shot_task = ZeroShotHDF5Dataset(
        zero_shot_test_datasets, zero_shot_dataset_list, num_classes, num_classes
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
    tatget_dataset_list, _ = deal_with_dataset(args)
    zero_shot_dataset_str = args.held_out_dataset
    zero_shot_dataset_list = zero_shot_dataset_str.split(",")
    # hdf5_folder = os.path.join(args.data_root, "hdf5")
    hdf5_folder = args.data_root
    _, target_test_datasets = load_hdf5_files_to_datasets(tatget_dataset_list, hdf5_folder)
    _, zero_shot_test_datasets = load_hdf5_files_to_datasets(zero_shot_dataset_list, hdf5_folder)

    mix_task = MixHDF5Dataset(target_test_datasets, tatget_dataset_list)
    num_classes = len(mix_task.classes)
    zero_shot_task = ZeroShotHDF5Dataset(
        zero_shot_test_datasets, zero_shot_dataset_list, num_classes, num_classes
    )

    all_classes = mix_task.classes + zero_shot_task.zero_shot_classes
    all_classes_with_prompt = (
        mix_task.classes_with_prompt + zero_shot_task.zero_shot_classes_with_prompt
    )
    # mix_task.classes = all_classes
    # mix_task.classes_with_prompt = all_classes_with_prompt
    mix_task.add_additional_classes(zero_shot_task.zero_shot_classes, zero_shot_task.zero_shot_classes_with_prompt)

    zero_shot_task.classes = all_classes
    zero_shot_task.classes_with_prompt = all_classes_with_prompt

    return mix_task, zero_shot_task
