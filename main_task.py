import os.path
import csv
import numpy as np
from options.base_options import BaseOptions
from torch.utils.tensorboard import SummaryWriter
from models.memory_module import MemoryModule
from models.clip_module import CLIPModule
from models.mix_model import MixModel
from engines.main_engine import MainEngine
from data.HDF5_dataset import get_hdf5_continual_learning_dataset, get_hdf5_held_out_dataset
from data.scenario import (
    get_target_task,
    get_union_task,
    get_zero_shot_task,
    get_union_zero_shot_task,
    get_mix_task,
    get_mix_zero_shot_task,
)


def seed_everything(seed):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_everything(42)
    # Parse arguments
    opt = BaseOptions()
    module_list = [
        MixModel,
        MemoryModule,
        CLIPModule,
        MainEngine
    ]
    args = opt.parse(module_list, is_train=True)

    # create model
    model = MixModel(args)
    model = model.to(args.device)
    # print(model)

    learnable_params = []

    # create engine
    engine = MainEngine(args, model)

    # resume or load model
    if args.resume:
        engine.resume(args.resume_ckpt)

    # create logger
    logger = SummaryWriter(log_dir=args.results_dir)
    engine.logger = logger

    # create datasets
    (
        incremental_train_dataset,
        incremental_test_dataset,
    ) = get_hdf5_continual_learning_dataset(args)
    _, held_out_test_datasets = get_hdf5_held_out_dataset(args)

    # Flexible inference
    target_task = get_target_task(args)
    zero_shot_task = get_zero_shot_task(args)
    union_task = get_union_task(args)
    union_zero_shot_task = get_union_zero_shot_task(args)
    mix_task = get_mix_task(args)
    mix_zero_shot_task = get_mix_zero_shot_task(args)

    # overall_acc_list = []
    # current_acc_list = []
    # past_acc_list = []
    overall_acc_list = []

    # columns represents stages and rows represents datasets
    overall_acc_array = np.zeros(
        (len(incremental_test_dataset), len(incremental_test_dataset))
    )
    heldout_acc_array = np.zeros(
        (len(held_out_test_datasets), len(incremental_test_dataset))
    )
    for i in range(incremental_train_dataset.num_stages):
        print(f"Stage {i}")
        if hasattr(model, "retrieval_branch"):
            model.retrieval_branch.extend_memory(incremental_train_dataset)
        acc_list = []
        for j in range(len(incremental_test_dataset)):
            acc = engine.evaluate(
                [incremental_test_dataset[j]],
                epoch=args.n_epochs,
                evaluation_tags=["target_dataset"],
                # evaluate_current_past=True,
                stage=i,
            )
            overall_acc_array[j, i] = acc["target_dataset"]["overall"]
            acc_list.append(acc["target_dataset"]["overall"])
        overall_acc_list.append(np.mean(acc_list))  

        acc_list = []
        for j in range(len(held_out_test_datasets)):
            acc = engine.evaluate(
                [held_out_test_datasets[j]],
                epoch=args.n_epochs,
                evaluation_tags=["zero_shot_dataset"],
                # evaluate_current_past=True,
                stage=i,
            )
            heldout_acc_array[j, i] = acc["zero_shot_dataset"]["overall"]
            acc_list.append(acc["zero_shot_dataset"]["overall"])
        overall_acc_list.append(np.mean(acc_list))

        incremental_train_dataset.forward_stage()

    np.savetxt(
        os.path.join(args.results_dir, "overall_acc_array.csv"),
        overall_acc_array,
        fmt="%.3e",
        delimiter=",",
    )
    np.savetxt(
        os.path.join(args.results_dir, "heldout_acc_array.csv"),
        heldout_acc_array,
        fmt="%.3e",
        delimiter=",",
    )

    with open(os.path.join(args.results_dir, args.csv_file), "a") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(overall_acc_list)

    # Target task
    target_acc = []
    for i in range(len(target_task)):
        acc = engine.evaluate(
            [target_task[i]], epoch=i, evaluation_tags=["target_dataset"], stage=i
        )
        target_acc.append(acc["target_dataset"]["overall"])
    print("Target task: ", np.mean(target_acc))
    task_acc = [np.mean(target_acc)]

    # Zero-shot task
    zero_shot_acc = []
    for i in range(len(zero_shot_task)):
        acc = engine.evaluate(
            [zero_shot_task[i]], epoch=i, evaluation_tags=["zero_shot_dataset"], stage=i
        )
        zero_shot_acc.append(acc["zero_shot_dataset"]["overall"])
    print("Zero-shot task: ", np.mean(zero_shot_acc))
    task_acc.append(np.mean(zero_shot_acc))

    # Union task
    union_acc = []
    for i in range(union_task.num_stages):
        acc = engine.evaluate(
            [union_task], epoch=i, evaluation_tags=["union_dataset"], stage=i
        )
        union_acc.append(acc["union_dataset"]["overall"])
        union_task.forward_stage()
    print("Union task: ", np.mean(union_acc))
    task_acc.append(np.mean(union_acc))

    # Union zero-shot task
    union_task_combining_zs_labels_acc = []
    for i in range(union_zero_shot_task[0].num_stages):
        acc = engine.evaluate(
            [union_zero_shot_task[0]],
            epoch=i,
            evaluation_tags=["union_dataset"],
            stage=i,
        )
        union_task_combining_zs_labels_acc.append(acc["union_dataset"]["overall"])
        union_zero_shot_task[0].forward_stage()

    acc = engine.evaluate(
        [union_zero_shot_task[1]],
        epoch=i,
        evaluation_tags=["zero_shot_dataset"],
        stage=i,
    )
    print(
        "Union zero-shot task: ",
        np.mean([np.mean(union_task_combining_zs_labels_acc), acc["zero_shot_dataset"]["overall"]]),
    )
    task_acc.append(np.mean([np.mean(union_task_combining_zs_labels_acc), acc["zero_shot_dataset"]["overall"]]))

    # Mix task
    mix_task_acc = []
    for i in range(mix_task.num_stages):
        acc = engine.evaluate(
            [mix_task], epoch=i, evaluation_tags=["mix_dataset"], stage=i
        )
        mix_task_acc.append(acc["mix_dataset"]["overall"])
        mix_task.forward_stage()
    print("Mix task: ", np.mean(mix_task_acc))
    task_acc.append(np.mean(mix_task_acc))

    # Mix zero-shot task
    mix_task_acc = []
    for i in range(mix_task.num_stages):
        acc = engine.evaluate(
            [mix_zero_shot_task[0]], epoch=i, evaluation_tags=["mix_dataset"], stage=i
        )
        mix_task_acc.append(acc["mix_dataset"]["overall"])
        mix_zero_shot_task[0].forward_stage()
    print("Mix zero-shot task: ", np.mean(mix_task_acc))
    task_acc.append(np.mean(mix_task_acc))
    # mix_zero_shot_acc = [np.mean(mix_task_acc)]
    # acc = engine.evaluate(
    #     [mix_zero_shot_task[1]], epoch=i, evaluation_tags=["zero_shot_dataset"], stage=i
    # )
    # mix_zero_shot_acc.append(acc["zero_shot_dataset"]["overall"])
    # print("Mix zero-shot task: ", np.mean(mix_zero_shot_acc))
    # task_acc.append(np.mean(mix_zero_shot_acc))

    if not os.path.isdir(os.path.join(args.results_dir, "flexible_inference")):
        os.makedirs(os.path.join(args.results_dir, "flexible_inference"))

    with open(
        os.path.join(args.results_dir, "flexible_inference", args.csv_file), "a"
    ) as outfile_2:
        writer_2 = csv.writer(outfile_2)
        writer_2.writerow(
            ["target", "zero_shot", "union", "union_zero_shot", "mix", "mix_zero_shot"]
        )
        writer_2.writerow(task_acc)
