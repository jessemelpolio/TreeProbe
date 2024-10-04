import os.path
import csv
import numpy as np
from options.base_options import BaseOptions
from torch.utils.tensorboard import SummaryWriter
from models.memory_module import MemoryModule
from models.clip_module import CLIPModule
from models.mix_model import MixModel
from engines.tune_engine import TuneEngine
from data.image_dataset import (
    get_image_continual_learning_dataset,
    get_image_held_out_dataset,
    get_conceptual_captions_dataset,
    get_image_dataset_with_name,
)

from data.scenario_image import (
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
        MemoryModule,
        CLIPModule,
        MixModel,
        TuneEngine,
    ]
    args = opt.parse(module_list, is_train=True)

    learnable_params = []
    model = None

    # create engine
    engine = TuneEngine(args, model)
    learnable_params.append("model")

    # resume or load model
    if args.resume:
        engine.resume(args.resume_ckpt)

    # create logger
    logger = SummaryWriter(log_dir=args.log_dir)
    engine.logger = logger

    # create datasets
    # _, incremental_test_dataset = get_continual_learning_dataset(args)
    # create image datasets
    (
        incremental_train_dataset,
        incremental_test_dataset,
    ) = get_image_continual_learning_dataset(args)
    # create held out datasets
    _, held_out_test_datasets = get_image_held_out_dataset(args)

    if isinstance(incremental_test_dataset, list):
        eval_tags = [
            "test_dataset_{}".format(itd.name) for itd in incremental_test_dataset
        ]
    else:
        eval_tags = ["test_dataset_{}".format(incremental_test_dataset.name)]
        incremental_test_dataset = [incremental_test_dataset]

    acc_list = []
    for i in range(incremental_train_dataset.num_stages):
        print("Stage {}".format(i))
        # fit model
        engine.fit(
            incremental_train_dataset,
            param_keys=learnable_params,
            requires_grad=True,
            test_datasets=incremental_test_dataset,
            evaluation_tags=eval_tags,
            stage=i,
        )

        target_acc = []
        for j in range(len(incremental_test_dataset)):
            acc = engine.evaluate(
                [incremental_test_dataset[j]],
                epoch=i,
                evaluation_tags=["target_dataset"],
                stage=i,
            )
            target_acc.append(acc["target_dataset"]["overall"])
        acc_list.append(np.mean(target_acc))

        held_out_acc = []
        for stage in range(len(held_out_test_datasets)):
            acc = engine.evaluate(
                [held_out_test_datasets[stage]],
                epoch=i,
                evaluation_tags=["zero_shot_dataset"],
                stage=i,
            )
            held_out_acc.append(acc["zero_shot_dataset"]["overall"])
        acc_list.append(np.mean(held_out_acc))

        incremental_train_dataset.forward_stage()

    with open(os.path.join(args.results_dir, args.csv_file), "a") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(acc_list)

    # Flexible inference
    target_task = get_target_task(args)
    zero_shot_task = get_zero_shot_task(args)
    union_task = get_union_task(args)
    union_zero_shot_task = get_union_zero_shot_task(args)
    mix_task = get_mix_task(args)
    mix_zero_shot_task = get_mix_zero_shot_task(args)

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