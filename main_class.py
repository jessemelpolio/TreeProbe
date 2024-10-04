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

    overall_acc_list = []
    seen_acc_list = []
    unseen_acc_list = []
    overall_loss_list = []

    print(f"Training on {incremental_train_dataset.name}, total stages: {incremental_train_dataset.num_stages}")
    for i in range(incremental_train_dataset.num_stages):
        print(f"Stage {i}")
        if hasattr(model, "retrieval_branch"):
            model.retrieval_branch.extend_memory(incremental_train_dataset)
        target_acc = {"overall": [], "seen": [], "unseen": [], "overall_loss": []}
        acc = engine.evaluate(
            [incremental_test_dataset],
            epoch=args.n_epochs,
            evaluation_tags=["target_dataset"],
            evaluate_seen_unseen=True,
            stage=i,
        )
        target_acc["overall"].append(acc["target_dataset"]["overall"])
        target_acc["seen"].append(acc["target_dataset"]["seen"])
        target_acc["unseen"].append(acc["target_dataset"]["unseen"])
        target_acc["overall_loss"].append(acc["target_dataset"]["overall_loss"])
        overall_acc_list.append(np.mean(target_acc["overall"]))
        seen_acc_list.append(np.mean(target_acc["seen"]))
        unseen_acc_list.append(np.mean(target_acc["unseen"]))
        overall_loss_list.append(np.mean(target_acc["overall_loss"]))

        # held_out_acc = []
        held_out_acc = {"overall": [], "seen": [], "unseen": [], "overall_loss": []}
        for j in range(len(held_out_test_datasets)):
            acc = engine.evaluate(
                [held_out_test_datasets[j]],
                epoch=args.n_epochs,
                evaluation_tags=["zero_shot_dataset"],
                evaluate_seen_unseen=False,
                stage=i,
            )
            held_out_acc["seen"].append(acc["zero_shot_dataset"]["overall"])
            held_out_acc["unseen"].append(acc["zero_shot_dataset"]["overall"])
            held_out_acc["overall"].append(acc["zero_shot_dataset"]["overall"])
            held_out_acc["overall_loss"].append(
                acc["zero_shot_dataset"]["overall_loss"]
            )
        overall_acc_list.append(np.mean(held_out_acc["overall"]))
        seen_acc_list.append(np.mean(held_out_acc["seen"]))
        unseen_acc_list.append(np.mean(held_out_acc["unseen"]))
        overall_loss_list.append(np.mean(held_out_acc["overall_loss"]))

        incremental_test_dataset.forward_stage()
        incremental_train_dataset.forward_stage()

    with open(os.path.join(args.results_dir, args.csv_file), "a") as outfile:
        writer = csv.writer(outfile)
        writer.writerow([incremental_train_dataset.name, "overall"] + overall_acc_list)
        writer.writerow([incremental_train_dataset.name, "seen"] + seen_acc_list)
        writer.writerow([incremental_train_dataset.name, "unseen"] + unseen_acc_list)
        writer.writerow(
            [incremental_train_dataset.name, "overall_loss"] + overall_loss_list
        )

