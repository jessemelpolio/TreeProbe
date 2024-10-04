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
        MemoryModule,
        CLIPModule,
        MixModel,
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
    incremental_train_dataset, concat_test_dataset = get_hdf5_continual_learning_dataset(
        args
    )
    _, held_out_test_datasets = get_hdf5_held_out_dataset(args)

    acc_list = []
    for i in range(incremental_train_dataset.num_stages):
        print("Stage {}".format(i))
        if hasattr(model, "retrieval_branch"):
            model.retrieval_branch.extend_memory(incremental_train_dataset)

        acc = engine.evaluate(
            concat_test_dataset, epoch=i, evaluation_tags=["target_dataset"], stage=i
        )
        acc_list.append(acc["target_dataset"]["overall"])
        held_out_acc = []
        for j in range(len(held_out_test_datasets)):
            acc = engine.evaluate(
                [held_out_test_datasets[j]],
                epoch=j,
                evaluation_tags=["zero_shot_dataset"],
                stage=i,
            )
            held_out_acc.append(acc["zero_shot_dataset"]["overall"])
        acc_list.append(np.mean(held_out_acc))
        incremental_train_dataset.forward_stage()

    with open(os.path.join(args.results_dir, args.csv_file), "a") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(acc_list)
