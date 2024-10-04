import clip
import torch
import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from data.image_dataset import (
    get_image_continual_testset_dataset,
    get_united_dataset,
)
from data.combined_dataset import get_open_world_single_datasets
from options.base_options import BaseOptions
import h5py
import numpy as np
from tqdm import tqdm


def get_clip_features(model, dataset):
    dl = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=0
    )
    image_features = []
    label_features = []
    normalized_image_features = []
    normalized_label_features = []
    labels = []
    for batch in tqdm(dl):
        image, label, label_text = batch
        image = image.to("cuda:0")
        label = label.to("cuda:0")
        with torch.no_grad():
            image_feat = model.encode_image(image)
            label_feat = model.encode_text(clip.tokenize(label_text).to("cuda:0"))
            normalized_image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            normalized_label_feat = label_feat / label_feat.norm(dim=-1, keepdim=True)
            image_features.append(image_feat)
            label_features.append(label_feat)
            normalized_image_features.append(normalized_image_feat)
            normalized_label_features.append(normalized_label_feat)
            labels.append(label)
    image_features = torch.cat(image_features, dim=0).cpu().numpy()
    label_features = torch.cat(label_features, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    normalized_image_features = (
        torch.cat(normalized_image_features, dim=0).cpu().numpy()
    )
    normalized_label_features = (
        torch.cat(normalized_label_features, dim=0).cpu().numpy()
    )

    return {
        "image_features": image_features,
        "label_features": label_features,
        "labels": labels,
        "normalized_image_features": normalized_image_features,
        "normalized_label_features": normalized_label_features,
    }
    

class CustomOptions:
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--store_folder', type=str, default='/home/nick/anytime_learning/intermediate_features_npy_layer_0', help='Path to store the encoded features')
        parser.add_argument('--subsets', type=str, default='train', choices=['train', 'test', 'train_test'], help='Subset to encode')
        parser.parse_known_args()
        return parser


if __name__ == "__main__":
    opt = BaseOptions()
    module_list = [CustomOptions]
    args = opt.parse(module_list, is_train=True)

    model, preprocess = clip.load("ViT-B/32", device="cuda:0")
    model = model.float()
    model.eval()
    cl_train_datasets, cl_test_datasets = get_open_world_single_datasets(args)

    if "train" in args.subsets:
        for dataset in cl_train_datasets:
            print(f"Processing dataset {dataset.name}")
            out_dict = get_clip_features(model, dataset)

            h5py_file_path = os.path.join(
                args.store_folder, "{}_train.hdf5".format(dataset.name.split("_")[1])
            )
            print(f"Saving to {h5py_file_path}")

            with h5py.File(h5py_file_path, "w") as hf:
                for key, vectors in out_dict.items():
                    # Convert the vectors to a NumPy array if they are not already
                    vectors = np.array(vectors)
                    hf.create_dataset(key, data=vectors)
                string_array = np.array(
                    dataset.original_classes, dtype=h5py.string_dtype(encoding="utf-8")
                )
                hf.create_dataset("original_classes", data=string_array)
                string_array = np.array(
                    dataset.classes, dtype=h5py.string_dtype(encoding="utf-8")
                )
                hf.create_dataset("classes", data=string_array)

    if "test" in args.subsets:
        for dataset in cl_test_datasets:
            print(f"Processing dataset {dataset.name}")
            out_dict = get_clip_features(model, dataset)
            # h5py_file_path = os.path.join(
            #     "/data/owcl_data/hdf5_ViT-L14@336px",
            #     "{}_test.hdf5".format(dataset.name.split("_")[1]),
            # )

            h5py_file_path = os.path.join(
                args.store_folder, "{}_test.hdf5".format(dataset.name.split("_")[1])
            )
            print(f"Saving to {h5py_file_path}")

            with h5py.File(h5py_file_path, "w") as hf:
                for key, vectors in out_dict.items():
                    # Convert the vectors to a NumPy array if they are not already
                    vectors = np.array(vectors)
                    hf.create_dataset(key, data=vectors)
                string_array = np.array(
                    dataset.original_classes, dtype=h5py.string_dtype(encoding="utf-8")
                )
                hf.create_dataset("original_classes", data=string_array)
                string_array = np.array(
                    dataset.classes, dtype=h5py.string_dtype(encoding="utf-8")
                )
                hf.create_dataset("classes", data=string_array)
