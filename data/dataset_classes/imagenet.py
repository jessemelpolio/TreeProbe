from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive

class ImageNet(Dataset):
    def __init__(self, root, split='train', download=True, transform=None, target_transform=None):
        super(ImageNet, self).__init__()
        assert split in ['train', 'val', 'test']
        if root == '/data/owcl_data':
            self.root = os.path.join(os.path.dirname(root), 'imagenet')
        else:
            self.root = os.path.join(root, 'ILSVRC2012')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.number_to_classes = json.load(open('./data/dataset_classes/imagenet_classes.json', 'r'))     
        
        self.samples = []
        self.targets = []
        self.text_targets = []
        
        self.number = sorted(os.listdir(os.path.join(self.root, self.split)))
        
        self.idx_to_class = {idx: self.number_to_classes[self.number[idx]] for idx in range(len(self.number))}
        self.class_to_idx = {self.number_to_classes[self.number[idx]]: idx for idx in range(len(self.number))}
        self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        for idx, num in enumerate(sorted(os.listdir(os.path.join(self.root, self.split)))):
            for img in sorted(os.listdir(os.path.join(self.root, self.split, num))):
                self.samples.append(os.path.join(self.split, num, img))
                self.targets.append(idx)
                self.text_targets.append(self.number_to_classes[num])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.root, self.samples[index])
        img = Image.open(img_path).convert('RGB')
        target = self.targets[index]
        text_target = self.text_targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, text_target


# only return the folders in os.listdir(folder)
def filter_folders(folder):
    return [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]


class ImageNetA(Dataset):
    def __init__(self, root, split='train', download=True, transform=None, target_transform=None):
        super(ImageNetA, self).__init__()
        if root == '/data/owcl_data':
            self.root = os.path.join(os.path.dirname(root), 'imagenet-a')
        else:
            self.root = os.path.join(root, 'imagenet-a')
        self.transform = transform
        self.target_transform = target_transform

        self.number_to_classes = json.load(open('./data/dataset_classes/imagenet_classes.json', 'r'))

        # instead of using filter_folders, we SHOULD just use the number_to_classes dict since otherwise it will cause
        # errors in later code such as combined_datasets.py which assumes that all idx and class are contiguous.
        self.numbers = [cls for cls, name in self.number_to_classes.items()]
        self.numbers = sorted(self.numbers)

        self.idx_to_class = {idx: self.number_to_classes[self.numbers[idx]] for idx in range(len(self.numbers))}
        self.class_to_idx = {self.number_to_classes[self.numbers[idx]]: idx for idx in range(len(self.numbers))}
        self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        self.samples = []
        self.targets = []
        self.text_targets = []

        for idx, num in enumerate(sorted(os.listdir(self.root))):
            if not os.path.isdir(os.path.join(self.root, num)):
                print(f"Skipping {num} in ImageNetA")
                continue
            for img in sorted(os.listdir(os.path.join(self.root, num))):
                self.samples.append(os.path.join(num, img))
                self.targets.append(self.class_to_idx[self.number_to_classes[num]])
                self.text_targets.append(self.number_to_classes[num])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.root, self.samples[index])
        img = Image.open(img_path).convert('RGB')
        target = self.targets[index]
        text_target = self.text_targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, text_target


class ImageNetR(Dataset):
    def __init__(self, root, split='train', download=True, transform=None, target_transform=None):
        super(ImageNetR, self).__init__()
        if root == '/data/owcl_data':
            self.root = os.path.join(os.path.dirname(root), 'imagenet-r')
        else:
            self.root = os.path.join(root, 'imagenet-r')
        self.transform = transform
        self.target_transform = target_transform

        self.number_to_classes = json.load(open('./data/dataset_classes/imagenet_classes.json', 'r'))

        # instead of using filter_folders, we SHOULD just use the number_to_classes dict since otherwise it will cause
        # errors in later code such as combined_datasets.py which assumes that all idx and class are contiguous.
        self.numbers = [cls for cls, name in self.number_to_classes.items()]
        self.numbers = sorted(self.numbers)

        self.idx_to_class = {idx: self.number_to_classes[self.numbers[idx]] for idx in range(len(self.numbers))}
        self.class_to_idx = {self.number_to_classes[self.numbers[idx]]: idx for idx in range(len(self.numbers))}
        self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        self.samples = []
        self.targets = []
        self.text_targets = []

        for idx, num in enumerate(sorted(os.listdir(self.root))):
            if not os.path.isdir(os.path.join(self.root, num)):
                print(f"Skipping {num} in ImageNetR")
                continue
            for img in sorted(os.listdir(os.path.join(self.root, num))):
                self.samples.append(os.path.join(num, img))
                self.targets.append(self.class_to_idx[self.number_to_classes[num]])
                self.text_targets.append(self.number_to_classes[num])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.root, self.samples[index])
        img = Image.open(img_path).convert('RGB')
        target = self.targets[index]
        text_target = self.text_targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, text_target


from datasets import load_dataset
class ImageNetSketch(Dataset):
    def __init__(self, root, split='train', download=True, transform=None, target_transform=None):
        super(ImageNetSketch, self).__init__()
        self.dataset_core = load_dataset("imagenet_sketch")["train"]
        self.transform = transform
        self.target_transform = target_transform

        number_of_classes = len(self.dataset_core.features['label'].names)
        self.idx_to_class = {idx: self.dataset_core.features['label'].int2str(idx) for idx in range(number_of_classes)}
        self.class_to_idx = {self.dataset_core.features['label'].int2str(idx): idx for idx in range(number_of_classes)}
        self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        self.samples = []
        self.targets = self.dataset_core['label']
        self.text_targets = [self.idx_to_class[target] for target in self.targets]

    def __len__(self) -> int:
        return len(self.dataset_core)

    def __getitem__(self, index: int):
        instance_dict = self.dataset_core[index]
        img, target = instance_dict["image"], instance_dict["label"]
        text_target = self.text_targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, text_target
