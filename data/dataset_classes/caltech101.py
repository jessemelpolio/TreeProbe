from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive


class Caltech101(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, seed=0):
        super(Caltech101, self).__init__()
        assert split in ['train', 'val']
        self.root = os.path.join(root, 'caltech101')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download(root=self.root)
        
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {
            "Faces": "Faces_2",
            "Faces_easy": "Faces_3",
            "Motorbikes": "Motorbikes_16",
            "airplanes": "Airplanes_Side_2",
        }
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))
        self.class_to_idx = {c: idx for idx, c in enumerate(self.annotation_categories)}
        self.idx_to_class = {idx: c for idx, c in enumerate(self.annotation_categories)}
        self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        
        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

        np.random.seed(seed=seed)
        random_indices = np.random.permutation(len(self.index))
        
        self.train_val_test_split = {"train": random_indices[:int(0.7 * len(random_indices))],
                                     "val": random_indices[int(0.7 * len(random_indices)):]}

        self.samples = []
        self.targets = []
        self.text_targets = []
        
        for index in self.train_val_test_split[self.split]:
            self.samples.append(os.path.join("101_ObjectCategories", self.categories[self.y[index]], f"image_{self.index[index]:04d}.jpg"))
            target: Any = []
            target.append(self.y[index])
            target = tuple(target) if len(target) > 1 else target[0]
            self.targets.append(target)
            self.text_targets.append(self.idx_to_class[target])

    def __len__(self) -> int:
        return len(self.samples)
    
    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))
    
    def download(self, root) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
            root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9",
        )
        download_and_extract_archive(
            "https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m",
            root,
            filename="Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91",
        )
        
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