import numpy as np
import torch


def _return_array(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy() if x.is_cuda else x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError("image_features should be torch.Tensor or numpy.ndarray")


class EncodedFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, image_features, label_features, labels):
        self.image_features = _return_array(image_features)
        self.label_features = _return_array(label_features)
        self.labels = _return_array(labels)
        # remap labels to 0, 1, 2, ...
        self.label2idx = {}
        for i, label in enumerate(np.unique(self.labels)):
            self.label2idx[label] = i
        self.labels = np.array([self.label2idx[label] for label in self.labels])
        # according to label2idx, select unique label features
        self.unique_label_features = np.zeros(
            (len(self.label2idx), self.label_features.shape[1])
        )
        for idx in self.label2idx.values():
            self.unique_class_features[idx] = self.label_features[self.labels == idx][0]

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.image_features[index]),
            torch.tensor(self.labels[index]).long(),
            torch.from_numpy(self.label_features[index]),
        )

    def __len__(self):
        return self.image_features.shape[0]
