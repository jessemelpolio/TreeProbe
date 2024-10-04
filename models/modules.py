import torch
import torch.nn as nn
import numpy as np


class DynamicModule(nn.Module):
    """Dynamic Modules are Avalanche modules that can be incrementally
    expanded to allow architectural modifications (multi-head
    classifiers, progressive networks, ...).
    Compared to pytoch Modules, they provide an additional method,
    `model_adaptation`, which adapts the model given the current experience.
    """

    def adaptation(self, classes_in_this_experience):
        """Adapt the module (freeze units, add units...) using the current
        data. Optimizers must be updated after the model adaptation.
        Avalanche strategies call this method to adapt the architecture
        *before* processing each experience. Strategies also update the
        optimizer automatically.
        :param classes_in_this_experience: number of classes in this experience.
        :return:
        """
        if self.training:
            self.train_adaptation(classes_in_this_experience)
        else:
            self.eval_adaptation(classes_in_this_experience)

    def train_adaptation(self, classes_in_this_experience):
        """Module's adaptation at training time."""
        pass

    def eval_adaptation(self, classes_in_this_experience):
        """Module's adaptation at evaluation time."""
        pass


# Adapted from Avalanche
class IncrementalClassifier(DynamicModule):
    """
    Output layer that incrementally adds units whenever new classes are
    encountered.

    Typically used in class-incremental benchmarks where the number of
    classes grows over time.
    """

    def __init__(
        self,
        in_features,
        initial_out_features=2,
        masking=True,
        mask_value=-1000,
    ):
        """
        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        :param masking: whether unused units should be masked (default=True).
        :param mask_value: the value used for masked units (default=-1000).
        """
        super().__init__()
        self.masking = masking
        self.mask_value = mask_value

        self.classifier = torch.nn.Linear(in_features, initial_out_features)
        au_init = torch.zeros(initial_out_features, dtype=torch.bool)
        self.register_buffer("active_units", au_init)

    @torch.no_grad()
    def adaptation(self, dataset_in_this_experience):
        """If `dataset` contains unseen classes the classifier is expanded.

        :param classes_in_this_experience: class indices from the current experience.
        :return:
        """
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        curr_classes = list(dataset_in_this_experience.class_to_idx.values())
        print("self.classifier.out_features", self.classifier.out_features)
        print("max(curr_classes) + 1", max(curr_classes) + 1)
        new_nclasses = max(self.classifier.out_features, max(curr_classes) + 1)

        # update active_units mask
        if self.masking:
            if old_nclasses != new_nclasses:  # expand active_units mask
                old_act_units = self.active_units
                self.active_units = torch.zeros(new_nclasses, dtype=torch.bool)
                self.active_units[: old_act_units.shape[0]] = old_act_units
            # update with new active classes
            if self.training:
                self.active_units[curr_classes] = 1

        # update classifier weights
        if old_nclasses == new_nclasses:
            return
        old_w, old_b = self.classifier.weight, self.classifier.bias
        self.classifier = torch.nn.Linear(in_features, new_nclasses).to(old_w.device)
        self.classifier.weight[:old_nclasses] = old_w
        self.classifier.bias[:old_nclasses] = old_b

    def forward(self, x):
        """compute the output given the input `x`. This module does not use
        the task label.

        :param x:
        :return:
        """
        out = self.classifier(x)
        if self.masking:
            out[..., torch.logical_not(self.active_units)] = self.mask_value
        return out


# Adapted from https://github.com/mlfoundations/wise-ft/blob/58b7a4b343b09dc06606aa929c2ef51accced8d1/src/models/modeling.py#L35
class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


class PytorchClassifierAsSklearnClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, label_to_class):
        super().__init__()
        assert out_dim == len(label_to_class)
        self.model = nn.Linear(in_dim, out_dim)
        self.label_to_class = label_to_class

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        # x is a numpy array
        x = torch.tensor(x).to(self.model.weight.device)
        pred_labels = self.model(x).argmax(dim=-1).cpu().numpy()
        return np.array([self.label_to_class[label] for label in pred_labels])
