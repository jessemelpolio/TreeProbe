import torch
import torch.nn.functional as F


class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self, weight=1e6):
        super().__init__()
        self.weight = weight
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        return (1 - self.cosine_similarity(x, y)).mean()  # * self.weight


class CosineSimilarityWithThresholdMinimizationLoss(torch.nn.Module):
    def __init__(self, weight=1e6):
        super().__init__()
        self.weight = weight
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, x, y, att_mean):
        return (1 - self.cosine_similarity(x, y)).mean() + (
            att_mean - 0.5
        )  # * self.weight


class AttentionLossWithMainLoss(torch.nn.Module):
    def __init__(self, main_loss, weight=100):
        super().__init__()
        self.main_loss = main_loss
        self.weight = weight

    def forward(self, x, targets, text_targets, att_mask, retrieved_text):
        # retrieved_text is B, K, D, text_targets is B, D
        # targets and text_targets can be the same
        main_loss = self.main_loss(x, targets)
        # retrieved_text = retrieved_text[:, 1:, :]
        att_targets = torch.where(
            text_targets.unsqueeze(1) @ retrieved_text.transpose(1, 2) > 0.9, 1.0, 0.0
        ).squeeze()
        # print("att_targets", att_targets.sum(-1))
        att_loss = F.l1_loss(att_mask.squeeze(), att_targets)
        return main_loss + self.weight * att_loss


class CrossEntropyLossWithL1Regularization(torch.nn.Module):
    def __init__(self, weight=1e6):
        super().__init__()
        self.weight = weight
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, x, y, alpha):
        l1_norm = alpha.abs().sum()
        return self.ce(x, y) + self.weight * l1_norm


class OpenSetCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, x, y, label_features):
        y_pred = 100 * x @ label_features.T
        return self.ce(y_pred, y)
