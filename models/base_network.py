import torch
import torch.nn as nn
import clip
from typing import List


class BaseNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.clip_encoder = None

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def encode_class_features(self, label_texts: List[str]):
        if not label_texts:
            return None

        tokenizer = clip.tokenize
        all_features = []

        for i in range(0, len(label_texts), self.args.batch_size):
            batch = label_texts[i:i + self.args.batch_size]
            labels = tokenizer(batch).to(self.device)
            
            with torch.no_grad():
                features = self.clip_encoder.encode_text(labels)
            
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features)

        self.class_features = torch.cat(all_features, dim=0)
        return self.class_features
