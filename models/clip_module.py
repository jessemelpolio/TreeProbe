import torch
from .base_network import BaseNetwork
import clip
import torch.nn as nn
import open_clip


class CLIPModule(BaseNetwork):
    def __init__(self, args):
        super().__init__(args)
        self.device = args.device
        if self.args.backbone == "ViT-H/14":
            self.clip_encoder = (
                open_clip.create_model_and_transforms(
                    "ViT-H-14", pretrained="laion2b_s32b_b79k", device=self.device
                )[0]
                .eval()
                .requires_grad_(False)
            )
        else:
            self.clip_encoder = (
                clip.load(args.backbone, jit=False, device=self.device)[0]
                .eval()
                .requires_grad_(False)
            )
        self.clip_encoder = self.clip_encoder.float()
        self.runtime_get_dim()

    @torch.no_grad()
    def runtime_get_dim(self):
        if "336px" in self.args.backbone:
            tensor = torch.randn(1, 3, 336, 336).to(self.device)
        else:
            tensor = torch.randn(1, 3, 224, 224).to(self.device)
        out = self.clip_encoder.encode_image(tensor)
        self.enc_dim = out.shape[-1]

    def forward(self, x):
        return self.clip_encoder.encode_image(x)
