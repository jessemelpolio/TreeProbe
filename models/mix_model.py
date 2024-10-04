import torch
from .base_network import BaseNetwork
from .clip_module import CLIPModule
from .memory_module import MemoryModule


class MixModel(BaseNetwork):
    def __init__(self, args):
        super().__init__(args)
        self.device = args.device
        self.clip_branch = CLIPModule(args)
        self.retrieval_branch = MemoryModule(args, self.clip_branch.clip_encoder)
        self.clip_encoder = self.clip_branch.clip_encoder
        self.mix_mode = args.mix_mode
        self.runtime_get_dim()

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--mix_mode", type=str, default="complementary", help="Mix mode", choices=["complementary", "clip_only"])
        parser.parse_known_args()
        return parser

    @torch.no_grad()
    def runtime_get_dim(self):
        if "336px" in self.args.backbone:
            tensor = torch.randn(1, 3, 336, 336).to(self.device)
        else:
            tensor = torch.randn(1, 3, 224, 224).to(self.device)
        out = self.clip_encoder.encode_image(tensor)
        if "concat" in self.mix_mode:
            self.enc_dim = out.shape[-1] * 2
        else:
            self.enc_dim = out.shape[-1]

    def forward(self, x):
        # If x has 4 dimensions, it is a batch of images
        if len(x.shape) == 4:
            output_clip = self.clip_branch(x)
        # If x has 3 dimensions, it is a batch of encoded images
        elif len(x.shape) == 2:
            output_clip = x.float()
        else:
            raise NotImplementedError

        if self.mix_mode == "clip_only":
            return output_clip / output_clip.norm(dim=-1, keepdim=True)
        if self.mix_mode == "complementary":
            return self.retrieval_branch(output_clip)
