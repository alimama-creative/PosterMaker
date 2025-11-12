import torch
from torch import nn
from .RNN import SequenceEncoder, Im2Seq, Im2Im
from .RecMv1_enhance import MobileNetV1Enhance

from .RecCTCHead import CTCHead

backbone_dict = {"MobileNetV1Enhance":MobileNetV1Enhance}
neck_dict = {'SequenceEncoder': SequenceEncoder, 'Im2Seq': Im2Seq,'None':Im2Im}
head_dict = {'CTCHead':CTCHead}


class RecModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert 'in_channels' in config, 'in_channels must in model config'
        backbone_type = config.backbone.pop('type')
        assert backbone_type in backbone_dict, f'backbone.type must in {backbone_dict}'
        self.backbone = backbone_dict[backbone_type](config.in_channels, **config.backbone)

        neck_type = config.neck.pop('type')
        assert neck_type in neck_dict, f'neck.type must in {neck_dict}'
        self.neck = neck_dict[neck_type](self.backbone.out_channels, **config.neck)

        head_type = config.head.pop('type')
        assert head_type in head_dict, f'head.type must in {head_dict}'
        self.head = head_dict[head_type](self.neck.out_channels, **config.head)

        self.name = f'RecModel_{backbone_type}_{neck_type}_{head_type}'

    def load_3rd_state_dict(self, _3rd_name, _state):
        self.backbone.load_3rd_state_dict(_3rd_name, _state)
        self.neck.load_3rd_state_dict(_3rd_name, _state)
        self.head.load_3rd_state_dict(_3rd_name, _state)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.neck(x1)
        x3 = self.head(x2)
        if isinstance(x3, dict):
            B, C, H, W = x1.shape
            x1 = torch.randn(B, C, H, W)
            x1 = x1.permute(0, 2, 3, 1)
            x1 = x1.reshape(B, H*W, C)
            x3['backbone'] = x1
        return x3

    def encode(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head.ctc_encoder(x)
        return x

