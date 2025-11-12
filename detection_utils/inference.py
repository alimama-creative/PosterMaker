# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm

from detection_utils.segment_anything_training import sam_model_registry
from detection_utils.segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

import json

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoderHQ(MaskDecoder):
    def __init__(self, model_type):
        super().__init__(transformer_dim=256,
                        transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=256,
                                mlp_dim=2048,
                                num_heads=8,
                            ),
                        num_multimask_outputs=3,
                        activation=nn.GELU,
                        iou_head_depth= 3,
                        iou_head_hidden_dim= 256)        
        transformer_dim=256
        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )

        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

        self.last_conv = nn.Sequential(
                            nn.Conv2d(35, 128, 3, 1, 1), 
                            LayerNorm2d(128),
                            nn.GELU(),
                            nn.Conv2d(128, 128, 3, 1, 1))
        self.score_mlp = MLP(128, 64, 1, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,
        mask_ori: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """
        
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        upscaled_embeds = []
        for i_batch in range(batch_len):
            mask, iou_pred, upscaled_embed = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature = hq_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
            upscaled_embeds.append(upscaled_embed)
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)
        upscaled_embeds = torch.cat(upscaled_embeds, 0)

        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1,self.num_mask_tokens-1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds,dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            masks_sam = masks[:,mask_slice]
        masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :]
        score_feat = self.last_conv(torch.cat([upscaled_embeds, masks_sam, mask_ori, F.sigmoid(masks_sam)-F.sigmoid(mask_ori)], 1))
        score = self.score_mlp(score_feat.view(batch_len, 128, -1).mean(2))
        return score.squeeze(-1)

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_ours],dim=1)
        
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred, upscaled_embedding_sam

def get_box_from_mask(mask):
    """get the coordinates of the tight box from the mask

    Args:
        mask (np.ndarray): 0-1 mask or bool-mask of shape [h, w] 
    """
    where_h, where_w = np.where(mask>0)
    y1, y2 = where_h.min(), where_h.max()
    x1, x2 = where_w.min(), where_w.max()
    return x1, y1, x2, y2

def enlarge_bounding_box(bbox, enlarge_percentage=0.1):
    """
    Enlarge the given bounding box by a specified percentage.

    Parameters:
    bbox (tuple): A bounding box represented by a tuple (x_min, y_min, x_max, y_max).
    enlarge_percentage (float): The percentage by which to enlarge the bounding box.

    Returns:
    tuple: An enlarged bounding box.
    """
    x_min, y_min, x_max, y_max = bbox
    # 计算bounding box的宽度和高度
    width = x_max - x_min
    height = y_max - y_min
    
    # 计算放大的像素值
    enlarge_width = width * enlarge_percentage / 2
    enlarge_height = height * enlarge_percentage / 2
    
    # 创建新的bounding box，其在每个方向上都放大了
    new_x_min = x_min - enlarge_width
    new_y_min = y_min - enlarge_height
    new_x_max = x_max + enlarge_width
    new_y_max = y_max + enlarge_height
    
    # 确保新的bounding box仍然在图片的范围内
    # 这里我们假设图片的坐标是从0开始的，如果有实际的图片尺寸限制，可以在这里添加检查
    new_x_min = max(new_x_min, 0)
    new_y_min = max(new_y_min, 0)
    
    # 返回新的bounding box
    return (round(new_x_min), round(new_y_min), round(new_x_max), round(new_y_max))

def initialize_model(net_type, sam_dir, net_dir):
    sam = sam_model_registry[net_type](checkpoint=sam_dir)
    sam.cuda()
    net = MaskDecoderHQ(net_type)     
    net.load_state_dict(torch.load(net_dir,map_location="cpu"))
    net.cuda()
    return sam, net

def infer_for_train(input_image, mask, sam, net, device, weight_dtype):
    try:
        box = get_box_from_mask(mask>127)
        box = enlarge_bounding_box(box, 0.1)
        bad_box = False
    except:
        h, w = mask.shape
        box = (0,0,w,h)
        bad_box = True

    mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0).to(device=device)
    box = torch.tensor(box, dtype=torch.float).unsqueeze(0).to(device=device)

    mask_256 = F.interpolate(mask.unsqueeze(0), size=(256, 256), mode='bilinear')
    mask = (mask_256 - 128) / 128
    mask = mask.to(dtype=weight_dtype)

    dict_input = dict()
    dict_input['image'] = input_image 
    dict_input['boxes'] = box
    dict_input['mask_inputs'] = mask
    dict_input['original_size'] = (1024, 1024)
    for key, value in dict_input.items():
        if isinstance(value, torch.Tensor):
            dict_input[key] = value.to(dtype=weight_dtype)
    batched_input = [dict_input]
    batched_output, interm_embeddings = sam(batched_input, multimask_output=False)

    encoder_embedding = batched_output[0]['encoder_embedding']
    image_pe = [batched_output[0]['image_pe']]
    sparse_embeddings = [batched_output[0]['sparse_embeddings']]
    dense_embeddings = [batched_output[0]['dense_embeddings']]

    score_logit = net(
        image_embeddings=encoder_embedding,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        hq_token_only=False,
        interm_embeddings=interm_embeddings,
        mask_ori=mask,
    )

    return score_logit, bad_box