import torch
import os
import logging
import time
import math
import random 
import numpy as np

from diffusers import AutoencoderKL
from transformers import PretrainedConfig
from models.controlnet_sd3 import SD3ControlNetModel
from models.transformer_sd3 import SD3Transformer2DModel


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

##################################
##### Text Encoder
##################################


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    drop_rate=0,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    # random drop prompt
    if drop_rate > 0:
        prompt = [x if random.random() > drop_rate else "" for x in prompt]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
    drop_rate=0,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    # random drop prompt
    if drop_rate > 0:
        prompt = [x if random.random() > drop_rate else "" for x in prompt]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
    drop_rate=0,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    ## Encode with CLIP
    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            drop_rate=drop_rate,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    ## Encode with T5
    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
        drop_rate=drop_rate,
    )

    ## Concat embedding
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def compute_text_embeddings(prompt, text_encoders, tokenizers, drop_rate, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, drop_rate=drop_rate
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu"
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def prompt_embedding_drop(prompt_embed, pooled_prompt_embed, neg_prompt_embed, neg_pooled_prompt_embed, drop_rate=0):
    pooled_prompt_1_embed = pooled_prompt_embed[:768] # 768
    pooled_prompt_2_embed = pooled_prompt_embed[768:] # 1280

    neg_pooled_prompt_1_embed = neg_pooled_prompt_embed[:768] # 768
    neg_pooled_prompt_2_embed = neg_pooled_prompt_embed[768:] # 1280

    prompt_1_embed = prompt_embed[:77, :768] # 77*768
    prompt_2_embed = prompt_embed[:77, 768:2048] # 77*1280
    t5_prompt_embed = prompt_embed[77:, :] # 77*4096

    neg_prompt_1_embed = neg_prompt_embed[:77, :768] # 77*768
    neg_prompt_2_embed = neg_prompt_embed[:77, 768:2048] # 77*1280
    neg_t5_prompt_embed = neg_prompt_embed[77:, :] # 77*4096

    # prompt 1
    if random.random() < drop_rate:
        prompt_1_embed = neg_prompt_1_embed
        pooled_prompt_1_embed = neg_pooled_prompt_1_embed

    # prompt 2
    if random.random() < drop_rate:
        prompt_2_embed = neg_prompt_2_embed
        pooled_prompt_2_embed = neg_pooled_prompt_2_embed

    # t5 prompt
    if random.random() < drop_rate:
        t5_prompt_embed = neg_t5_prompt_embed

    pooled_prompt_embeds = np.concatenate((pooled_prompt_1_embed, pooled_prompt_2_embed), axis=-1) # 2048
    clip_prompt_embeds = np.concatenate((prompt_1_embed, prompt_2_embed), axis=-1) # 77*2048
    clip_prompt_embeds = np.pad(
        clip_prompt_embeds, 
        ((0,0), (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])),
        mode='constant',
        constant_values=0
    )
    prompt_embeds = np.concatenate([clip_prompt_embeds, t5_prompt_embed], axis=-2)

    return prompt_embeds, pooled_prompt_embeds


def load_text_encoders(args, class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_3",
        revision=args.revision,
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def load_vae(args):
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    return vae


def load_transfomer(args):
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
    )
    return transformer


def load_controlnet(args, transformer, additional_in_channel=0, num_layers=12, scratch=False):
    if args.controlnet_model_name_or_path and not scratch:
        controlnet = SD3ControlNetModel.from_pretrained(
            args.controlnet_model_name_or_path, additional_in_channel=additional_in_channel
        )
    else:
        controlnet = SD3ControlNetModel.from_transformer(
            transformer, num_layers=num_layers, additional_in_channel=additional_in_channel
        )
    return controlnet