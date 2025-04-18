import os
import argparse

import cv2
import numpy as np
from PIL import Image
import torch
import transformers
from diffusers import FlowMatchEulerDiscreteScheduler

from models.adapter_models import *
from utils.sd3_utils import *
from utils.utils import save_image, post_process
from utils.data_processor import UserInputProcessor


# inference arguments
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of PosterMaker inference.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--controlnet_model_name_or_path2", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--resolution_h", type=int, default=1024)
    parser.add_argument("--resolution_w", type=int, default=1024)

    # number of SD3 ControlNet Layers
    parser.add_argument("--ctrl_layers", type=int, default=23,help="control layers",)
    
    # inference
    parser.add_argument('--num_inference_steps', type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="classifier-free guidance scale")
    parser.add_argument("--erode_mask", action='store_true')
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--use_float16", action='store_true')

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    # parse arguments
    args = parse_args()

    # load text encoders
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        args, text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    ) 
    # Load tokenizers
    tokenizer_one = transformers.CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = transformers.CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = transformers.T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # load vae
    vae = load_vae(args)
    # load sd3
    transformer = load_transfomer(args)
    # load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    # create SceneGenNet
    controlnet_inpaint = load_controlnet(args, transformer, additional_in_channel=1, num_layers=args.ctrl_layers, scratch=True)
    # create TextRenderNet
    controlnet_text = load_controlnet(args, transformer, additional_in_channel=0, scratch=True)
    # load adapter
    adapter = LinearAdapterWithLayerNorm(128, 4096)

    controlnet_inpaint.load_state_dict(torch.load(args.controlnet_model_name_or_path, map_location='cpu'))
    textrender_net_state_dict = torch.load(args.controlnet_model_name_or_path2, map_location='cpu')
    controlnet_text.load_state_dict(textrender_net_state_dict['controlnet_text'])
    adapter.load_state_dict(textrender_net_state_dict['adapter'])

    # set device and dtype
    weight_dtype =  (torch.float16 if args.use_float16 else torch.float32)
    device = torch.device("cuda")

    # move all models to device
    vae.to(device=device)
    text_encoder_one.to(device=device, dtype=weight_dtype)
    text_encoder_two.to(device=device, dtype=weight_dtype)
    text_encoder_three.to(device=device, dtype=weight_dtype)
    controlnet_inpaint.to(device=device, dtype=weight_dtype)
    controlnet_text.to(device=device, dtype=weight_dtype)
    adapter.to(device=device, dtype=weight_dtype)
    
    # load pipeline
    from pipelines.pipeline_sd3 import StableDiffusion3ControlNetPipeline
    pipeline = StableDiffusion3ControlNetPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(
            noise_scheduler.config
            ),
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        text_encoder_3=text_encoder_three,
        tokenizer_3=tokenizer_three,
        controlnet_inpaint=controlnet_inpaint,
        controlnet_text=controlnet_text,
        adapter=adapter,
    )

    pipeline = pipeline.to(dtype=weight_dtype, device=device)

    # user input processor
    data_processor = UserInputProcessor()

    # single user input
    filename = '571507774301'
    image_path = f'./images/rgba_images/{filename}.png'
    mask_path  = f'./images/subject_masks/{filename}.png'
    prompt = """The subject rests on a smooth, dark wooden table, surrounded by a few scattered leaves and delicate flowers,\
                with a serene garden scene complete with blooming flowers and lush greenery in the background."""
    texts = [
            {"content": "护肤美颜贵妇乳", "pos": [69, 104, 681, 185]},
            {"content": "99.9%纯度玻色因", "pos": [165, 226, 585, 272]},
            {"content": "持久保年轻", "pos": [266, 302, 483, 347]}
    ]

    # load image and mask
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # preprocess single user input
    input_data = data_processor(
        image=image,
        mask=mask,
        texts=texts,
        prompt=prompt
    )

    # pipeline input
    cond_image_inpaint = input_data['cond_image_inpaint']
    control_mask = input_data['control_mask']
    prompt = input_data['prompt']
    text_embeds = input_data['text_embeds']
    controlnet_im = input_data['controlnet_im']
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # inference
    results = pipeline(
        prompt=prompt,
        negative_prompt='deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW',
        height=args.resolution_h,
        width=args.resolution_w,
        control_image=[cond_image_inpaint, controlnet_im],  # B, C, H, W
        control_mask=control_mask,  # B,1,H,W
        text_embeds=text_embeds, # B, L, C
        num_inference_steps=28, # number of diffusion steps
        generator=generator,
        controlnet_conditioning_scale=1.0,
        guidance_scale=5.0, # classifier-free guidance scale
        num_images_per_prompt=args.num_images_per_prompt, # number of images to generate for each user input
    ).images # return a list of PIL.Image
    
    # save result
    if len(results) == 1: 
        image = results[0] # num_images_per_prompt == 1
        image = post_process(image, input_data['target_size'])
        output_path = f"./images/results/{filename}.jpg"
        save_image(image, output_path)
    else: 
        for i, image in enumerate(results): # num_images_per_prompt > 1
            image = post_process(image, input_data['target_size'])
            output_path = f"./images/results/{filename}_{i}.jpg"
            save_image(image, output_path)