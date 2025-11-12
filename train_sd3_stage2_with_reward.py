
import logging
import math
import os
from pathlib import Path
import traceback
import copy
from PIL import Image

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from torchvision.transforms import Resize, InterpolationMode
import torch.nn.functional as F

import transformers
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)

from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module

from models.controlnet_sd3 import SD3ControlNetModel
from models.transformer_sd3 import SD3Transformer2DModel
from models.wrapper_models import WrapperModel_SD3_ControlNet_with_Adapter
from models.adapter_models import LinearAdapterWithLayerNorm

from utils.utils import check_and_create_directory
from utils.args_utils import parse_args
from utils.sd3_utils import *
from utils.eval_utils import log_validation_with_pipeline, get_validation_dataset_and_dataloader_e2e


logger = get_logger(__name__)

def load_transfomer(args):
    logger.info(
        f"Loading existing transformer weights from : {args.pretrained_model_name_or_path}"
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
    )
    return transformer

def load_vae(args):
    logger.info(
        f"Loading existing vae weights from : {args.pretrained_model_name_or_path}"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    return vae

def load_controlnet(args, transformer, additional_in_channel=0, pretrained_path=None):
    if pretrained_path:
        logger.info(f"Loading existing controlnet weights from : {args.controlnet_model_name_or_path}")
        controlnet = SD3ControlNetModel.from_pretrained(
            pretrained_path, additional_in_channel=additional_in_channel
        )
    else:
        logger.info("Initializing controlnet weights from transformer")
        controlnet = SD3ControlNetModel.from_transformer(
            transformer, num_layers=args.ctrl_layers, additional_in_channel=additional_in_channel
        )
    return controlnet

def load_text_encoders(args, class_one, class_two, class_three):
    logger.info(
        f"Loading existing text_encoder weights from : {args.pretrained_model_name_or_path}"
    )
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

def forward_one_step(wrapper_model, latent_model_input, t, prompt_embeds, controlnet_pooled_projections,
                     control_image_inpaint, cond_latents_text, text_embeds, transformer, pooled_prompt_embeds, controlnet_inpaint):
    timestep = t.expand(latent_model_input.shape[0])

    control_block_samples_inpaint = controlnet_inpaint(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=controlnet_pooled_projections,
        controlnet_cond=control_image_inpaint,
        return_dict=False,
    )[0]

    # wrapper all learnable parameter
    wrapper_rel = wrapper_model(
        noisy_model_input=latent_model_input,
        timestep=timestep,
        prompt_embeds=prompt_embeds,
        controlnet_pooled_projections=controlnet_pooled_projections,
        controlnet_cond=cond_latents_text,
        text_embeds=text_embeds,
    )
    block_interval = (len(control_block_samples_inpaint) + 1) // len(wrapper_rel)
    control_block_samples = []
    for block_i in range(len(control_block_samples_inpaint)):
        control_block_sample = control_block_samples_inpaint[block_i] + wrapper_rel[block_i // block_interval]
        control_block_samples.append(control_block_sample.to(dtype=transformer.dtype))

    # TODO: 需不需要.sample(参考ImageReward)
    model_pred = transformer(
        hidden_states=latent_model_input.to(dtype=transformer.dtype),
        timestep=timestep,
        encoder_hidden_states=prompt_embeds.to(dtype=transformer.dtype),
        pooled_projections=pooled_prompt_embeds.to(dtype=transformer.dtype),
        block_controlnet_hidden_states=control_block_samples,
        return_dict=False,
    )[0]

    return model_pred

def decode_img_from_latents(vae, latents, save_name=None):
    img = vae.decode((latents.float() / vae.config.scaling_factor) + vae.config.shift_factor, return_dict=False)[0]
    img = (img + 1.0) / 2.0 * 255.0
    if save_name is not None:
        img_np = img[0].permute(1,2,0).detach().cpu().numpy()
        Image.fromarray(img_np.astype(np.uint8)).save(save_name)
    return img

def main(args):
    args.output_dir = os.path.join(args.output_dir, args.name)
    check_and_create_directory(args.output_dir)

    logging_dir = Path(args.output_dir, args.logging_dir)

    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if args.deepspeed:
        from configs.deepspeed_config import get_ds_plugin
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            deepspeed_plugin=get_ds_plugin(args),
            kwargs_handlers=[ddp_kwargs]
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs]
        )

    args.num_processes = accelerator.num_processes
    args.process_index = accelerator.process_index

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler
    logger.info(f"Loading scheduler from : {args.pretrained_model_name_or_path}")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    noise_scheduler_copy2 = copy.deepcopy(noise_scheduler)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32, scheduler=None):
        sigmas = scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
 
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

    # Load text encoder
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

    if args.reward_loss_weight:
        from detection_utils.inference import initialize_model
        from detection_utils.inference import infer_for_train as reward_model_forward
        sam, net = initialize_model(args.sam_net_type, args.sam_dir, args.sam_head_dir)
        sam.requires_grad_(False)
        net.requires_grad_(False)
        sam.eval()
        net.eval()
        sam.to(accelerator.device)
        net.to(accelerator.device)

    # load vae
    vae = load_vae(args)

    # load transformers
    transformer = load_transfomer(args)

    # load controlnet
    controlnet_inpaint = load_controlnet(args, transformer, additional_in_channel=1, pretrained_path=args.controlnet_model_name_or_path)

    # Initialize text controlnet, NOTE: pretrained_path = None
    controlnet_text = load_controlnet(args, transformer, additional_in_channel=0, pretrained_path=None)

    # Load ocr related
    adapter = LinearAdapterWithLayerNorm(128, 4096)

    # load pretrained text_controlnet
    wrapper_model = WrapperModel_SD3_ControlNet_with_Adapter(controlnet_text, adapter)

    # load Stage1 checkpoint
    wrapper_model.load_state_dict(torch.load(args.controlnet_model_name_or_path2))
    controlnet_text = wrapper_model.controlnet
    adapter = wrapper_model.adapter 
    
    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # freeze models
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    adapter.requires_grad_(False)
    controlnet_text.requires_grad_(False) # for frozen text_controlnet
    wrapper_model.requires_grad_(False) 

    controlnet_inpaint.requires_grad_(True) # for bg inpaint learning
    controlnet_inpaint.train() # for bg inpaint learning


    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet_inpaint).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet_inpaint).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        state_dict = torch.load(args.resume_from_checkpoint, map_location='cpu')

        try:
            controlnet_inpaint.load_state_dict(state_dict)
            print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
        except Exception as e:
            print("model weights don't match the model")
            raise e

        step_info = args.resume_from_checkpoint.split('/')[-1].split('_')[0]
        if step_info.startswith('last-'):
            global_step = int(step_info[5:])
        elif step_info.isdigit():
            global_step = int(step_info)
        else:
            global_step = 0
    else:
        global_step = 0


    # Optimizer creation
    params_to_optimize = controlnet_inpaint.parameters()
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    from data_utils.poster_dataset_e2e_train import Poster_Dataset
    print("Imported Poster_Dataset from poster_dataset_e2e_train.")
    train_dataset = Poster_Dataset(args=args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    import functools
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    process_caption_fn = functools.partial(
        compute_text_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        drop_rate=args.p_drop_caption,
        device=accelerator.device,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet_inpaint, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet_inpaint, optimizer, train_dataloader, lr_scheduler
    )
        
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=torch.float32)  # VAE need to be float32
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)
    wrapper_model.to(accelerator.device, dtype=weight_dtype)
    
    # Gradient checkpoint
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        controlnet_inpaint.enable_gradient_checkpointing()
        controlnet_text.enable_gradient_checkpointing()

    # test pipeline, dataset, dataloader, args
    if accelerator.is_main_process:
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
            controlnet_inpaint=unwrap_model(controlnet_inpaint),
            controlnet_text=controlnet_text,
            adapter=adapter,
        )
        print("load pipeline successfully!")

        test_dataloader, _ , test_args = get_validation_dataset_and_dataloader_e2e(args)
        print("load test data successfully!")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # tensorboard cannot handle list types for config
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    resize = Resize(size=(args.resolution_h // 8, args.resolution_w // 8), interpolation = InterpolationMode.NEAREST, antialias=True)
    accelerator.wait_for_everyone()
    controlnet_inpaint.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        ori_loss = 0.0
        reward_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            for train_mode in ['original', 'reward']:
                with accelerator.accumulate(controlnet_inpaint):
                    prompt_embeds, pooled_prompt_embeds = process_caption_fn(
                        batch["caption"]
                    )

                    # our custom input
                    gt_im = batch['gt_im'].to(memory_format=torch.contiguous_format).to(dtype=vae.dtype, device=accelerator.device)
                    controlnet_im = batch['controlnet_im'].to(memory_format=torch.contiguous_format).to(dtype=vae.dtype, device=accelerator.device)
                    subject_mask = batch['subject_mask'].to(memory_format=torch.contiguous_format).to(dtype=vae.dtype, device=accelerator.device)

                    # condition image latents
                    with torch.no_grad():
                        # inpaint condition image
                        subject_mask = (subject_mask + 1.0) / 2.0 # [-1,1]->[0,1], 0 means need inpaint
                        cond_image_inpaint = (gt_im + 1) * subject_mask - 1
                        cond_latents_inpaint = vae.encode(cond_image_inpaint.to(dtype=vae.dtype)).latent_dist.sample()

                        cond_latents_inpaint = (cond_latents_inpaint - vae.config.shift_factor) * vae.config.scaling_factor
                        cond_latents_inpaint = cond_latents_inpaint.to(dtype=weight_dtype)
                        control_image_inpaint = torch.cat(
                            [cond_latents_inpaint, resize(subject_mask.to(dtype=weight_dtype))], dim=1
                        )  # Bx17xHxW

                        # text condition image
                        cond_latents_text = vae.encode(controlnet_im.to(dtype=vae.dtype)).latent_dist.sample()
                        cond_latents_text = (cond_latents_text - vae.config.shift_factor) * vae.config.scaling_factor
                        cond_latents_text = cond_latents_text.to(dtype=weight_dtype) # Bx16xHxW

                        # Convert images to latent space
                        latents = vae.encode(gt_im.to(dtype=vae.dtype)).latent_dist.sample()
                        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                        latents = latents.to(dtype=weight_dtype)

                    # Get the text embedding for conditioning
                    text_embeds = batch['text_embeds'].to(memory_format=torch.contiguous_format, dtype=weight_dtype, device=accelerator.device)

                    # controlnet(s) inpaint inference
                    zero_pooled_prompt_embeds = True
                    if zero_pooled_prompt_embeds:
                        controlnet_pooled_projections = torch.zeros_like(
                            pooled_prompt_embeds
                        )
                    else:
                        controlnet_pooled_projections = pooled_prompt_embeds

                    if not args.reward_loss_weight or train_mode=="original":
                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]

                        # Sample a timestep for each image
                        u = compute_density_for_timestep_sampling(
                            args.weighting_scheme,
                            bsz,
                            args.logit_mean,
                            args.logit_std,
                            args.mode_scale,
                        )

                        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                        timesteps = noise_scheduler_copy.timesteps[indices].to(
                            device=latents.device
                        )

                        # Add noise according to flow matching.
                        sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype, scheduler=noise_scheduler_copy)
                        noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents
                        
                        control_block_samples_inpaint = controlnet_inpaint(
                            hidden_states=noisy_model_input,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=controlnet_pooled_projections,
                            controlnet_cond=control_image_inpaint,
                            return_dict=False,
                        )[0]

                        # wrapper all learnable parameter
                        wrapper_rel = wrapper_model(
                            noisy_model_input=noisy_model_input,
                            timestep=timesteps,
                            prompt_embeds=prompt_embeds,
                            controlnet_pooled_projections=controlnet_pooled_projections,
                            controlnet_cond=cond_latents_text,
                            text_embeds=text_embeds,
                        )

                        block_interval = (len(control_block_samples_inpaint) + 1) // len(wrapper_rel)
                        control_block_samples = []
                        for block_i in range(len(control_block_samples_inpaint)):
                            control_block_sample = control_block_samples_inpaint[block_i] + wrapper_rel[block_i // block_interval]
                            control_block_samples.append(control_block_sample.to(dtype=weight_dtype))

                        model_pred = transformer(
                            hidden_states=noisy_model_input,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            block_controlnet_hidden_states=control_block_samples,
                            return_dict=False,
                        )[0]
                    
                        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                        # Preconditioning of the model outputs.
                        model_pred = model_pred * (-sigmas) + noisy_model_input

                        weighting = compute_loss_weighting_for_sd3(
                            args.weighting_scheme, sigmas
                        )

                        # simplified flow matching aka 0-rectified flow matching loss
                        target = latents

                        # Compute regular loss.
                        loss = torch.mean(
                            (
                                weighting.float() * (model_pred.float() - target.float()) ** 2
                            ).reshape(target.shape[0], -1),
                            1,
                        )
                        loss = loss.mean()
                    
                    # reward loss
                    if args.reward_loss_weight and train_mode=="reward":
                        assert args.train_batch_size == 1, "current implementation only support batch size=1"
                        noise_scheduler_copy2.set_timesteps(args.num_inference_steps, device=accelerator.device)
                    
                        # TODO: 去掉magic number
                        latents = torch.randn((args.train_batch_size, 16, 128, 128), device=accelerator.device, dtype=weight_dtype)
                        
                        timesteps = noise_scheduler_copy2.timesteps

                        mid_timestep = random.randint(args.num_inference_steps-10, args.num_inference_steps-1)
                        with torch.no_grad():
                            for t_i, t in enumerate(timesteps[:mid_timestep]):
                                latent_model_input = latents
                                
                                model_pred = forward_one_step(wrapper_model, latent_model_input, t, prompt_embeds, controlnet_pooled_projections, control_image_inpaint, cond_latents_text, text_embeds, transformer, pooled_prompt_embeds, controlnet_inpaint)

                                latents = noise_scheduler_copy2.step(
                                    model_pred, t, latents, return_dict=False
                                )[0]
                        latent_model_input = latents.detach()
                        model_pred = forward_one_step(wrapper_model, latent_model_input, timesteps[mid_timestep], prompt_embeds, controlnet_pooled_projections, control_image_inpaint, cond_latents_text, text_embeds, transformer, pooled_prompt_embeds, controlnet_inpaint)
                        sigmas = get_sigmas(timesteps[mid_timestep].unsqueeze(0), n_dim=latent_model_input.ndim, dtype=latents.dtype, scheduler=noise_scheduler_copy2)
                        model_pred = model_pred * (-sigmas) + latent_model_input
                        img = decode_img_from_latents(vae, model_pred)
                        input_image = img[0]

                        mask = subject_mask * 255.0
                        mask = mask[0][0].detach().cpu().numpy()
                        score_logit, bad_box = reward_model_forward(input_image, mask, sam, net, accelerator.device, weight_dtype=torch.float32)
                        loss_raw = F.binary_cross_entropy_with_logits(score_logit, torch.zeros_like(score_logit), reduction='none')
                        # loss_mask to threshold out those samples that already have low score_logit
                        loss_mask = (score_logit.sigmoid() > 0.3).float()
                        if bad_box:
                            print("bad_box:", batch['url'])
                            loss_mask *= 0
                        loss = (loss_mask * loss_raw).mean()
                        loss = args.reward_loss_weight * loss

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps
                    if train_mode == "original":
                        ori_loss += avg_loss.item() / args.gradient_accumulation_steps * 2
                    elif train_mode == "reward":
                        reward_loss += avg_loss.item() / args.gradient_accumulation_steps * 2

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = controlnet_inpaint.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                    # print("backward: ", time.time() - stime)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                try:
                    accelerator.log({"loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                except Exception as e:
                    print("log error:", e)
                    traceback.print_exc()
                train_loss = 0.0
                
                # save ckpt & eval
                if accelerator.is_main_process:
                    try:
                        # save checkpoint
                        if global_step % args.checkpointing_steps == 0:
                            save_filename = '%s_net_%s.pth' % (global_step, args.name)
                            save_path = os.path.join(args.output_dir, save_filename)
                            torch.save(unwrap_model(controlnet_inpaint).state_dict(), save_path)

                        # validation
                        if  global_step % args.validation_steps == 0:
                            controlnet_inpaint.eval()
                            log_validation_with_pipeline(
                                logger=logger, pipeline=pipeline, dataloader=test_dataloader,
                                args=test_args, accelerator=accelerator, step=global_step
                            )
                            controlnet_inpaint.train()

                    except Exception as e:
                        print("validation error:", e)
                        traceback.print_exc()
                
                accelerator.wait_for_everyone() # wait for all processes

            # print("next step: ", time.time() - stime)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    os.environ['NCCL_MIN_NCHANNELS'] = '4'
    # 根据GPU型号，设置合适的通信参数
    cuda_device_name = torch.cuda.get_device_name()
    if 'A100' in cuda_device_name or 'A800' in cuda_device_name or 'H800' in cuda_device_name:
        os.environ['NCCL_MIN_NCHANNELS'] = '16'

    args = parse_args()
    main(args)
