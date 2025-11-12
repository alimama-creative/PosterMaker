### Stage1, Recommend using 32 A100 cluster training takes approximately 1.5 days, but you can also use 8 A100 with more time

# Basic Configuration
BATCH_SIZE=4
EPOCHS=50
GAC=1
LR=1e-4

# Training Parameters
TRAIN_ARGS="--output_dir=/tmp \
--name=postermaker_debug_stage1 \
--mixed_precision=fp16 \
--learning_rate=${LR} \
--num_train_epochs=${EPOCHS} \
--train_batch_size=${BATCH_SIZE} \
--gradient_accumulation_steps=${GAC} \
--lr_warmup_steps=500 \
--lr_scheduler=constant_with_warmup \
--adam_epsilon=1e-15 \
--dataloader_num_workers=10 \
--checkpointing_steps=10 \
--validation_steps=10 \
--resolution_h=1024 \
--resolution_w=1024 \
--pretrained_model_name_or_path=./checkpoints/stable-diffusion-3-medium-diffusers/ \
--ctrl_layers=12 \
--controlnet_model_name_or_path=./checkpoints/SD3-Controlnet-Inpainting \
--max_num_texts=7 \
--char_padding_to_len=16 \
--text_feature_drop=0.1 \
--p_drop_caption=0 \
--cfg_scale=5.0"

#Start Training
HF_HUB_OFFLINE=1 WORLD_SIZE=1 RANK=-1 python3 train_sd3_stage1.py $TRAIN_ARGS \


### Stage2, Recommend using 32 A100 cluster training takes approximately 1.5 days, but you can also use 8 A100 with more time

# Basic Configuration
BATCH_SIZE=2
EPOCHS=50
GAC=1
LR=1e-4

#Training Parameters
TRAIN_ARGS="--output_dir=/tmp \
--name=postermaker_debug_stage2_wo_reward \
--mixed_precision=fp16 \
--learning_rate=${LR} \
--num_train_epochs=${EPOCHS} \
--train_batch_size=${BATCH_SIZE} \
--gradient_accumulation_steps=${GAC} \
--lr_warmup_steps=500 \
--lr_scheduler=constant_with_warmup \
--adam_epsilon=1e-15 \
--dataloader_num_workers=10 \
--checkpointing_steps=20 \
--validation_steps=1000 \
--resolution_h=1024 \
--resolution_w=1024 \
--pretrained_model_name_or_path=./checkpoints/stable-diffusion-3-medium-diffusers/ \
--ctrl_layers=12 \
--controlnet_model_name_or_path=./checkpoints/SD3-Controlnet-Inpainting \
--controlnet_model_name_or_path2=/tmp/postermaker_debug_stage1/0_net_postermaker_debug_stage1.pth \
--max_num_texts=7 \
--char_padding_to_len=16 \
--text_feature_drop=0 \
--cfg_scale=5.0 \
--bg_inpaint" # NOTE: add this to enable stage2 training

# #Start Training
HF_HUB_OFFLINE=1 WORLD_SIZE=1 RANK=-1 python3 train_sd3_stage2.py $TRAIN_ARGS


# Stage2 with reward, must use deepspeed with >= 8 GPUs, single GPU will OOM, recommend using 32 A100 cluster.
## Basic Configuration
BATCH_SIZE=1
EPOCHS=50
GAC=1
LR=1e-4

#Training Parameters
TRAIN_ARGS="--output_dir=/tmp \
--name=postermaker_debug_stage2_w_reward \
--mixed_precision=fp16 \
--learning_rate=${LR} \
--num_train_epochs=${EPOCHS} \
--train_batch_size=${BATCH_SIZE} \
--gradient_accumulation_steps=${GAC} \
--lr_warmup_steps=500 \
--lr_scheduler=constant_with_warmup \
--adam_epsilon=1e-15 \
--dataloader_num_workers=10 \
--checkpointing_steps=20 \
--validation_steps=1000 \
--resolution_h=1024 \
--resolution_w=1024 \
--pretrained_model_name_or_path=./checkpoints/stable-diffusion-3-medium-diffusers/ \
--ctrl_layers=12 \
--controlnet_model_name_or_path=./checkpoints/SD3-Controlnet-Inpainting \
--controlnet_model_name_or_path2=/tmp/postermaker_debug_stage2_wo_reward/0_net_postermaker_debug_stage2_wo_reward.pth \
--max_num_texts=7 \
--char_padding_to_len=16 \
--text_feature_drop=0 \
--cfg_scale=5.0 \
--bg_inpaint \
--sam_net_type=vit_h \
--sam_dir=./checkpoints/outgrowth_det/sam_vit_h_4b8939.pth \
--sam_head_dir=./checkpoints/outgrowth_det/vit_h_epoch10.pth \
--gradient_checkpointing \
--deepspeed"


HF_HUB_OFFLINE=1 WORLD_SIZE=1 RANK=-1 python3 train_sd3_stage2_with_reward.py $TRAIN_ARGS
