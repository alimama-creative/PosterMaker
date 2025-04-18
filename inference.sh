python inference.py \
 --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers/' \
 --controlnet_model_name_or_path='./checkpoints/ours_weights/scenegen_net-0415.pth' \
 --controlnet_model_name_or_path2='./checkpoints/ours_weights/textrender_net-0415.pth' \
 --seed=42 \
 --num_images_per_prompt=4 \
 --use_float16


python inference.py \
 --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers/' \
 --controlnet_model_name_or_path='./checkpoints/ours_weights/scenegen_net-rl-0415.pth' \
 --controlnet_model_name_or_path2='./checkpoints/ours_weights/textrender_net-0415.pth' \
 --seed=42 \
 --num_images_per_prompt=4 \
 --use_float16


python inference.py \
 --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers/' \
 --controlnet_model_name_or_path='./checkpoints/ours_weights/scenegen_net-1m-0415.pth' \
 --controlnet_model_name_or_path2='./checkpoints/ours_weights/textrender_net-1m-0415.pth' \
 --seed=42 \
 --num_images_per_prompt=4 \
 --use_float16