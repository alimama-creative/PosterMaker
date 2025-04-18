import gradio as gr
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

import cv2
import torch
import transformers
from diffusers import FlowMatchEulerDiscreteScheduler
import textwrap

from models.adapter_models import LinearAdapterWithLayerNorm
from utils.data_processor import UserInputProcessor
from utils.sd3_utils import *
from utils.utils import post_process
from utils.data_processor import UserInputProcessor
from pipelines.pipeline_sd3 import StableDiffusion3ControlNetPipeline

def check_and_process_texts(texts_str, width, height):
    try:
        if not texts_str:
            raise ValueError("texts_str cannot be None or empty")
            
        texts = json.loads(texts_str)
        
        if not texts or not isinstance(texts, list):
            raise ValueError("Invalid texts format: must be a non-empty list")
            
        if len(texts) > 7:
            raise ValueError("Too many text lines: maximum allowed is 7 lines")
            
        processed_texts = []
        
        for text in texts:
            if not isinstance(text, dict) or 'content' not in text or 'pos' not in text:
                raise ValueError("Invalid text format: each item must be a dict with 'content' and 'pos'")
                
            content = text['content']
            pos = text['pos']
            
            # 检查文本长度
            if len(content) > 16:
                raise ValueError(f"Text too long: '{content}' exceeds 16 characters")
                
            # 检查并修正边界值
            x1, y1, x2, y2 = pos
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # 确保 x1 < x2, y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            processed_texts.append({
                "content": content,
                "pos": [x1, y1, x2, y2]
            })
            
        return processed_texts
        
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in texts_str")
    except Exception as e:
        raise ValueError(f"Error processing texts: {str(e)}")


class ImageGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor = UserInputProcessor()
        
        # 初始化模型和管道
        self.initialize_models()
        
    def initialize_models(self):
        # 这里加载所有必要的模型和组件
        args = self.get_default_args()
        
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
        controlnet_inpaint = load_controlnet(args, transformer, additional_in_channel=1, num_layers=23, scratch=True)
        # create TextRenderNet
        controlnet_text = load_controlnet(args, transformer, additional_in_channel=0, scratch=True)      
        # load adapter
        adapter = LinearAdapterWithLayerNorm(128, 4096)
        
        controlnet_inpaint.load_state_dict(torch.load(args.controlnet_model_name_or_path, map_location='cpu'))
        textrender_net_state_dict = torch.load(args.controlnet_model_name_or_path2, map_location='cpu')
        controlnet_text.load_state_dict(textrender_net_state_dict['controlnet_text'])
        adapter.load_state_dict(textrender_net_state_dict['adapter'])

        # set device and dtype
        weight_dtype = (torch.float16 if torch.cuda.is_available() else torch.float32)
        device = self.device

        # move all models to device
        vae.to(device=device)
        text_encoder_one.to(device=device, dtype=weight_dtype)
        text_encoder_two.to(device=device, dtype=weight_dtype)
        text_encoder_three.to(device=device, dtype=weight_dtype)
        controlnet_inpaint.to(device=device, dtype=weight_dtype)
        controlnet_text.to(device=device, dtype=weight_dtype)
        adapter.to(device=device, dtype=weight_dtype)

        # load pipeline
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

        self.pipeline = pipeline.to(dtype=weight_dtype, device=device)
        
    def generate(self, main_image, mask_image, texts_str, prompt, seed_generator):
        try:
            # 将输入图像转换为numpy格式，RGB
            main_image = np.array(main_image)
            mask = cv2.cvtColor(np.array(mask_image), cv2.COLOR_BGR2GRAY)
            
            # 解析文本布局
            texts = json.loads(texts_str)
            
            # 预处理输入数据
            input_data = self.data_processor(
                image=main_image,
                mask=mask,
                texts=texts,
                prompt=prompt
            )
            
            # 执行推理
            results = self.pipeline(
                prompt=prompt,
                negative_prompt='deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW',
                height=1024,
                width=1024,
                control_image=[input_data['cond_image_inpaint'], input_data['controlnet_im']],
                control_mask=input_data['control_mask'],
                text_embeds=input_data['text_embeds'],
                num_inference_steps=(28 if torch.cuda.is_available() else 1),
                generator=seed_generator,
                controlnet_conditioning_scale=1.0,
                guidance_scale=5.0,
                num_images_per_prompt=1,
            ).images
            
            # 后处理，根据im_h, im_w从rel中裁剪[0, 0, im_w, im_h]区域
            rel = post_process(results[0], input_data['target_size'])
            
            # 返回生成的图像
            return rel
            
        except Exception as e:
            return f"Error in image generation: {str(e)}"
        
    def get_default_args(self):
        # 返回默认参数配置
        class Args:
            def __init__(self):
                self.pretrained_model_name_or_path = './checkpoints/stable-diffusion-3-medium-diffusers/'
                self.controlnet_model_name_or_path='./checkpoints/ours_weights/scenegen_net-1m-0415.pth'
                self.controlnet_model_name_or_path2='./checkpoints/ours_weights/textrender_net-1m-0415.pth'
                self.revision = None
        return Args()


def visualize_layout(main_image, mask_image, texts_str, prompt,
                     font_path: str = "./assets/fonts/AlibabaPuHuiTi-3-55-Regular.ttf",
                     margin_ratio: float = 0.92):
    """
    渲染带有文字 bbox 的布局示意图  
    - texts_str -> [{'pos': (x1,y1,x2,y2), 'content': "..."}] 由外部 `check_and_process_texts` 解析
    - 文本大小、位置会根据框大小自动调整并保持居中
    """
    try:
        # -------- 1. 获取画布尺寸 -------- #
        if main_image is not None:
            height, width = main_image.shape[:2]
        elif mask_image is not None:
            height, width = mask_image.shape[:2]
        else:
            width = height = 1024

        # -------- 2. 底图与 mask 合成 -------- #
        canvas = Image.new("RGBA", (width, height), "white")

        if main_image is not None:
            pil_main = Image.fromarray(main_image).convert("RGBA")
        else:
            pil_main = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        if mask_image is not None:
            gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            alpha = Image.fromarray(np.where(gray > 127, 255, 0).astype(np.uint8))
            pil_main.putalpha(alpha)

        canvas.alpha_composite(pil_main)

        # -------- 3. 工具函数 -------- #
        def get_font(size):
            """优雅地降级到默认字体"""
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                return ImageFont.load_default()

        def optimal_font_size(text, box_w, box_h, draw):
            """二分搜索可容纳最大字号（无需传 font_path，因为 get_font 自带回退）"""
            low, high, best = 1, box_h, 1
            while low <= high:
                mid = (low + high) // 2
                font = get_font(mid)
                bbox = draw.textbbox((0, 0), text, font=font)
                txt_w = bbox[2] - bbox[0]
                txt_h = bbox[3] - bbox[1]
                if txt_w <= box_w * margin_ratio and txt_h <= box_h * margin_ratio:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            return best

        def wrap_text(text, font, box_w, draw):
            """根据宽度自动换行；返回 lines 列表"""
            words = text.split()  # 兼容英文空格，中文空格同理
            if len(words) == 1:
                # 纯中文或没有空格时：按字符粗暴截断
                wrapped = textwrap.wrap(text, width=len(text))
            else:
                wrapped = textwrap.wrap(text, width=len(words))
            # 尝试不断扩行，直到所有行宽都 <= box_w*margin_ratio
            while True:
                line_too_long = False
                for i, line in enumerate(wrapped):
                    if draw.textlength(line, font=font) > box_w * margin_ratio:
                        # 把该行再拆一半
                        midpoint = max(1, len(line) // 2)
                        wrapped[i:i+1] = [line[:midpoint], line[midpoint:]]
                        line_too_long = True
                        break
                if not line_too_long:
                    break
            return wrapped

        draw = ImageDraw.Draw(canvas)

        # -------- 4. 渲染每个文本框 -------- #
        texts = check_and_process_texts(texts_str, width, height)

        for item in texts:
            (x1, y1, x2, y2) = item["pos"]
            content = item["content"]

            box_w, box_h = x2 - x1, y2 - y1

            # 4.1 计算最佳字号
            size = optimal_font_size(content, box_w, box_h, draw)
            font = get_font(size)

            # 4.2 如果单行仍超宽则自动换行并调整字号
            lines = wrap_text(content, font, box_w, draw)
            # 若换行后总高度超框，再缩小字号
            line_height = size * 1.2
            while line_height * len(lines) > box_h * margin_ratio and size > 1:
                size -= 1
                font = get_font(size)
                lines = wrap_text(content, font, box_w, draw)
                line_height = size * 1.2

            # 4.3 计算整体文本块尺寸
            txt_h = line_height * len(lines)
            txt_w = max(draw.textlength(line, font=font) for line in lines)

            # 左上角坐标（居中）
            start_x = round(x1 + (box_w - txt_w) / 2)
            start_y = round(y1 + (box_h - txt_h) / 2)

            # 4.4 绘制 bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # 4.5 绘制文字
            for idx, line in enumerate(lines):
                draw.text((start_x, start_y + idx * line_height),
                          line, fill="blue", font=font, align="center")

        return canvas.convert("RGB")  # 与旧接口保持一致

    except Exception as e:
        # 返回 Image 以外的对象可能会破坏调用链，直接 raise 更好
        return RuntimeError(f"visualize_layout failed: {e}")


# 修改generate_image函数来使用ImageGenerator
generator = ImageGenerator()

def generate_image(main_image, mask_image, texts_str, prompt, seed):
    if main_image is None:
        return "Error: Main image is required"

    try:
        # 处理main_image的格式
        if isinstance(main_image, np.ndarray):
            # 处理numpy array格式
            if main_image.ndim == 3:
                if main_image.shape[2] == 4:  # RGBA格式
                    rgb_array = main_image[..., :3]
                    alpha_channel = main_image[..., 3]
                    main_image = Image.fromarray(rgb_array)
                    # 如果没有提供mask，使用alpha通道作为mask
                    if mask_image is None:
                        mask_array = (alpha_channel > 128).astype(np.uint8) * 255
                        mask_image = Image.fromarray(mask_array)
                elif main_image.shape[2] == 3:  # RGB格式
                    main_image = Image.fromarray(main_image)
                    if mask_image is None:
                        return "Error: When using RGB image, a mask image must be provided"
                else:
                    return "Error: Invalid number of channels in main image"
            else:
                return "Error: Invalid dimensions for main image"
        elif isinstance(main_image, Image.Image):
            # 处理PIL.Image格式
            if main_image.mode == 'RGBA':
                rgb_image = main_image.convert('RGB')
                alpha_channel = main_image.split()[3]
                # 如果没有提供mask，使用alpha通道作为mask
                if mask_image is None:
                    mask_image = alpha_channel.point(lambda x: 255 if x > 128 else 0)
                main_image = rgb_image
            elif main_image.mode != 'RGB' and mask_image is None:
                return "Error: When using RGB image, a mask image must be provided"
        else:
            return "Error: Main image must be numpy array or PIL.Image format"

        # 确保main_image是RGB模式
        if isinstance(main_image, Image.Image) and main_image.mode != 'RGB':
            main_image = main_image.convert('RGB')

        # 处理mask_image的格式
        if mask_image is not None:
            if isinstance(mask_image, np.ndarray):
                mask_image = Image.fromarray(mask_image)
            elif not isinstance(mask_image, Image.Image):
                return "Error: Mask image must be numpy array or PIL.Image format"

        # 使用设定的seed
        seed_generator = torch.Generator(device=generator.device).manual_seed(int(seed))
        # 使用ImageGenerator生成图像
        generated_image = generator.generate(main_image, mask_image, texts_str, prompt, seed_generator)
        return generated_image
    except Exception as e:
        return f"Error: {str(e)}"

# For debugging
# def generate_image(main_image, mask_image, texts_str, prompt, seed):
#     try:
#         # 这里是生成图像的逻辑
#         # 现在只返回一个占位图像
#         if main_image is None:
#             return "Error: Main image is required"
            
#         generated_image = Image.fromarray(main_image)  # 临时使用输入图像作为输出
#         return generated_image
    
#     except Exception as e:
#         return f"Error: {str(e)}"


# 清除所有输入和输出的函数
def clear_all():
    return [None, None, "", "", 42, None, None]

# 修改Gradio界面部分
with gr.Blocks() as iface:
    gr.Markdown("""
    # 🎨 [CVPR2025] PosterMaker: Towards High-Quality Product Poster Generation with Accurate Text Rendering

    ## 文字海报图像生成 | A text poster image generation
                
    ## **作者 | Authors:** Yifan Gao\*, Zihang Lin\*, Chuanbin Liu, Min Zhou, Tiezheng Ge, Bo Zheng, Hongtao Xie
                                
    <div style="display: flex; gap: 10px; justify-content: left;">
        <a href="https://github.com/eafn/PosterMaker"><img src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github" alt="GitHub"></a>
        <a href="https://arxiv.org/abs/2504.06632"><img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv" alt="Paper"></a>
    </div>    
    """)
    gr.Markdown("""
        ---
        ## 📝 文本布局格式 | Text Layout Format
        """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 文本JSON格式要求 | Text JSON Format Requirements:
            ```json
            [
                {"content": "第一行文本", "pos": [x1, y1, x2, y2]},
                {"content": "第二行文本", "pos": [x1, y1, x2, y2]}
            ]
            ```
            """)
        
        with gr.Column():
            gr.Markdown("""
            ### 文本限制 | Text Limitations:
            - 最多7行文本 | Maximum 7 lines of text
            - 每行≤16个字符 | ≤16 characters per line
            - 坐标不超过图像边界 | Coordinates within image boundaries
            """)

    # 第一排：文本输入框和seed设置
    with gr.Row():
        texts_input = gr.Textbox(
            label="Input JSON text layout", 
            lines=6,
            placeholder="Enter the layout JSON here...",
            scale=1,
        )
        prompt_input = gr.Textbox(
            label="Prompt", 
            lines=6,
            placeholder="Enter the generation prompt here...",
            scale=1,
        )
        seed_input = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=10000,
            step=1,  # 步长为1，确保是整数
            value=42,
            scale=0.3,
        )
    
    gr.Markdown("""
        ---
        ## 📷 图像上传规则 | Image Upload Rules:
        """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 主图像(必需) | Subject Image (Required):
            - 支持RGB格式 | Supports RGB format

            ### 蒙版图像(必需) | Mask Image (Required):
            - RGB图像必须上传额外的蒙版图像 | RGB image must have a separate mask image
            """)
        
        with gr.Column():
            gr.Markdown("""
            ### 蒙版规则 | Mask Rules:
            - 白色区域：保留的部分 | White areas: areas to keep
            - 黑色区域：生成的部分 | Black areas: areas to generate
            """)

    # 第二排：图像输入
    with gr.Row():
        with gr.Column(scale=1):
            main_image_input = gr.Image(
                label="Upload Subject Image", 
                height=400,
            )
        with gr.Column(scale=1):
            mask_image_input = gr.Image(
                label="Upload Mask Image", 
                height=400,
            )
    
    # 提醒信息
    gr.Markdown("""
        ---
        ## ⚠️ 重要提示 | Important Notes:
        """)
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 预览步骤 | Preview Steps:
            - 请先使用"Visualize Layout"按钮预览文本布局 | Please use "Visualize Layout" button first to preview text layout
            - 确认布局无误后再点击"Generate Image"生成图像 | Click "Generate Image" after confirming the layout is correct
            """)
        
        with gr.Column():
            gr.Markdown("""
            ### 等待说明 | Wait Time:
            - 图像生成可能需要较长时间，请耐心等待 | Image generation may take some time, please be patient
            """)
        
    # 第三排：按钮
    with gr.Row():
        visualize_btn = gr.Button("Visualize Layout")
        generate_btn = gr.Button("Generate Image")
        clear_btn = gr.Button("Clear All")
    
    # 第四排：输出图像
    with gr.Row():
        with gr.Column(scale=1):
            layout_output = gr.Image(
                label="Layout Visualization", 
                height=400,
            )
        with gr.Column(scale=1):
            generated_output = gr.Image(
                label="Generated Image", 
                height=400,
            )

    gr.Markdown("""
        ---
        ## 示例 | Examples:
        """)
    # 设置示例
    examples = [
        [
            json.dumps([
                {"content": "护肤美颜贵妇乳", "pos": [69, 104, 681, 185]},
                {"content": "99.9%纯度玻色因", "pos": [165, 226, 585, 272]},
                {"content": "持久保年轻", "pos": [266, 302, 483, 347]}
            ], ensure_ascii=False),
            "The subject rests on a smooth, dark wooden table, surrounded by a few scattered leaves and delicate flowers,with a serene garden scene complete with blooming flowers and lush greenery in the background.",
            './images/rgba_images/571507774301.png',
            './images/subject_masks/571507774301.png',
            42
        ],
        [
            json.dumps([
                {"content": "增强免疫力", "pos": [38, 38, 471, 127]},
                {"content": "幼儿奶粉", "pos": [38, 143, 356, 224]},
                {"content": "易于冲调", "pos": [67, 259, 219, 296]}
            ], ensure_ascii=False),
            "The golden can of milk powder rests on a smooth, dark wooden table, surrounded by a few scattered leaves and delicate flowers, with a serene garden scene complete with blooming flowers and lush greenery in the background.",
            './images/rgba_images/652158680541.png',
            './images/subject_masks/652158680541.png',
            42
        ],
        [
            json.dumps([
                {"content": "CAB恒久气垫", "pos": [85, 101, 720, 192]},
                {"content": "持久不脱妆", "pos": [294, 226, 511, 271]}
            ], ensure_ascii=False),
            "A subject sits elegantly on smooth, light beige fabric, surrounded by a backdrop of similarly draped material that offers a silky appearance. To the left, a delicate white flower injects a subtle natural element into the composition. The overall environment is clean, bright, and minimalistic, exuding a sense of sophistication and simplicity that highlights the subject beautifully.",
            './images/rgba_images/809702153676.png',
            './images/subject_masks/809702153676.png',
            888
        ],
            [
        json.dumps([
            {"content": "原创新款", "pos": [135, 60, 686, 199]},
            {"content": "卡通款手机壳", "pos": [246, 236, 575, 299]}
        ], ensure_ascii=False),
        "The poster features a vibrant yellow background adorned with playful cartoons, including rainbows, clouds, and stars. Characters perform activities like carrying bags and holding hearts, adding a dynamic feel. Comic-style text amplifies the cheerful vibe. The solid yellow backdrop ensures the product stands out. The poster uses eye-catching fonts and text, offering clear visuals that blend harmonious design with sophistication.",
        './images/rgba_images/749870344644.png',
        './images/subject_masks/749870344644.png',
        1000
        ]
    ]
    gr.Examples(
        examples=examples,
        inputs=[texts_input, prompt_input, main_image_input, mask_image_input, seed_input]
    )

    # 设置按钮事件
    visualize_btn.click(
        fn=visualize_layout,
        inputs=[main_image_input, mask_image_input, texts_input, prompt_input],
        outputs=layout_output
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[main_image_input, mask_image_input, texts_input, prompt_input, seed_input],
        outputs=generated_output
    )
    
    # 清除按钮事件
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[main_image_input, mask_image_input, texts_input, prompt_input, 
                seed_input, layout_output, generated_output]
    )


# 启动应用
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0",server_port=7861)

