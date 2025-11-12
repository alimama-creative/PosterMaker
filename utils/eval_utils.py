import os
import gc
import copy

import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from modelscope.pipelines import pipeline as modelscope_pipline
from modelscope.utils.constant import Tasks

from models.recognizer import TextRecognizer
from utils.utils import (
    check_and_create_directory, 
    pos2coords, 
    full_to_half_width, 
    get_ld, 
    pre_process
)

def process(batch, args, pipeline):
    with torch.no_grad():
        prompt = batch['caption']        
        bg_im = batch['bg_im'].float().cuda() # [-1,1], BCHW
        controlnet_im = batch['controlnet_im'].float().cuda()

        # random seed
        generator = torch.Generator(device='cuda').manual_seed(args.seed) if not args.seed is None else None

        if args.bg_inpaint:
            control_mask = batch['subject_mask'].float().cuda() # [-1,1], BCHW
        else:
            control_mask = batch['mask'].float().cuda() # [-1,1], BCHW

        control_mask = ((control_mask + 1.0) / 2.0) # [-1,1]->[0,1], 0 means need inpaint
        cond_image_inpaint = (bg_im + 1) * control_mask - 1 
        
        text_embeds = batch['text_embeds'].float().cuda()
    
        image = pipeline(
            prompt=prompt,
            negative_prompt='deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW',
            height=args.resolution_h,
            width=args.resolution_w,
            control_image=[cond_image_inpaint, controlnet_im],  # B, C, H, W
            control_mask=control_mask,  # B,1,H,W
            text_embeds=text_embeds, # B, L, C
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=1.0,
            guidance_scale=args.cfg_scale,
        ).images[0]

        # image.save('diffusers_reuslt.jpg')
        results = np.array(image) # h*w*c, [0,255], rgb, uint8
    return  results


def post_process(batch, result):
    gen_im = result

    gt_im = (batch['gt_im'] * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    gt_im = gt_im[0,...] # 1*C*H*W -> C*H*W
    gt_im = np.transpose(gt_im, (1, 2, 0)) # C*H*W -> H*W*C

    batch['gt_im'] = gt_im
    batch['model_out'] = gen_im
        

def log_validation_with_pipeline(logger, pipeline, dataloader, args, accelerator, step):
    logger.info("Running validation... ")
    
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()


    save_path = os.path.join(args.output_dir, 'eval_results')
    check_and_create_directory(save_path)
    check_and_create_directory(os.path.join(save_path, 'gt'))
    check_and_create_directory(os.path.join(save_path, 'gen'))

    # test 
    images = []
    ocr_images = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            with torch.autocast("cuda"):
                result = process(batch, args, pipeline) # h*w*c, [0,255], rgb, uint8
            post_process(batch, result)

            images.append(result)
            for text in batch['texts']:
                content = text['content'][0].replace(" ", "") #去除空格
                text_pos = text['pos']
                text_pos = [p.item() for p in text_pos]
                text_coords = pos2coords(text_pos)
                ocr_images.append({
                    'url':batch['url'][0],
                    'content': content, 
                    'gen_poster_im': batch['model_out'],
                    'text_coords' :text_coords,
                })

            # save image
            filename = batch['url'][0].split('/')[-1][:-4]+ f'_{i}'
            gt_im = batch['gt_im'][..., ::-1]
            cv2.imwrite(os.path.join(save_path, 'gt', f'{filename}.jpg'), gt_im) # rgb -> bgr
            poster_im = batch['model_out'][..., ::-1]
            cv2.imwrite(os.path.join(save_path, 'gen', f'{filename}.jpg'), poster_im) # rgb -> bgr

    # ocr eval
    ocr_args = edict()
    ocr_args.rec_image_shape = "3, 48, 320"
    ocr_args.rec_char_dict_path = os.path.join('./ocr_recog', 'ppocr_keys_v1.txt')
    ocr_args.rec_batch_num = 1
    ocr_args.use_fp16 = False 

    predictor = modelscope_pipline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
    text_recognizer = TextRecognizer(ocr_args, None)

    with torch.no_grad():
        preds_all = []
        sen_acc = []
        edit_dist = []
        for i, batch in tqdm(enumerate(ocr_images)):
            text_coords = batch['text_coords']
            gt_text = batch['content']

            gt_text = gt_text.replace(" ", "") #去除空格
            gt_text = full_to_half_width(gt_text)

            img = batch['gen_poster_im'] # h*w*c, [0,255], rgb, uint8

            pred_img = img[text_coords[1]:text_coords[1]+text_coords[3],
                        text_coords[0]:text_coords[0]+text_coords[2], ::-1] # bgr

            pred_img = torch.from_numpy(pred_img.copy())
            pred_img = pred_img.permute(2, 0, 1).float()  # HWC-->CHW

            pred_img = pre_process([pred_img], ocr_args.rec_image_shape)[0]

            rst = predictor(pred_img)
            pred_text = rst['text'][0]
            pred_text = full_to_half_width(pred_text)
            pred_text = pred_text.replace(" ", "")

            preds_all += [pred_text]
            
            gt_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars)-1) for m in gt_text]
            pred_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars)-1) for m in pred_text]
        
            if pred_text == gt_text:
                sen_acc += [1]
            else:
                sen_acc += [0]
            edit_dist += [get_ld(pred_order, gt_order)]

            ocr_images[i]['pred_text'] = pred_text
            ocr_images[i]['gt_text'] = gt_text
            ocr_images[i]['sen_acc'] = sen_acc[-1]
            ocr_images[i]['edit_dist'] = edit_dist[-1]

    logger.info({'global_step':step, "sen_acc": np.array(sen_acc).mean(), "edit_dist": np.array(edit_dist).mean()})

    del predictor
    del text_recognizer
    gc.collect()
    torch.cuda.empty_cache()


def get_validation_dataset_and_dataloader_e2e(args):
    from data_utils.poster_dataset_e2e_eval import Poster_Dataset
    test_args = copy.deepcopy(args)
    dataset = Poster_Dataset(args=test_args)
    dataloader = DataLoader(dataset, 
                            num_workers=0, 
                            batch_size=1, 
                            shuffle=False)
    
    return dataloader, dataset, test_args