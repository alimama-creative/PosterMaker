import os
import json
import copy

import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from models.text_embedder import FourierEmbedder
from utils.utils import *

# stage2 eval
STAGE2_GT_IM_SAVE_PATH = './dataset/cvpr25_release_benchmark_stage2/gt/'
STAGE2_SUBJECT_MASK_SAVE_PATH = './dataset/cvpr25_release_benchmark_stage2/mask/'
STAGE2_DATA_SAMPLES_PATH = './dataset/cvpr25_release_benchmark_stage2/text_render_benchmark.json'

# stage1 eval
STAGE1_GT_IM_SAVE_PATH = './dataset/cvpr25_release_benchmark_stage1/gt/'
STAGE1_SUBJECT_MASK_SAVE_PATH = './dataset/cvpr25_release_benchmark_stage1/mask/'
STAGE1_DATA_SAMPLES_PATH = './dataset/cvpr25_release_benchmark_stage1/text_render_benchmark.json'

class Poster_Dataset(data.Dataset):
    def __init__(self, args, **kwargs):
        super(Poster_Dataset, self).__init__()
        self.bg_inpaint = getattr(args, 'bg_inpaint', None)

        if self.bg_inpaint:
            self.data_samples_path = STAGE2_DATA_SAMPLES_PATH
            self.gt_im_path = STAGE2_GT_IM_SAVE_PATH
            self.mask_im_path = STAGE2_SUBJECT_MASK_SAVE_PATH
        else:
            self.data_samples_path = STAGE1_DATA_SAMPLES_PATH
            self.gt_im_path = STAGE1_GT_IM_SAVE_PATH
            self.mask_im_path = STAGE1_SUBJECT_MASK_SAVE_PATH

        with open(self.data_samples_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        '''
        self.samples: List of dict
        example: 
            {
                'url': 'xxxxxx.png',
                'caption': 'xxxxxx',
                'texts': [
                    {'content': 'xxxxxx', 'pos': [xx, xx, xx, xx]},
                    {'content': 'xxxxxx', 'pos': [xx, xx, xx, xx]},
                    ...
                ],
                'logo': [
                    [xx, xx, xx, xx],
                    [xx, xx, xx, xx],
                    ...
                ],
                'texts_out': [
                    {'content': 'xxxxxx', 'pos': [xx, xx, xx, xx]},
                    {'content': 'xxxxxx', 'pos': [xx, xx, xx, xx]},
                    ...
                ],
            }
        '''
    
        self.len = len(self.samples)
        print(f"total {self.len} test samples")


        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

        self.backup_item = None

        # image size
        self.input_size = (args.resolution_h, args.resolution_w)

        # prompt  text
        self.prompt_fix = getattr(args, "prompt", "")

        self.erode_mask = getattr(args, 'erode_mask', None)

        self.max_num_texts = getattr(args, 'max_num_texts', 7)

        self.char_padding_to_len = getattr(args, 'char_padding_to_len', None)

        self.text_feature_drop = getattr(args, "text_feature_drop", None)
        
        self.char_pos_encoding_dim = getattr(args, 'char_pos_encoding_dim', 32)

        self.text_pos_encoding_dim = getattr(args, 'text_pos_encoding_dim', 32)

        self.char2feat = torch.load('./assets/char2feat_ppocr_neck64_avg.pth')
        print("dataset load ppocr64 char2feat successfully!")
        
        self.fourier_embedder = FourierEmbedder(num_freqs=self.text_pos_encoding_dim // (4*2)) # 4 is xywh, 2 is cos&sin

        self.debug = args.debug


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.debug:
            try:
                item = self.__load_item(idx)
                return item
            except Exception as e:
                sample = self.samples[idx]
                url = sample['url']
                texts = sample['texts']
                print(idx, url, texts)
                raise e

        try:
            item = self.__load_item(idx)
            self.backup_item = item
        except (KeyboardInterrupt, SystemExit):
            raise
        except (Exception) as e:
            """Handling errors introduced by random mask generation step introduced in dataloader."""
            print('loading error: item ' + self.samples[idx]['url'])
            sample = self.samples[idx]
            url = sample['url']
            texts = sample['texts']
            print(idx, url, texts)
            if self.backup_item is not None:
                item = self.backup_item
            else:
                cur_idx = idx
                while True:
                    cur_idx = (cur_idx + 1) % self.len
                    try:
                        item = self.__load_item(cur_idx)
                        self.backup_item = item
                    except (Exception) as e: 
                        print('loading error: item ' + self.samples[cur_idx]['url'])
                    if item:
                        return item
                        
        return item


    def __load_item(self, idx):
        samples = self.samples[idx]
        url = samples['url']
        texts = copy.deepcopy(samples['texts'])
        num_texts = len(texts)
        optional_rel = {}

       # limit texts nums
        if num_texts > self.max_num_texts:
            texts = texts[:self.max_num_texts]
            num_texts = self.max_num_texts

        # sort texts by pos(x1,y1, x2, y2)
        texts = sort_texts_by_pos(texts)

        # gt img
        poster_im = read_im(url, root=self.gt_im_path)

        # check text pos
        poster_h, poster_w, _ = poster_im.shape
        for i in range(num_texts):
            texts[i]['pos'] = clamp_bbox_to_image(texts[i]['pos'], poster_w, poster_h)

        # pre-process each image' size to adapt the model's input
        poster_h, poster_w, _ = poster_im.shape
        new_h, new_w, reszie_scale = cal_resize_and_padding((poster_h, poster_w), self.input_size)
        poster_im = cv2.resize(poster_im, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        optional_rel['target_size'] = (new_h, new_w)
        optional_rel['original_size'] = (poster_h, poster_w)
        

        # pre-process each text' pos to adapt the model's input
        poster_h, poster_w, _ = poster_im.shape
        for i in range(num_texts):
            texts[i]['pos'] = reisize_box_by_scale(texts[i]['pos'], reszie_scale)

        # get bg image
        bg_im = poster_im.copy()

        if self.bg_inpaint:
            poster_h, poster_w, _ = poster_im.shape
            subject_mask = read_im(url[:-4] + '.png', root=self.mask_im_path)
            subject_mask = cv2.cvtColor(subject_mask, cv2.COLOR_RGB2GRAY)
            subject_mask = cv2.resize(subject_mask, (poster_w, poster_h), interpolation=cv2.INTER_NEAREST)
            if self.erode_mask:
                subject_mask = cv2.erode(subject_mask, np.ones((3, 3), np.uint8), iterations=1)
        else:
            subject_mask = None

        # gt poster
        gt_im = bg_im.copy()        
        gt_im = copy_text_to_bg(poster_im, gt_im, texts)

        # mask, text region->0, other region->1, but after transform text->-1, other->1 
        text_mask = create_mask_by_text((poster_h, poster_w), texts)
       
        # create empty image
        empty_image = np.zeros_like(gt_im)

        # padding image
        gt_im = pad_image_to_shape(gt_im, self.input_size, pad_value=255)
        bg_im = pad_image_to_shape(bg_im, self.input_size, pad_value=255)
        text_mask = pad_image_to_shape(text_mask, self.input_size, pad_value=255)
        empty_image = pad_image_to_shape(empty_image, self.input_size, pad_value=0)
        if self.bg_inpaint:
            optional_rel['subject_mask'] = pad_image_to_shape(subject_mask, self.input_size, pad_value=255)

        prompt = samples['caption'] if 'caption' in samples else self.prompt_fix

        with torch.no_grad():
            # Get texts feature list
            text_features, ocr_token_masks = get_char_features_by_text(texts, self.char2feat, self.char_padding_to_len)
            optional_rel['text_embeds'] = text_features
            optional_rel['text_token_masks'] = ocr_token_masks

            # ocr feature dim and pos dim
            pos_dim = self.char_pos_encoding_dim + self.text_pos_encoding_dim
            feature_dim = text_features[0].shape[-1]

            # Get char_level pos encoding
            char_positional_encoding = get_positional_encoding(self.char_padding_to_len, self.char_pos_encoding_dim) # N*32
            for i in range(len(text_features)):
                text_features[i] = torch.cat([text_features[i], ocr_token_masks[i].unsqueeze(-1) * char_positional_encoding], dim=-1) # N*(C+32)

            # Text_level pos encoding
            for i in range(len(text_features)):
                coords = pos2coords(texts[i]['pos']) # xyxy -> xywh
                coords_norm = torch.tensor(normalize_coordinates(coords, self.input_size[1], self.input_size[0])) # 4
                text_coords_embed = self.fourier_embedder(coords_norm) # 4-> 32
                text_coords_embed = text_coords_embed.unsqueeze(0).repeat(self.char_padding_to_len, 1) # N*32
                text_features[i] = torch.cat([text_features[i], ocr_token_masks[i].unsqueeze(-1) * text_coords_embed], dim=-1) # N*(C+32)

            # Handle ocr features(list) to a tensor
            max_token_num = self.char_padding_to_len * self.max_num_texts # to simplfiy, only no SEP
            # padding_token_num = max_token_num
            padding_token_num = max_token_num - self.char_padding_to_len * len(text_features)
            texts_and_sep_list = []
            for i in range(len(text_features)):
                texts_and_sep_list.append(text_features[i])
            texts_and_sep_list.append(torch.zeros((padding_token_num, pos_dim+feature_dim)))

            texts_all_features = torch.cat(texts_and_sep_list, dim=0) # eg. 5*16 = 80
            optional_rel['text_embeds'] = texts_all_features

            # Handle masks(list) to a tensor
            ocr_token_masks = [mask.unsqueeze(0) for mask in ocr_token_masks]
            ocr_token_masks = torch.cat(ocr_token_masks, dim=0)
            ocr_token_masks = torch.cat([ocr_token_masks, torch.zeros((self.max_num_texts - ocr_token_masks.shape[0], ocr_token_masks.shape[1]))], dim=0)
            optional_rel['text_token_masks'] = ocr_token_masks


        rel = {
            'url':url,
            'texts': texts,
            'gt_im': gt_im,
            'bg_im': bg_im,
            'mask':text_mask,
            "num_texts":num_texts,
            'caption':prompt,
            'controlnet_im':empty_image
        }

        rel.update(optional_rel)


        if self.transform:
            for k,v in rel.items():
                if isinstance(v, np.ndarray):
                    rel[k] = self.transform(v)

                
        return rel 

