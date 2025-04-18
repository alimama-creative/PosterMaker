import cv2

import torch.nn.functional as F
from torchvision import transforms

from models.text_embedder import TextEmbedder
from utils.utils import *


class UserInputProcessor():
    def __init__(self, input_size=(1024, 1024), erode_mask=False):
        self.input_size = input_size
        self.erode_mask = erode_mask
        self.text_embedder = TextEmbedder()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __call__(self, image, mask, texts, prompt):
        """
        Preprocess user input image and text data
        Parameters:
        image: numpy array, Input image (H, W, C)
        texts: list of dict, A list containing text content and location information
        Each dict format: {"content": str, "pos": [x1,y1,x2,y2]}
        input_size: tuple, Model input size (H, W)
        prompt: str, Text Tips
        return:
        dict: Preprocessed data
        """

        # rgba to rgb 
        image = convert_to_rgb(image)

        # resize
        input_size = self.input_size
        poster_h, poster_w, _ = image.shape
        new_h, new_w, resize_scale = cal_resize_and_padding((poster_h, poster_w), input_size)
        processed_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        subject_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        if self.erode_mask:
            subject_mask = cv2.erode(subject_mask, np.ones((3, 3), np.uint8), iterations=1)

        # adjust the position of the text box
        for text in texts:
            text['pos'] = reisize_box_by_scale(text['pos'], resize_scale)
            text['pos'] = clamp_bbox_to_image(text['pos'], new_w, new_h)
        
        # create text mask
        text_mask = create_mask_by_text((new_h, new_w), texts)

        # create empty image
        empty_image = np.zeros_like(processed_image)

        # paddding to input_size
        processed_image = pad_image_to_shape(processed_image, input_size, pad_value=255)
        text_mask = pad_image_to_shape(text_mask, input_size, pad_value=255)
        subject_mask = pad_image_to_shape(subject_mask, input_size, pad_value=255)
        empty_image = pad_image_to_shape(empty_image, input_size, pad_value=0)
        
        # text feature
        text_embeds = self.text_embedder(texts)

        # tensor and normalize
        processed_image = self.transform(processed_image)
        subject_mask = self.transform(subject_mask)
        text_mask = self.transform(text_mask)
        empty_image = self.transform(empty_image)

        # sd3 inpaint_controlnet input
        control_mask = subject_mask
        control_mask = ((control_mask + 1.0) / 2.0) # [-1,1]->[0,1], 0 means need inpaint
        cond_image_inpaint = (processed_image + 1) * control_mask - 1

        # unsqueeze 1, C, H, W
        cond_image_inpaint = cond_image_inpaint.unsqueeze(0)
        control_mask = control_mask.unsqueeze(0)
        text_embeds = text_embeds.unsqueeze(0)
        empty_image = empty_image.unsqueeze(0)
        
        # return data
        result = {
            'cond_image_inpaint': cond_image_inpaint,
            'control_mask': control_mask,
            # 'texts': texts,
            'prompt': prompt,
            # 'text_mask': text_mask,
            'text_embeds': text_embeds,
            'target_size': (new_h, new_w),
            # 'original_size': (poster_h, poster_w),
            'controlnet_im':empty_image
        }
        
        return result
