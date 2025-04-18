from PIL import Image
import numpy as np
import cv2
import torch

# x1,y1,x2,y2 -> x1,y1,w,h
def pos2coords(pos):
    return (pos[0], pos[1], pos[2]-pos[0], pos[3]-pos[1])


# x1,y1,w,h -> x1,y1,x2,y2
def coords2pos(coords):
    return (coords[0], coords[1], coords[2]+coords[0], coords[3]+coords[1])


def normalize_coordinates(coordinates, original_width, original_height):
    """
    Normalize coordinates to the range [0, 1].
    
    Args:
        coordinates (list): A list of coordinates in the form [x, y, w, h] or [x1, y1, x2, y2].
        original_width (int): The width of the original image.
        original_height (int): The height of the original image.
    
    Returns:
        list: A list of normalized coordinates.
    """
    if len(coordinates) == 4:  # Handle [x, y, w, h] or [x1, y1, x2, y2] format
        x, y, w, h = coordinates
        normalized_coords = [
            x / original_width,
            y / original_height,
            w / original_width,
            h / original_height
        ]
    else:
        raise ValueError("Coordinates must be in the form [x, y, w, h] or [x1, y1, x2, y2].")

    return normalized_coords


def convert_to_rgb(image):
    """
    Convert RGBA or RGB images to RGB format

    Parameters:
    image: numpy.ndarray, Input image (in RGB or RGBA format)

    Returns:
    numpy.ndarray: Image in RGB format
    """
    # If the image is in RGBA format, convert it to RGB
    if image.shape[-1] == 4:
        # Use a white background
        background = np.ones_like(image[..., :3]) * 255
        alpha = image[..., 3:] / 255.0
        image = image[..., :3] * alpha + background * (1 - alpha)
        image = image.astype(np.uint8)
    
    return image[..., :3]  # Ensure the return is in RGB format


def cal_resize_and_padding(img_size, model_input_size):
    ori_h, ori_w = img_size
    target_h, target_w = model_input_size
    
    scale = min(target_h/ori_h, target_w/ori_w)
    new_h, new_w = int(scale * ori_h), int(scale * ori_w)


    return new_h, new_w, scale


def reisize_box_by_scale(box, scale):
    return [int(x * scale) for x in box]


def pad_image_to_shape(image, target_shape, pad_value=0):
    """
    Pad the image to the specified shape.

    Parameters:
    - image: Image to be padded, in numpy array format.
    - target_shape: Target shape (height, width).
    - pad_value: Default value for padding, defaults to 0.

    Returns:
    - Padded image, in numpy array format.
    """
    original_shape = image.shape[:2]
    padding = [
        (0, max(0, target_shape[0] - original_shape[0])),  # Padding in the height direction
        (0, max(0, target_shape[1] - original_shape[1])),  # Padding in the width direction
    ]
    
    if len(image.shape) == 3:  # Check if the image is colored
        padding.append((0, 0))  # Do not pad the channel dimension for colored images
    
    padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
    return padded_image
    

def clamp_bbox_to_image(bbox, image_width, image_height):
    """
    Adjusts bounding box coordinates to ensure they do not exceed image boundaries.

    :param bbox: A tuple containing 4 values (x1, y1, x2, y2), representing bounding box coordinates.
    :param image_width: Width of the image.
    :param image_height: Height of the image.
    :return: Adjusted bounding box coordinates (new x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = bbox

    # Ensure bounding box coordinates do not exceed image boundaries
    x1_clamped = max(0, min(x1, image_width))
    y1_clamped = max(0, min(y1, image_height))
    x2_clamped = max(0, min(x2, image_width))
    y2_clamped = max(0, min(y2, image_height))

    return (x1_clamped, y1_clamped, x2_clamped, y2_clamped)


def create_mask_by_text(im_size, texts):
    h, w = im_size
    mask = np.ones((h, w, 1),dtype=np.uint8) * 255
    for text in texts:
        x1, y1, x2, y2 = text['pos']
        mask[y1:y2, x1:x2, :] = 0

    return mask


def get_char_features_by_text(texts, char2feat, char_padding_num):
    text_features = []
    token_masks = []

    for text in texts:
        content = text['content']
        with torch.no_grad():
            # Get default feature
            default_feature = char2feat[' '][None, ...]

            # Pre-allocate space and fill with default features
            char_features = torch.empty((len(content), *default_feature.shape[1:]), dtype=default_feature.dtype)
            default_val = default_feature.squeeze(0)
            for i, c in enumerate(content):
                if c in char2feat:
                    char_features[i] = char2feat[c]
                else:
                    char_features[i] = default_val

            # Get shape information
            N = char_features.shape[0]

            # Use zeros to create a padding tensor for concatenation
            padding_tensor = torch.zeros((char_padding_num - N, *char_features.shape[1:]), dtype=char_features.dtype)

            # Concatenation
            char_features = torch.cat([char_features, padding_tensor], dim=0)

            assert char_features.shape[0] == char_padding_num, "len(char_features) == padding_to_len"

            text_features.append(char_features)

            # char token mask
            char_token_mask = torch.zeros(char_padding_num)

            char_token_mask[:N] = 1

            token_masks.append(char_token_mask)
            
    return text_features, token_masks


# define cosine position encoding function
def get_positional_encoding(length, channels):
    position = np.arange(length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, channels, 2) * -(np.log(10000.0) / channels))
    pe = np.zeros((length, channels))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe)


def save_image(image, im_path):
    if isinstance(image, Image.Image):
        image = np.array(image) 
    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
    else:
        raise ValueError("image must be PIL.Image.Image or numpy.ndarray")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(im_path, image)


def post_process(image, target_size):
    """
    对模型输出结果进行后处理
    
    Args:
        image: 模型的原始输出结果，是PIL.Image格式
        target_size: 目标尺寸，包含(height, width)的元组或列表
        
    Returns:
        PIL.Image: 裁剪后的图像
    """
    im_h, im_w = target_size
    
    # 裁剪图像到目标尺寸
    crop_rel = image.crop((0, 0, im_w, im_h))
    
    return crop_rel