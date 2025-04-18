import torch
from utils.utils import *


class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):

        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )  

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = [] # B, L ,4
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)


class TextEmbedder():
    def __init__(self, feature_dict_path='./assets/char2feat_ppocr_neck64_avg.pth'):
        # char-level text feature params
        self.max_num_texts = 7
        self.char_padding_to_len = 16
        self.char_pos_encoding_dim = 32
        self.text_pos_encoding_dim = 32
        self.input_size = (1024, 1024)
        
        # positional encoding
        self.fourier_embedder = FourierEmbedder(num_freqs=self.text_pos_encoding_dim // (4*2)) # 4 is xywh, 2 is cos&sin

        # load char feature dict
        self.char2feat = torch.load(feature_dict_path)


    # texts: [{'content': 'xxx', 'pos': [x1, y1, x2, y2]}, {'content': 'xxx', 'pos': [x1, y1, x2, y2]}, ...]
    def __call__(self, texts):
        with torch.no_grad():
            # Get texts feature list
            text_features, ocr_token_masks = get_char_features_by_text(texts, self.char2feat, self.char_padding_to_len)

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
            padding_token_num = max_token_num - self.char_padding_to_len * len(text_features)
            texts_and_sep_list = []
            for i in range(len(text_features)):
                texts_and_sep_list.append(text_features[i])
            texts_and_sep_list.append(torch.zeros((padding_token_num, pos_dim+feature_dim)))

            texts_all_features = torch.cat(texts_and_sep_list, dim=0) # eg. 5*16 = 80
            
            return  texts_all_features


    def get_text_embeds_batch(self, batch_texts):
        """
        Args:
            batch_texts: list of texts list
            eg. [
                [{'content': 'xxx', 'pos': [x1,y1,x2,y2]}, ...],  # sample 1
                [{'content': 'xxx', 'pos': [x1,y1,x2,y2]}, ...],  # sample 2 
                ...
            ]
        Returns:
            batch_embeds: tensor of shape (batch_size, max_token_num, feature_dim)
        """
        batch_embeds = []
        for texts in batch_texts:
            # Process single sample
            text_embeds = self.__call__(texts)  # (max_token_num, feature_dim) 
            batch_embeds.append(text_embeds)

        # Stack along batch dimension
        batch_embeds = torch.stack(batch_embeds, dim=0)  # (batch_size, max_token_num, feature_dim)

        return batch_embeds

