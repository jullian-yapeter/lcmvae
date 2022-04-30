from transformers import BertTokenizer, BertModel, ViTModel, ViTMAEModel
import torch
import cv2
from utils import has_internet

class BertEncoder():
    def __init__(self, device=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.pretrained_path = 'bert-base-uncased' \
            if has_internet() else './saved_models/BERT'

        self.tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_path, do_lower_case=True)
        self.model = BertModel.from_pretrained(
            self.pretrained_path).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = self.model.config.hidden_size

    def forward(self, sentences):
        with torch.no_grad():
            tokenized_sentences = self.tokenizer(
                sentences, add_special_tokens=True, truncation=True,
                padding="max_length", return_attention_mask=True, return_tensors="pt").to(self.device)
            sentence_embeddings = self.model(
                **tokenized_sentences).last_hidden_state[:, 0, :]
        return sentence_embeddings


class VitEncoder():
    def __init__(self, device=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.pretrained_path = 'google/vit-base-patch16-224-in21k' \
            if has_internet() else './saved_models/ViT'
        # NOTE: feature_extractor was created during building a dataset in PreTrainer 
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained(
        #     "google/vit-base-patch16-224-in21k", do_resize=True)
        self.model = ViTModel.from_pretrained(self.pretrained_path).to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = self.model.config.hidden_size

    def forward(self, images):
        # image_features = self.feature_extractor(
        #     images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeddings = self.model(
                images).last_hidden_state[:, 0, :]
        return image_embeddings


class VitMaeEncoder():
    def __init__(self, mask_ratio=0.75, mode="all", device=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.pretrained_path = 'facebook/vit-mae-base' \
            if has_internet() else './saved_models/ViTMAE'
        # NOTE: feature_extractor was created during building a dataset in PreTrainer 
        # self.feature_extractor = AutoFeatureExtractor.from_pretrained(
        #     "facebook/vit-mae-base")
        if mode not in ["all", "mean", "cls"]:
            raise ValueError(
                f"mode should be 'all', 'mean', or 'cls', but now is {mode}")
        self.mode = mode
        self.model = ViTMAEModel.from_pretrained(
            self.pretrained_path, mask_ratio=mask_ratio).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = self.model.config.hidden_size

    def forward(self, images):
        # NOTE: feature_extractor was created during building a dataset in PreTrainer 
        # image_features = self.feature_extractor(
        #     images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(images)
            mask = outputs.mask
            if self.mode == "cls":
                image_embeddings = self.model(images).last_hidden_state[:, 0, :]
            elif self.mode == "all":
                image_embeddings = self.model(images)[0][:, 1:]
                image_embeddings = image_embeddings.reshape(image_embeddings.shape[0], -1)
            elif self.mode == "mean":
                image_embeddings = torch.mean(self.model(
                    images)[0][:, 1:], dim=1)
        return image_embeddings, mask


class ImageCaptionEncoder():
    def __init__(self, is_mae=True, mask_ratio=0.75, no_caption=False, mode="all", device=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.bert = None if no_caption else BertEncoder(device=self.device)
        self.is_mae = is_mae
        if is_mae:
            self.vit = VitMaeEncoder(mask_ratio=mask_ratio, mode=mode, device=self.device)
        else:
            self.vit = VitEncoder(device=self.device)

    def forward(self, images, captions):
        mask = None
        if self.is_mae:
            image_encodings, mask = self.vit.forward(images)
        else:
            image_encodings = self.vit.forward(images)
        caption_encodings = self.bert.forward(captions) if self.bert else torch.zeros((image_encodings.shape[0], 768))
        return torch.cat((image_encodings, caption_encodings), dim=-1), mask


if __name__=="__main__":
    import cv2
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    dog_im = cv2.imread("dataset/images/dog.png")
    cat_im = cv2.imread("dataset/images/cat.png")
    images = [dog_im, cat_im]
    captions = ["smiling happy dog", "confused orange cat"]
    
    im_cap_encoder = ImageCaptionEncoder(device=device)
    print(im_cap_encoder.forward(images, captions).shape)
