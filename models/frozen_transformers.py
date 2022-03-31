from transformers import AutoFeatureExtractor, BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel, ViTMAEModel
import torch
import cv2


class BertEncoder():
    def __init__(self, device=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        self.model = BertModel.from_pretrained(
            "bert-base-uncased").to(self.device)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, sentences):
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
        # FIXME: feature_extractor was created during building a dataset in PreTrainer 
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained(
        #     "google/vit-base-patch16-224-in21k", do_resize=True)
        # FIXME: dataset.MyCocoCaption is using 'google/vit-base-patch16-224'
        self.model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k").to(self.device)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, images):
        # image_features = self.feature_extractor(
        #     images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeddings = self.model(
                images).last_hidden_state[:, 0, :]
        return image_embeddings


class VitMaeEncoder():
    def __init__(self, mask_ratio=0.75, device=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # FIXME: feature_extractor was created during building a dataset in PreTrainer 
        # self.feature_extractor = AutoFeatureExtractor.from_pretrained(
        #     "facebook/vit-mae-base")
        self.model = ViTMAEModel.from_pretrained(
            "facebook/vit-mae-base", mask_ratio=mask_ratio).to(self.device)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, images):
        # FIXME: feature_extractor was created during building a dataset in PreTrainer 
        # image_features = self.feature_extractor(
        #     images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(images)
            mask = outputs.mask
            image_embeddings = outputs.last_hidden_state[:, 0, :]
        return image_embeddings, mask


class ImageCaptionEncoder():
    def __init__(self, is_mae=True, mask_ratio=0.75, device=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.bert = BertEncoder(device=self.device)
        self.is_mae = is_mae
        if is_mae:
            self.vit = VitMaeEncoder(mask_ratio=mask_ratio, device=self.device)
        else:
            self.vit = VitEncoder(device=self.device)

    def forward(self, images, captions):
        mask = None
        if self.is_mae:
            image_encodings, mask = self.vit.forward(images)
        else:
            image_encodings = self.vit.forward(images)
        caption_encodings = self.bert.forward(captions)
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
