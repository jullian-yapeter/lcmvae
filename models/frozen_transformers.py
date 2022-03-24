from transformers import AutoFeatureExtractor, BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel, ViTMAEModel
import torch


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
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k", do_resize=True)
        self.model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k").to(self.device)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, images):
        image_features = self.feature_extractor(
            images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeddings = self.model(**image_features).last_hidden_state[:, 0, :]
        return image_embeddings


class VitMaeEncoder():
    def __init__(self, device=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/vit-mae-base")
        self.model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        self.hidden_size = self.model.config.hidden_size

    def forward(self, images):
        image_features = self.feature_extractor(
            images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeddings = self.model(
                **image_features).last_hidden_state[:, 0, :]
        return image_embeddings


class ImageCaptionEncoder():
    def __init__(self, is_mae=True, device=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.bert = BertEncoder(device=self.device)
        if is_mae:
            self.vit = VitMaeEncoder(device=self.device)
        else:
            self.vit = VitEncoder(device=self.device)

    def forward(self, images, captions):
        image_encodings = self.vit.forward(images)
        caption_encodings = self.bert.forward(captions)
        return torch.cat((image_encodings, caption_encodings), dim=-1)


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
