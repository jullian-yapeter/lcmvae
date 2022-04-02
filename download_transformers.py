from transformers import AutoFeatureExtractor, BertTokenizer, BertModel, ViTModel, ViTMAEModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenizer.save_pretrained('./saved_models/BERT')
del tokenizer

BERT = BertModel.from_pretrained('bert-base-uncased')
BERT.save_pretrained('./saved_models/BERT')
del BERT

ViT = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
ViT.save_pretrained('./saved_models/ViT')
del ViT

ViTMAE = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
ViTMAE.save_pretrained('./saved_models/ViTMAE')
del ViTMAE

AFE = AutoFeatureExtractor.from_pretrained('facebook/vit-mae-base')
AFE.save_pretrained('./saved_models/ViTMAE')
del AFE