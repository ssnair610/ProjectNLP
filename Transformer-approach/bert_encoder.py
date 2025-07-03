from transformers import AutoTokenizer, AutoModel
import torch


# 768 is the default hidden size for BERT base models

class Bert_Encoder:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state[:, 0, :]  # [CLS] Token


