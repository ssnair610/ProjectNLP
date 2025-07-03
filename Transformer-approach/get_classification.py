# get_classification.py

import torch
from bert_encoder import Bert_Encoder

def get_classification(text, bert_encoder, classification_model_path):
    embedding = bert_encoder.get_embeddings(text).squeeze(0)

    # Set weights_only=False to allow full model unpickling
    model = torch.load(classification_model_path, weights_only=False)
    model.eval()

    with torch.no_grad():
        output = model(embedding.unsqueeze(0))  # predict
        result = (torch.sigmoid(output) > 0.5).int().squeeze(0).tolist()

    return result  # list of 0s and 1s, e.g. [0, 1, 1, 0, 0]