import pandas as pd

class Structure_data_Pipeline:
    def __init__(self,sentence_path, bert_encoder):
        self.bert_encoder = bert_encoder
        self.sentence_data = pd.read_csv(sentence_path)
        

    def return_sentence_labels_and_embeddings(self):
        three_data_dict = []

        for i in range(len(self.sentence_data)):
            
            anger = self.sentence_data.iloc[i,2]
            fear = self.sentence_data.iloc[i,3]
            joy = self.sentence_data.iloc[i,4]
            sadness = self.sentence_data.iloc[i,5]
            surprise = self.sentence_data.iloc[i,6]
            texts = self.sentence_data.iloc[i,1]
            embedding = self.bert_encoder.get_embeddings(texts)
            key_value_pair = {"text": texts, "embedding": embedding, "emotion_labels": {"anger": anger, "fear": fear, "joy": joy, "sadness": sadness, "surprise": surprise}}
            three_data_dict.append(key_value_pair)
        return three_data_dict




