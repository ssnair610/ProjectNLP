import os
import spacy
import numpy as np
import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


from utils.constants import LABEL_COLUMNS, MAX_VOCAB_SIZE
from utils.cleaner import text_cleaner, vocabulary_builder

nlp_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class EvaluationData:
    file_path: str | None
    data_frame: DataFrame
    stop_words: set[str]
    label_cols: list[str] = LABEL_COLUMNS
    num_labels: int
    vocabulary: tuple[dict[str, int], list[str]]

    def __init__(self, file_path: str) -> None:
        self.file_path = os.path.join(file_path)
        self.data_frame = pd.read_csv(self.file_path)
        self.stop_words = set(stopwords.words("english"))
        self.label_cols = [col for col in self.label_cols if col in self.data_frame]

        cleaned_data = self.data_frame["text"].astype(str).apply(text_cleaner)
        y_data = self.data_frame[self.label_cols].values.astype(np.float32)
        x_all = cleaned_data.tolist()
        x_train, x_val, y_train, y_val = train_test_split(
            x_all, y_data, test_size=0.1, random_state=69
        )

        self.num_labels = y_train.shape[1]
        self.vocabulary = vocabulary_builder(x_train, vocabsize=MAX_VOCAB_SIZE)
