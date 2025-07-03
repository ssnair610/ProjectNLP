import re
import numpy as np
import spacy
from nltk.corpus import stopwords
from collections import Counter

from utils.constants import MAX_SEQUENCE_LENGTH


nlp_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])
stop_words = set(stopwords.words("english"))


def clean_text_svm(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text.strip()


def text_cleaner(text):
    tempDoc = nlp_model(text)
    token = [
        tok.lemma_.lower()
        for tok in tempDoc
        if not tok.is_stop and not tok.is_punct and tok.lemma_ != "-PRON-"
    ]
    return " ".join(token)


def simple_tokenize(text: str) -> list:
    TOKEN_PATTERN = re.compile(r"\b\w+\b")

    return TOKEN_PATTERN.findall(text.lower())


def vocabulary_builder(texts: list, vocabsize: int):
    counter = Counter()
    for t in texts:
        tokens = simple_tokenize(t)
        counter.update(tokens)

    most_common = counter.most_common(vocabsize - 2)
    index_to_word = ["<pad>", "<unk>"] + [token for token, _ in most_common]
    word_to_index = {w: i for i, w in enumerate(index_to_word)}

    return word_to_index, index_to_word


def encode_pad_function(
    texts: list, wordtoindex: dict, sequenceLength: int = MAX_SEQUENCE_LENGTH
) -> np.ndarray:
    encodings = []
    padIndex = wordtoindex["<pad>"]
    unkIndex = wordtoindex["<unk>"]

    for t in texts:
        tokens = simple_tokenize(t)
        tokenIDs = [wordtoindex.get(tok, unkIndex) for tok in tokens]
        if len(tokenIDs) > sequenceLength:
            tokenIDs = tokenIDs[:sequenceLength]
        else:
            tokenIDs = tokenIDs + [padIndex] * (sequenceLength - len(tokenIDs))
        encodings.append(tokenIDs)

    return np.array(encodings, dtype=np.int64)
