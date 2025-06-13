import re
import numpy as np
from collections import Counter


TOKEN_PATTERN = re.compile(r"\b\w+\b")

def simpleTokenize(text: str) -> list:
    return TOKEN_PATTERN.findall(text.lower())

def vocabularyBuilder(texts: list, vocabsize: int):
    counter = Counter()
    for t in texts:
        tokens = simpleTokenize(t)
        counter.update(tokens)
    
    mostCommon = counter.most_common(vocabsize - 2)
    indextoword = ["<pad>", "<unk>"] + [token for token, _ in mostCommon]
    wordtoindex = {w: i for i, w in enumerate(indextoword)}

    return wordtoindex, indextoword



MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100

wordtoindex, indextoword = vocabularyBuilder(xTrain, vocabsize = MAX_VOCAB_SIZE)
vocabSize = len(indextoword)

padIndex = wordtoindex["<pad>"]
unkIndex = wordtoindex["<unk>"]

def encodePadFunction(texts: list, wordtoindex: dict, sequenceLength: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    encodings = []

    for t in texts:
        tokens = simpleTokenize(t)
        tokenIDs = [wordtoindex.get(tok, unkIndex) for tok in tokens]
        if len(tokenIDs) > sequenceLength:
            tokenIDs = tokenIDs[:sequenceLength]
        else:
            tokenIDs = tokenIDs + [padIndex] * (sequenceLength - len(tokenIDs))
        encodings.append(tokenIDs)

    return np.array(encodings, dtype = np.int64)