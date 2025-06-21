
import os
import re
import pandas as pd
import numpy as np
from collections import Counter
import spacy

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

import joblib


dataPath = os.path.join("Data", "track-a.csv")


dataFrame = pd.read_csv(dataPath)


nlpModel = spacy.load("en_core_web_sm", disable=["parser","ner"])


def cleanerFunction(text):
    tempDoc = nlpModel(text)
    token = [
        tok.lemma_.lower()
        for tok in tempDoc
        if not tok.is_stop and not tok.is_punct and tok.lemma_ != "-PRON-"
    ]
    return " ".join(token)


dataFrame["Spacy_text"] = dataFrame["text"].astype(str).apply(cleanerFunction)


labelColumns = [col for col in ["anger", "fear", "joy", "sadness", "surprise"] if col in dataFrame]
yData = dataFrame[labelColumns].values.astype(np.float32)


xAll = dataFrame["Spacy_text"].tolist()
xTrain, xVal, yTrain, yVal = train_test_split(xAll, yData, test_size = 0.1, random_state = 69)


xTrain = xTrain.copy()
xVal = xVal.copy()

numLabels = yTrain.shape[1]


userDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def encodePadFunction(texts: list, wordtoindex: dict, sequenceLength: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    encodings = []
    padIndex = wordtoindex["<pad>"]
    unkIndex = wordtoindex["<unk>"]

    for t in texts:
        tokens = simpleTokenize(t)
        tokenIDs = [wordtoindex.get(tok, unkIndex) for tok in tokens]
        if len(tokenIDs) > sequenceLength:
            tokenIDs = tokenIDs[:sequenceLength]
        else:
            tokenIDs = tokenIDs + [padIndex] * (sequenceLength - len(tokenIDs))
        encodings.append(tokenIDs)

    return np.array(encodings, dtype = np.int64)


trainEncode = encodePadFunction(xTrain, wordtoindex, sequenceLength = MAX_SEQUENCE_LENGTH)
valEncode = encodePadFunction(xVal, wordtoindex, sequenceLength = MAX_SEQUENCE_LENGTH)


class feedforwardNeuralNetwork(nn.Module):
    def __init__(
            self,
            vocabSize: int,
            embeddingDim: int,
            hiddenUnit1: int,
            hiddenUnit2: int,
            dropoutRate: float,
            numLabels: int,
            padIndex: int
        ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx = padIndex)
        self.dropout1 = nn.Dropout(dropoutRate)
        self.fc1 = nn.Linear(embeddingDim, hiddenUnit1)
        self.bn1 = nn.BatchNorm1d(hiddenUnit1)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropoutRate)
        self.fc2 = nn.Linear(hiddenUnit1, hiddenUnit2)
        self.bn2 = nn.BatchNorm1d(hiddenUnit2)
        self.dropout3 = nn.Dropout(dropoutRate)
        self.fc_out = nn.Linear(hiddenUnit2, numLabels)

    def forward(self, x):
        emb = self.embedding(x)             
        avgEmb = emb.mean(dim = 1) 
        h1 = self.fc1(avgEmb)
        h1 = self.bn1(h1)
        h1 = self.relu(h1)
        h1 = self.dropout2(h1)
        h2 = self.fc2(h1)
        h2 = self.bn2(h2)
        h2 = self.relu(h2)
        h2 = self.dropout3(h2)
        return self.fc_out(h2)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocabSize: int,
        embeddingDim: int,
        hiddenDim: int,
        rnnLayers: int,
        bidirectional: bool,
        dropoutRate: float,
        denseUnits: int,
        numLabels: int,
        padIndex: int,
        useAttention: bool = False,
    ) -> None:
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels = embeddingDim,
            out_channels = embeddingDim,
            kernel_size = 5,
            padding = 2
        )
        self.relu_cnn = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size = 2)

        self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx = padIndex)
        self.dropout_emb = nn.Dropout(dropoutRate)
        self.lstm = nn.LSTM(
            input_size = embeddingDim,
            hidden_size = hiddenDim,
            num_layers = rnnLayers,
            bidirectional = bidirectional,
            batch_first = True,
            dropout = dropoutRate if rnnLayers > 1 else 0.0
        )

        self.use_attention = useAttention
        directionFactor = 2 if bidirectional else 1
        if useAttention:
            self.attn_linear = nn.Linear(hiddenDim * directionFactor, hiddenDim * directionFactor)
            self.attn_v = nn.Linear(hiddenDim * directionFactor, 1, bias = False)

        self.fc1 = nn.Linear(hiddenDim * directionFactor, denseUnits)
        self.bn1 = nn.BatchNorm1d(denseUnits)
        self.relu = nn.ReLU()
        self.dropoutFc = nn.Dropout(dropoutRate)
        self.output_layer = nn.Linear(denseUnits, numLabels)

    def forward(self, x: torch.LongTensor) -> Tensor:
        pad_idx = self.embedding.padding_idx
        lengths = (x != pad_idx).sum(dim = 1)
        lengths = torch.clamp(lengths, min = 1)

        embTensor: Tensor = self.embedding(x)  # type: ignore
        embTensor = self.dropout_emb(embTensor)   #type: ignore

        c_in = embTensor.transpose(1, 2)              
        c_out = self.relu_cnn(self.conv1d(c_in))
        c_out = self.pool(c_out)

        rnn_in = c_out.transpose(1, 2)
        lengths = torch.clamp(lengths // 2, min = 1)

        packed = pack_padded_sequence(rnn_in, lengths.cpu(), batch_first = True, enforce_sorted = False)
        rnnOut, _ = self.lstm(packed)
        rnnOut, _ = pad_packed_sequence(rnnOut, batch_first = True)

        if self.use_attention:
            scores = torch.tanh(self.attn_linear(rnnOut))
            weights = torch.softmax(self.attn_v(scores), dim = 1)
            finalFeat = (weights * rnnOut).sum(dim = 1)
        else:
            idx = torch.arange(x.size(0), device = x.device)
            finalFeat = rnnOut[idx, lengths - 1]

        h: Tensor = self.fc1(finalFeat) #type: ignore
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropoutFc(h)  #type: ignore
        logits: Tensor = self.output_layer(h)   #type: ignore
        return logits


class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocabSize: int,
        embeddingDim: int,
        hiddenDim: int,
        rnnLayers: int,
        bidirectional: bool,
        dropoutRate: float,
        denseUnits: int,
        numLabels: int,
        padIndex: int,
        useAttention: bool = False
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels = embeddingDim,
            out_channels = embeddingDim,
            kernel_size = 5,
            padding = 2
        )
        self.relu_cnn = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size = 2)

        self.embedding: nn.Embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx = padIndex)
        self.dropout_emb: nn.Dropout = nn.Dropout(dropoutRate)
        self.gru: nn.GRU = nn.GRU(
            input_size = embeddingDim,
            hidden_size = hiddenDim,
            num_layers = rnnLayers,
            bidirectional = bidirectional,
            batch_first = True,
            dropout = dropoutRate if rnnLayers > 1 else 0.0
        )

        self.use_attention: bool = useAttention
        factor = 2 if bidirectional else 1
        if useAttention:
            self.attn_linear: nn.Linear = nn.Linear(hiddenDim * factor, hiddenDim * factor)
            self.attn_v: nn.Linear = nn.Linear(hiddenDim * factor, 1, bias = False)

        self.fc1: nn.Linear = nn.Linear(hiddenDim * factor, denseUnits)
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(denseUnits)
        self.relu: nn.ReLU = nn.ReLU()
        self.dropout_fc: nn.Dropout = nn.Dropout(dropoutRate)
        self.output_layer: nn.Linear = nn.Linear(denseUnits, numLabels)

    def forward(self, x: torch.LongTensor) -> Tensor:
        lengths = (x != self.embedding.padding_idx).sum(dim = 1)
        lengths = torch.clamp(lengths, min = 1)

        emb = self.dropout_emb(self.embedding(x))

        c_in = emb.transpose(1, 2)
        c_out = self.relu_cnn(self.conv1d(c_in))
        c_out = self.pool(c_out)

        rnn_in = c_out.transpose(1, 2)
        lengths = torch.clamp(lengths // 2, min = 1)

        packed = pack_padded_sequence(rnn_in, lengths.cpu(), batch_first = True, enforce_sorted = False)
        rnnOut, _ = self.gru(packed)
        rnnOut, _ = pad_packed_sequence(rnnOut, batch_first = True)

        if self.use_attention:
            scores = torch.tanh(self.attn_linear(rnnOut))
            weights = torch.softmax(self.attn_v(scores), dim = 1)
            finalFeat = (weights * rnnOut).sum(dim = 1)
        else:
            idx = torch.arange(x.size(0), device = x.device)
            finalFeat = rnnOut[idx, lengths - 1]

        h: Tensor = self.fc1(finalFeat)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout_fc(h)
        logits: Tensor = self.output_layer(h)
        return logits


class metrics_rfc():

    def __init__(self, confusion_mat, log_level, title) -> None:
        self.confusion_mat = confusion_mat
        self.log_level = log_level
        self.title = title

    def accuracy(self, tp, tn, fp, fn):
        return (tp + tn) / (tp + tn + fp + fn)

    def precision(self, tp, fp):
        return tp / (tp + fp)

    def recall(self, tp, fn):
        return tp / (tp + fn)

    def f1_score(self, tp,fp,fn):
        precision_val = self.precision(tp,fp)
        recall_val = self.recall(tp,fn)
        return 2 * (precision_val * recall_val) / (precision_val + recall_val)

    def print_eval(self, title, accuracy, precision, recall, f1_score):
        print(f"{title}\n")
        print(f"accuracy: {round(accuracy,2)}")
        print(f"precision: {round(precision,2)}")
        print(f"recall: {round(recall,2)}")
        print(f"f1 Score: {round(f1_score,2)}")
        print("=======\n")


    def present_data(self):
        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_f1_score = 0

        for i in range(0, len(self.confusion_mat)):
            tp, fp = self.confusion_mat[i][0]
            fn, tn = self.confusion_mat[i][1]

            accuracy_val = self.accuracy(tp, tn, fp, fn)
            precision_val = self.precision(tp, fp)
            recall_val = self.recall(tp, fn)
            f1_score_val = self.f1_score(tp, fp, fn)

            total_accuracy += accuracy_val
            total_precision += precision_val
            total_recall += recall_val
            total_f1_score += f1_score_val

            if(self.log_level == "emotions"):
                self.print_eval(
                f"Emotion: {labelColumns[i]}",
                    accuracy_val,
                    precision_val,
                    recall_val,
                    f1_score_val,
                )

        avg_accuracy = total_accuracy / len(self.confusion_mat)
        avg_precision = total_precision / len(self.confusion_mat)
        avg_recall = total_recall / len(self.confusion_mat)
        avg_f1_score = total_f1_score / len(self.confusion_mat)

        if(self.log_level=="macro"):
            self.print_eval("Macro Average:", avg_accuracy, avg_precision, avg_recall, avg_f1_score)


def predictFNN(csvFile: str):
    df = pd.read_csv(csvFile)
    texts = df["text"].astype(str).apply(cleanerFunction).tolist()
    
    ckpt = torch.load("savedModel/ffnn.pth", map_location = "cpu")
    sd = ckpt["model_state_dict"]
    w2i = ckpt["wordtoindex"]
    padIndex = ckpt["padIndex"]
    maxseqlen = ckpt["max_sequence_length"]

    vocabSize, embeddingDim = sd["embedding.weight"].shape
    hiddenUnit1 = sd["fc1.weight"].shape[0]
    hiddenUnit2 = sd["fc2.weight"].shape[0]
    
    model = feedforwardNeuralNetwork(
        vocabSize = vocabSize,
        embeddingDim = embeddingDim,
        hiddenUnit1 = hiddenUnit1,
        hiddenUnit2 = hiddenUnit2,
        dropoutRate = 0.0,
        numLabels = numLabels,
        padIndex = padIndex
    )
    model.load_state_dict(sd)
    model.eval()

    x = encodePadFunction(texts, w2i, maxseqlen)
    xTensor = torch.from_numpy(x)

    with torch.no_grad():
        logits = model(xTensor)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
    return preds.tolist()  



def predictLSTM(csv_path: str):
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).apply(cleanerFunction).tolist()

    ckpt = torch.load("savedModel/rnnLSTM.pth", map_location = "cpu", weights_only = False)
    sd = ckpt["model_state_dict"]
    w2i = ckpt["wordtoindex"]
    pad = ckpt["padIndex"]
    maxseqlen = ckpt["max_sequence_length"]
    hp = ckpt["hyperparameters"]

    vocabSize, embeddingDim = sd["embedding.weight"].shape
    numLabels = sd["output_layer.weight"].shape[0]
    denseUnits, inFeat = sd["fc1.weight"].shape
    factor = 2 if hp["lstm_bidirectional"] else 1
    hiddenDim = inFeat // factor

    bidirectional = hp["lstm_bidirectional"]
    useAttention = hp["lstm_useAttention"]

    model = LSTMClassifier(
        vocabSize, 
        embeddingDim,
        hiddenDim,
        rnnLayers = 1,
        bidirectional = bidirectional,
        dropoutRate = 0.0,
        denseUnits = denseUnits,
        numLabels = numLabels,
        padIndex = pad,
        useAttention = useAttention
    )
    model.load_state_dict(sd, strict = False)
    model.eval()

    x = encodePadFunction(texts, w2i, maxseqlen)
    xTensor = torch.from_numpy(x)

    with torch.no_grad():
        logits = model(xTensor)
        probs = torch.sigmoid(logits)
    
    preds = (probs >= 0.5).int()
    return preds.tolist()  


def predictGRU(csv_path: str):
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).apply(cleanerFunction).tolist()

    ckpt = torch.load("savedModel/rnnGRU.pth", map_location = "cpu", weights_only = False)
    sd = ckpt["model_state_dict"]
    w2i = ckpt["wordtoindex"]
    pad = ckpt["padIndex"]
    maxseqlen = ckpt["max_sequence_length"]
    hp = ckpt["hyperparameters"]

    vocabSize, embeddingDim = sd["embedding.weight"].shape
    numLabels = sd["output_layer.weight"].shape[0]
    denseUnits, inFeat = sd["fc1.weight"].shape

    bidirectional = hp["gru_bidirectional"]
    useAttention  = hp["gru_attn"]
    rnnLayers = hp["gru_layers"]

    factor = 2 if bidirectional else 1
    hiddenDim = inFeat // factor

    model = GRUClassifier(
        vocabSize, 
        embeddingDim,
        hiddenDim,
        rnnLayers = rnnLayers,
        bidirectional = bidirectional,
        dropoutRate = 0.0,
        denseUnits = denseUnits,
        numLabels = numLabels,
        padIndex = pad,
        useAttention = useAttention
    )
    model.load_state_dict(sd)
    model.eval()

    x = encodePadFunction(texts, w2i, maxseqlen)
    xTensor = torch.from_numpy(x)

    with torch.no_grad():
        logits = model(xTensor)
        probs = torch.sigmoid(logits)

    preds = (probs >= 0.5).int()
    return preds.tolist()   


def evaluate(csvPath: str, pred, predname):
    df = pd.read_csv(csvPath)
    label_cols = ["anger", "fear", "joy", "sadness", "surprise"]
    yTrue = df[label_cols].values.astype(int)

    yPred = np.array(pred(csvPath), dtype = int)

    hammingAccuracy = (yPred == yTrue).mean()

    exactMatch = np.all(yPred == yTrue, axis = 1).mean()

    labelAccuracy = (yPred == yTrue).mean(axis = 0)

    print(f"Metrics from {predname}")

    print(f"Hamming Accuracy : {hammingAccuracy:.4f}")
    print(f"Exact‐Match Ratio : {exactMatch:.4f}")
    for label, acc in zip(label_cols, labelAccuracy):
        print(f"{label:10s} Accuracy : {acc:.4f}")


def evaluate_rfc(csvPath: str, log_level: str):
    rfcModel = joblib.load("savedModel/rfcmodel.joblib")
    df = pd.read_csv(csvPath)
    
    if(log_level == "emotions"): 
        title = "Emotion: {labelColumns[i]}"
    elif(log_level == "macro"):
        title = "Macro Average:"
    else:
        raise Exception("Invalid log level")

    x_texts = df['text']
    vectorizer = CountVectorizer()
    x_texts_vec = vectorizer.fit_transform(x_texts)
    
    y_actual = df[labelColumns].values.astype(int)

    x_train_text, x_test_text, y_train_labels, y_test_labels = train_test_split(x_texts_vec, y_actual, test_size = 0.1)

    rfcModel.fit(x_train_text, y_train_labels)
    y_pred = rfcModel.predict(x_test_text)

    confusion_mat = multilabel_confusion_matrix(y_test_labels, y_pred)

    rfc_metric = metrics_rfc(confusion_mat, log_level, title)
    rfc_metric.present_data()


testfile = os.path.join("Data", "track-a-test-large.csv")


evaluate_rfc(testfile, "emotions")


evaluate(testfile, predictFNN, "FNN")


evaluate(testfile, predictGRU, "GRU")


evaluate(testfile, predictLSTM, "LSTM")


