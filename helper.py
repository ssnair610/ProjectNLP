
import os
import re
import pandas as pd
import numpy as np
from collections import Counter

import spacy
from nltk.corpus import stopwords

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.exceptions import InconsistentVersionWarning # type: ignore

import joblib
import warnings
import plotly.graph_objects as go


dataPath = os.path.join("Data", "track-a.csv")


dataFrame = pd.read_csv(dataPath)


nlpModel = spacy.load("en_core_web_sm", disable=["parser","ner"])


stop_words = set(stopwords.words('english'))


def cleanerFunction(text):
    tempDoc = nlpModel(text)
    token = [
        tok.lemma_.lower()
        for tok in tempDoc
        if not tok.is_stop and not tok.is_punct and tok.lemma_ != "-PRON-"
    ]
    return " ".join(token)


def clean_text_svm(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text.strip()


dataFrame["Spacy_text"] = dataFrame["text"].astype(str).apply(cleanerFunction)


label_columns = ["anger", "fear", "joy", "sadness", "surprise"]
labelColumns = [col for col in label_columns if col in dataFrame]
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


class metric_calc():

    def __init__(self, confusion_mat, log_level, title) -> None:
        self.confusion_mat = confusion_mat
        self.log_level = log_level
        self.title = title
        self.accuracy_list = []

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
        print(f"accuracy: {round(accuracy,4)}")
        print(f"precision: {round(precision,4)}")
        print(f"recall: {round(recall,4)}")
        print(f"f1 Score: {round(f1_score,4)}")
        print("=======\n")
        self.accuracy_list.append(accuracy)


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


class MetricCalc:
    def __init__(self, confusion_mat, log_level = "macro", title = "Accuracy", print_flag = False) -> None:
        self.confusion_mat = confusion_mat
        self.log_level = log_level
        self.title = title
        self.accuracy_list = []
        self.overall_accuracy = 0
        self.print_flag = print_flag

    @staticmethod
    def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
        denom = tp + tn + fp + fn
        return 0.0 if denom == 0 else (tp + tn) / denom

    def report(self) -> None:
        n_labels = len(self.confusion_mat)
        if n_labels == 0:
            print("No confusion matrices supplied.")
            return

        for cm in self.confusion_mat:
            tp, fp = cm[0]
            fn, tn = cm[1]
            self.accuracy_list.append(self.accuracy(tp, tn, fp, fn))

        macro_acc = sum(self.accuracy_list) / n_labels
        self.overall_accuracy = macro_acc

        if(self.print_flag == True):
            print(f"Hamming Accuracy : {macro_acc:.4f}")

            if self.log_level == "emotions":
                for idx, acc in enumerate(self.accuracy_list):  
                    print(f"{label_columns[idx].ljust(12)} Accuracy : {acc:.4f}")



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


def evaluate_nn(csvPath: str, pred, predname, print_flag = False):
    df = pd.read_csv(csvPath)
    label_cols = ["anger", "fear", "joy", "sadness", "surprise"]
    acc_list = []
    yTrue = df[label_cols].values.astype(int)

    y_preds = np.array(pred(csvPath), dtype = int)

    hammingAccuracy = (y_preds == yTrue).mean()

    labelAccuracy = (y_preds == yTrue).mean(axis = 0)

    for label, acc in zip(label_cols, labelAccuracy):
        acc_list.append(acc)

    if (print_flag == True):
        print(f"\nMetrics from {predname}")

        print(f"Hamming Accuracy : {hammingAccuracy:.4f}")
        for label, acc in zip(label_cols, labelAccuracy):
            print(f"{label:10s} Accuracy : {acc:.4f}")

    return y_preds, acc_list, hammingAccuracy


def evaluate_rfc(csvPath: str, log_level: str, print_flag = False):
    rfcModel = joblib.load("savedModel/rfcmodel.joblib")
    df = pd.read_csv(csvPath)
    
    if(log_level == "emotions"): 
        title = "Emotion: "
    elif(log_level == "macro"):
        title = "Macro Average:"
    else:
        raise Exception("Invalid log level")

    x_texts = df['text']
    vectorizer = CountVectorizer()
    x_texts_vec = vectorizer.fit_transform(x_texts)
    
    y_actual = df[label_columns].values.astype(int)

    x_train_text, x_test_text, y_train_labels, y_test_labels = train_test_split(x_texts_vec, y_actual, test_size = 0.1)

    rfcModel.fit(x_train_text, y_train_labels)
    y_preds = rfcModel.predict(x_test_text)

    confusion_mat = multilabel_confusion_matrix(y_test_labels, y_preds)

    rfc_metric = MetricCalc(confusion_mat, log_level, title, print_flag)

    if(print_flag == True):
        print("\nMetrics from Random Forest Classifier:")
    rfc_metric.report()

    return y_preds, rfc_metric


def evaluate_svm(csvPath: str, log_level: str, print_flag = False):
    clf_path = os.path.join("savedModel", "svm_model.joblib")
    vec_path = os.path.join("savedModel", "vectorizer.joblib")
    clf = joblib.load(clf_path)
    vectorizer = joblib.load(vec_path)
  
    df = pd.read_csv(csvPath)
    if 'text' not in df.columns:
        raise ValueError("Input CSV must have a 'text' column.")
    
    if(log_level == "emotions"): 
        title = "Emotion: "
    elif(log_level == "macro"):
        title = "Macro Average:"
    else:
        raise Exception("Invalid log level")

    df['clean_text'] = df['text'].fillna('').apply(clean_text_svm)

    x_test = vectorizer.transform(df['clean_text'])
    y_true = df[label_columns].astype(int)

    y_preds = clf.predict(x_test)
    y_preds = y_preds.astype(int).tolist()

    confusion_mat = multilabel_confusion_matrix(y_true, y_preds)
    
    if(print_flag == True):
        print("\nMetrics from Support Vector Machine:")
    svm_metric = MetricCalc(confusion_mat, log_level, title, print_flag)
    svm_metric.report()

    return y_preds, svm_metric


def evaluate_nb(csvPath: str, log_level: str, print_flag = False):
    nb_path = os.path.join("savedModel", "nbmodel.joblib")
    nb_model = joblib.load(nb_path)

    df = pd.read_csv(csvPath)

    if(log_level == "emotions"): 
        title = "Emotion: "
    elif(log_level == "macro"):
        title = "Macro Average:"
    else:
        raise Exception("Invalid log level")

    x_texts = df['text']
    y_actual = df[label_columns].values.astype(int)

    x_train_text, x_test_text, y_train_labels, y_test_labels = train_test_split(x_texts, y_actual, test_size = 0.1)

    y_preds = nb_model.predict(x_test_text)
    
    confusion_mat = multilabel_confusion_matrix(y_test_labels, y_preds)

    if(print_flag == True):
        print("\nMetrics from Naive Bayes Classifier:")
    nb_metric = MetricCalc(confusion_mat, log_level, title, print_flag)
    nb_metric.report()

    return y_preds, nb_metric



warnings.filterwarnings("ignore", category = InconsistentVersionWarning)

def label_plotter(fnn_acc, gru_acc, lstm_acc, rfc_acc, svm_acc, nb_acc, save_flag = False):
    model_names = ["Neural Network: Feed Forward","Neural Network: LSTM","Neural Network: GRU", "Random Forest Classifier", "Support Vector Machine", "Naive Bayes Classifier"]
    label_names = ["anger","fear","joy","sadness","surprise"]

    z_values = np.stack([fnn_acc, lstm_acc, gru_acc, rfc_acc, svm_acc, nb_acc], axis = 1)

    x_values = np.arange(len(model_names))
    y_values = np.arange(len(label_names))

    fig = go.Figure(
        data = go.Surface(
            x = x_values,
            y = y_values,
            z = z_values,
            colorscale = "agsunset"
        )
    )

    fig.update_layout(
        title = "Model vs Label Metric Surface",
        scene = dict(
            xaxis = dict(
                tickmode = "array",
                tickvals = x_values,
                ticktext = model_names,
                title = "Model"
            ),
            yaxis = dict(
                tickmode = "array",
                tickvals = y_values,
                ticktext = label_names,
                title = "Label"
            ),
            zaxis = dict(title = "Accuracy")
        ),
        width = 1200,
        height = 1200,
        margin = dict(l = 50, r = 50, b = 50, t = 50),
        scene_camera = dict(eye = dict(x = 1.7, y = 1.7, z = 1.7))
    )

    if save_flag == True:
        fig.write_image(file = "accuracy_plot.png", height = 1500, width = 1500, scale = 1)
        print(f"Plot saved! Do check 'accuracy_plot.png'! ^_^")
    else:
        fig.show()


def predict_plotter(testfile):
    pred_rfc, rfc_val = evaluate_rfc(testfile, "emotions")
    pred_fnn, fnn_val, fnn_total_acc = evaluate_nn(testfile, predictFNN, "FNN")
    pred_gru, gru_val, gru_total_acc = evaluate_nn(testfile, predictGRU, "GRU")
    pred_lstm, lstm_val, lstm_total_acc = evaluate_nn(testfile, predictLSTM, "LSTM")
    pred_svm, svm_val = evaluate_svm(testfile, "emotions")
    pred_nb, nb_val = evaluate_nb(testfile, "emotions")

    return rfc_val, fnn_val, gru_val, lstm_val, svm_val, nb_val


def predict(testfile, save_flag):
    pred_rfc, rfc_val = evaluate_rfc(testfile, "emotions")
    pred_fnn, fnn_val, fnn_total_acc = evaluate_nn(testfile, predictFNN, "FNN")
    pred_gru, gru_val, gru_total_acc = evaluate_nn(testfile, predictGRU, "GRU")
    pred_lstm, lstm_val, lstm_total_acc = evaluate_nn(testfile, predictLSTM, "LSTM")
    pred_svm, svm_val = evaluate_svm(testfile, "emotions")
    pred_nb, nb_val = evaluate_nb(testfile, "emotions")

    overall_accuracies = [rfc_val.overall_accuracy, fnn_total_acc, gru_total_acc, lstm_total_acc, svm_val.overall_accuracy, nb_val.overall_accuracy]
    predictions = [pred_rfc, pred_fnn, pred_gru, pred_lstm, pred_svm, pred_nb]

    max_index = overall_accuracies.index(max(overall_accuracies))
    best_pred = predictions[max_index]
    model_names = ["Random Forest Classifier", "Neural Network - Feed Forward", "Neural Network - GRU", "Neural Network - LSTM", "Support Vector Machine", "Naive Bayes Classifier"]

    return best_pred, overall_accuracies[max_index], model_names[max_index]



