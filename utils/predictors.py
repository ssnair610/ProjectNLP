import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor

from utils.cleaner import text_cleaner, encode_pad_function
from utils.evaluation_data import EvaluationData


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
            in_channels=embeddingDim,
            out_channels=embeddingDim,
            kernel_size=5,
            padding=2,
        )
        self.relu_cnn = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=padIndex)
        self.dropout_emb = nn.Dropout(dropoutRate)
        self.lstm = nn.LSTM(
            input_size=embeddingDim,
            hidden_size=hiddenDim,
            num_layers=rnnLayers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropoutRate if rnnLayers > 1 else 0.0,
        )

        self.use_attention = useAttention
        directionFactor = 2 if bidirectional else 1
        if useAttention:
            self.attn_linear = nn.Linear(
                hiddenDim * directionFactor, hiddenDim * directionFactor
            )
            self.attn_v = nn.Linear(hiddenDim * directionFactor, 1, bias=False)

        self.fc1 = nn.Linear(hiddenDim * directionFactor, denseUnits)
        self.bn1 = nn.BatchNorm1d(denseUnits)
        self.relu = nn.ReLU()
        self.dropoutFc = nn.Dropout(dropoutRate)
        self.output_layer = nn.Linear(denseUnits, numLabels)

    def forward(self, x: torch.LongTensor) -> Tensor:
        pad_idx = self.embedding.padding_idx
        lengths = (x != pad_idx).sum(dim=1)
        lengths = torch.clamp(lengths, min=1)

        embTensor: Tensor = self.embedding(x)  # type: ignore
        embTensor = self.dropout_emb(embTensor)  # type: ignore

        c_in = embTensor.transpose(1, 2)
        c_out = self.relu_cnn(self.conv1d(c_in))
        c_out = self.pool(c_out)

        rnn_in = c_out.transpose(1, 2)
        lengths = torch.clamp(lengths // 2, min=1)

        packed = pack_padded_sequence(
            rnn_in, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        rnnOut, _ = self.lstm(packed)
        rnnOut, _ = pad_packed_sequence(rnnOut, batch_first=True)

        if self.use_attention:
            scores = torch.tanh(self.attn_linear(rnnOut))
            weights = torch.softmax(self.attn_v(scores), dim=1)
            finalFeat = (weights * rnnOut).sum(dim=1)
        else:
            idx = torch.arange(x.size(0), device=x.device)
            finalFeat = rnnOut[idx, lengths - 1]

        h: Tensor = self.fc1(finalFeat)  # type: ignore
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropoutFc(h)  # type: ignore
        logits: Tensor = self.output_layer(h)  # type: ignore
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
        useAttention: bool = False,
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=embeddingDim,
            out_channels=embeddingDim,
            kernel_size=5,
            padding=2,
        )
        self.relu_cnn = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.embedding: nn.Embedding = nn.Embedding(
            vocabSize, embeddingDim, padding_idx=padIndex
        )
        self.dropout_emb: nn.Dropout = nn.Dropout(dropoutRate)
        self.gru: nn.GRU = nn.GRU(
            input_size=embeddingDim,
            hidden_size=hiddenDim,
            num_layers=rnnLayers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropoutRate if rnnLayers > 1 else 0.0,
        )

        self.use_attention: bool = useAttention
        factor = 2 if bidirectional else 1
        if useAttention:
            self.attn_linear: nn.Linear = nn.Linear(
                hiddenDim * factor, hiddenDim * factor
            )
            self.attn_v: nn.Linear = nn.Linear(hiddenDim * factor, 1, bias=False)

        self.fc1: nn.Linear = nn.Linear(hiddenDim * factor, denseUnits)
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(denseUnits)
        self.relu: nn.ReLU = nn.ReLU()
        self.dropout_fc: nn.Dropout = nn.Dropout(dropoutRate)
        self.output_layer: nn.Linear = nn.Linear(denseUnits, numLabels)

    def forward(self, x: torch.LongTensor) -> Tensor:
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        lengths = torch.clamp(lengths, min=1)

        emb = self.dropout_emb(self.embedding(x))

        c_in = emb.transpose(1, 2)
        c_out = self.relu_cnn(self.conv1d(c_in))
        c_out = self.pool(c_out)

        rnn_in = c_out.transpose(1, 2)
        lengths = torch.clamp(lengths // 2, min=1)

        packed = pack_padded_sequence(
            rnn_in, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        rnnOut, _ = self.gru(packed)
        rnnOut, _ = pad_packed_sequence(rnnOut, batch_first=True)

        if self.use_attention:
            scores = torch.tanh(self.attn_linear(rnnOut))
            weights = torch.softmax(self.attn_v(scores), dim=1)
            finalFeat = (weights * rnnOut).sum(dim=1)
        else:
            idx = torch.arange(x.size(0), device=x.device)
            finalFeat = rnnOut[idx, lengths - 1]

        h: Tensor = self.fc1(finalFeat)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout_fc(h)
        logits: Tensor = self.output_layer(h)
        return logits


class feedforwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        vocabSize: int,
        embeddingDim: int,
        hiddenUnit1: int,
        hiddenUnit2: int,
        dropoutRate: float,
        numLabels: int,
        padIndex: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=padIndex)
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
        avgEmb = emb.mean(dim=1)
        h1 = self.fc1(avgEmb)
        h1 = self.bn1(h1)
        h1 = self.relu(h1)
        h1 = self.dropout2(h1)
        h2 = self.fc2(h1)
        h2 = self.bn2(h2)
        h2 = self.relu(h2)
        h2 = self.dropout3(h2)
        return self.fc_out(h2)


def predictFNN(csvFile: str):
    data = EvaluationData(csvFile)
    df = data.data_frame
    texts = df["text"].astype(str).apply(text_cleaner).tolist()

    ckpt = torch.load("saved-model/ffnn.pth", map_location="cpu")
    sd = ckpt["model_state_dict"]
    w2i = ckpt["wordtoindex"]
    padIndex = ckpt["padIndex"]
    maxseqlen = ckpt["max_sequence_length"]

    vocabSize, embeddingDim = sd["embedding.weight"].shape
    hiddenUnit1 = sd["fc1.weight"].shape[0]
    hiddenUnit2 = sd["fc2.weight"].shape[0]

    model = feedforwardNeuralNetwork(
        vocabSize=vocabSize,
        embeddingDim=embeddingDim,
        hiddenUnit1=hiddenUnit1,
        hiddenUnit2=hiddenUnit2,
        dropoutRate=0.0,
        numLabels=data.num_labels,
        padIndex=padIndex,
    )
    model.load_state_dict(sd)
    model.eval()

    x = encode_pad_function(texts, w2i, maxseqlen)
    xTensor = torch.from_numpy(x)

    with torch.no_grad():
        logits = model(xTensor)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
    return preds.tolist()


def predictLSTM(csv_path: str):
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).apply(text_cleaner).tolist()

    ckpt = torch.load("saved-model/rnnLSTM.pth", map_location="cpu", weights_only=False)
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
        rnnLayers=1,
        bidirectional=bidirectional,
        dropoutRate=0.0,
        denseUnits=denseUnits,
        numLabels=numLabels,
        padIndex=pad,
        useAttention=useAttention,
    )
    model.load_state_dict(sd, strict=False)
    model.eval()

    x = encode_pad_function(texts, w2i, maxseqlen)
    xTensor = torch.from_numpy(x)

    with torch.no_grad():
        logits = model(xTensor)
        probs = torch.sigmoid(logits)

    preds = (probs >= 0.5).int()
    return preds.tolist()


def predictGRU(csv_path: str):
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).apply(text_cleaner).tolist()

    ckpt = torch.load("saved-model/rnnGRU.pth", map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    w2i = ckpt["wordtoindex"]
    pad = ckpt["padIndex"]
    maxseqlen = ckpt["max_sequence_length"]
    hp = ckpt["hyperparameters"]

    vocabSize, embeddingDim = sd["embedding.weight"].shape
    numLabels = sd["output_layer.weight"].shape[0]
    denseUnits, inFeat = sd["fc1.weight"].shape

    bidirectional = hp["gru_bidirectional"]
    useAttention = hp["gru_attn"]
    rnnLayers = hp["gru_layers"]

    factor = 2 if bidirectional else 1
    hiddenDim = inFeat // factor

    model = GRUClassifier(
        vocabSize,
        embeddingDim,
        hiddenDim,
        rnnLayers=rnnLayers,
        bidirectional=bidirectional,
        dropoutRate=0.0,
        denseUnits=denseUnits,
        numLabels=numLabels,
        padIndex=pad,
        useAttention=useAttention,
    )
    model.load_state_dict(sd)
    model.eval()

    x = encode_pad_function(texts, w2i, maxseqlen)
    xTensor = torch.from_numpy(x)

    with torch.no_grad():
        logits = model(xTensor)
        probs = torch.sigmoid(logits)

    preds = (probs >= 0.5).int()
    return preds.tolist()
