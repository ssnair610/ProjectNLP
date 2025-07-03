# %%
import os
import re
import pandas as pd
import numpy as np
from collections import Counter

import spacy
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import precision_recall_curve

import optuna

# %%
dataPath = os.path.join("data", "track-a.csv")

# %%
dataFrame = pd.read_csv(dataPath)

# %%
nlpModel = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# %%
textColumn = "text"
labelColumns = ["anger", "fear", "joy", "sadness", "surprise"]


# %%
def cleanerFunction(text: str) -> str:
    tempDoc = nlpModel(text)
    tokens = [
        tok.lemma_.lower()
        for tok in tempDoc
        if not tok.is_stop and not tok.is_punct and tok.lemma_ != "-PRON-"
    ]
    return " ".join(tokens)


# %%
dataFrame["Spacy_text"] = dataFrame["text"].astype(str).apply(cleanerFunction)

# %%
userDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
labelColumns = [
    col for col in ["anger", "fear", "joy", "sadness", "surprise"] if col in dataFrame
]
yData = dataFrame[labelColumns].values.astype(np.float32)

# %%
xAll = dataFrame["Spacy_text"].tolist()
xTrain, xVal, yTrain, yVal = train_test_split(
    xAll, yData, test_size=0.1, random_state=69
)

# %%
xTrain = xTrain.copy()
xVal = xVal.copy()

# %%
numLabels = yTrain.shape[1]

# %%
TOKEN_PATTERN = re.compile(r"\b\w+\b")


# %%
def simpleTokenize(text: str) -> list:
    return TOKEN_PATTERN.findall(text.lower())


# %%
def vocabularyBuilder(texts: list, vocabsize: int):
    counter = Counter()
    for t in texts:
        tokens = simpleTokenize(t)
        counter.update(tokens)

    mostCommon = counter.most_common(vocabsize - 2)
    indextoword = ["<pad>", "<unk>"] + [token for token, _ in mostCommon]
    wordtoindex = {w: i for i, w in enumerate(indextoword)}

    return wordtoindex, indextoword


# %%
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100

wordtoindex, indextoword = vocabularyBuilder(xTrain, vocabsize=MAX_VOCAB_SIZE)
vocabSize = len(indextoword)

padIndex = wordtoindex["<pad>"]
unkIndex = wordtoindex["<unk>"]


# %%
def encodePadFunction(
    texts: list, wordtoindex: dict, sequenceLength: int = MAX_SEQUENCE_LENGTH
) -> np.ndarray:
    encodings = []

    for t in texts:
        tokens = simpleTokenize(t)
        tokenIDs = [wordtoindex.get(tok, unkIndex) for tok in tokens]
        if len(tokenIDs) > sequenceLength:
            tokenIDs = tokenIDs[:sequenceLength]
        else:
            tokenIDs = tokenIDs + [padIndex] * (sequenceLength - len(tokenIDs))
        encodings.append(tokenIDs)

    return np.array(encodings, dtype=np.int64)


# %%
trainEncode = encodePadFunction(xTrain, wordtoindex, sequenceLength=MAX_SEQUENCE_LENGTH)
valEncode = encodePadFunction(xVal, wordtoindex, sequenceLength=MAX_SEQUENCE_LENGTH)


# %%
class TextDataset(Dataset):
    def __init__(self, encodings: np.ndarray, labels: np.ndarray) -> None:
        self.encodings = torch.from_numpy(encodings)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return self.encodings.size(0)

    def __getitem__(self, index):
        return self.encodings[index], self.labels[index]


# %% [markdown]
# ## Recurrent Neural Network - LSTM


# %%
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


# %%
def epochTrain(model, dataloader, optimizer, criterion, device):
    model.train()
    totalLoss = 0.0
    for batchInputs, batchLabels in dataloader:
        batchInputs = batchInputs.to(device)
        batchLabels = batchLabels.to(device)

        optimizer.zero_grad()
        logits = model(batchInputs)
        loss = criterion(logits, batchLabels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totalLoss += loss.item() * batchInputs.size(0)

    return totalLoss / len(dataloader.dataset)


# %%
def evaluateModel(model, dataloader, criterion, device):
    model.eval()
    totalLoss = 0.0
    with torch.no_grad():
        for batchInputs, batchLabels in dataloader:
            batchInputs = batchInputs.to(device)
            batchLabels = batchLabels.to(device)

            logits = model(batchInputs)
            loss = criterion(logits, batchLabels)
            totalLoss += loss.item() * batchInputs.size(0)

    return totalLoss / len(dataloader.dataset)


# %%
freq = np.maximum(yTrain.sum(axis=0) / len(yTrain), 1e-4)
classWeights = 1.0 / freq
classWeights = classWeights / classWeights.sum()
weightTensor = torch.FloatTensor(classWeights).to(userDevice)


# %%
def objectiveLSTM(trial):
    embeddingDim = trial.suggest_categorical("lstm_embeddingDim", [64, 128, 256])
    hiddenDim = trial.suggest_int("lstm_hiddenDim", 32, 256, step=16)
    rnnLayers = trial.suggest_int("lstm_rnnLayers", 1, 5)
    bidirectional = trial.suggest_categorical("lstm_bidirectional", [False, True])
    dropoutRate = trial.suggest_float("lstm_dropoutRate", 0.2, 0.6, step=0.1)
    denseUnits = trial.suggest_int("lstm_denseUnits", 32, 256, step=16)
    learnRate = trial.suggest_float("lstm_learnRate", 1e-4, 1e-2, log=True)
    weightDecay = trial.suggest_float("lstm_weightDecay", 1e-6, 1e-2, log=True)
    batchSize = trial.suggest_categorical("lstm_batchSize", [32, 64, 128, 256])
    epochs = trial.suggest_categorical("lstm_epochs", [3, 5, 7, 9])
    optName = trial.suggest_categorical("lstm_optimizer", ["Adam", "RMSprop", "SGD"])
    useAttention = trial.suggest_categorical("lstm_useAttention", [False, True])
    useScheduler = trial.suggest_categorical("lstm_useScheduler", [False, True])
    if optName == "SGD":
        sgdvar = trial.suggest_float("lstm_momentum", 0.1, 0.9)

    trainDataset = TextDataset(trainEncode, yTrain)
    valDataset = TextDataset(valEncode, yVal)

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True)

    model = LSTMClassifier(
        vocabSize,
        embeddingDim,
        hiddenDim,
        rnnLayers,
        bidirectional,
        dropoutRate,
        denseUnits,
        numLabels,
        padIndex,
        useAttention,
    ).to(userDevice)

    if optName == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learnRate, weight_decay=weightDecay
        )
    elif optName == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=learnRate, weight_decay=weightDecay
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=learnRate, momentum=sgdvar)  # type: ignore

    criterionOption = nn.BCEWithLogitsLoss(pos_weight=weightTensor)

    bestvalLoss = float("inf")
    counterVar = 0
    bestState = None

    if useScheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1
        )

    for epoch in range(1, epochs + 1):
        trainLoss = epochTrain(
            model, trainLoader, optimizer, criterionOption, userDevice
        )
        valLoss = evaluateModel(model, valLoader, criterionOption, userDevice)

        if useScheduler:
            scheduler.step(valLoss)  # type: ignore

        trial.report(valLoss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if valLoss < bestvalLoss:
            bestvalLoss = valLoss
            counterVar = 0
            bestState = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            counterVar += 1
            if counterVar >= 2:
                break

    model.load_state_dict(bestState)  # type: ignore
    return bestvalLoss


# %%
def _objectiveLSTM(trial):
    return objectiveLSTM(trial)


# %%
# optuna.delete_study(study_name = "studyLSTM", storage = "sqlite:///db.sqlite3")

# %%
studyLSTM = optuna.create_study(
    study_name="studyLSTM",
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    storage="sqlite:///db.sqlite3",
)

# %%
studyLSTM.optimize(_objectiveLSTM, n_trials=500)

# %%
print("\nOptuna Best Trial:")
best = studyLSTM.best_trial
print(f"Validation Loss: {best.value:.4f}")
for key, val in best.params.items():
    print(f"  {key}: {val}")
print("\n")

best_embeddingDim = best.params["lstm_embeddingDim"]
best_hiddenDim = best.params["lstm_hiddenDim"]
best_rnnLayers = best.params["lstm_rnnLayers"]
best_bidirectional = best.params["lstm_bidirectional"]
best_dropoutRate = best.params["lstm_dropoutRate"]
best_denseUnits = best.params["lstm_denseUnits"]
best_learnRate = best.params["lstm_learnRate"]
best_weightDecay = best.params["lstm_weightDecay"]
best_batchSize = best.params["lstm_batchSize"]
best_epochs = best.params["lstm_epochs"]
best_optName = best.params["lstm_optimizer"]
best_use_scheduler = best.params["lstm_useScheduler"]
best_useAttention = best.params["lstm_useAttention"]
if best_optName == "SGD":
    best_momentum = best.params["lstm_momentum"]

# %%
finalTrainDataset = TextDataset(trainEncode, yTrain)
finalValDataset = TextDataset(valEncode, yVal)

finalTrainLoader = DataLoader(
    finalTrainDataset, batch_size=best_batchSize, shuffle=True
)
finalValLoader = DataLoader(finalValDataset, batch_size=best_batchSize, shuffle=False)

# %%
finalModelLSTM = LSTMClassifier(
    vocabSize=vocabSize,
    embeddingDim=best_embeddingDim,
    hiddenDim=best_hiddenDim,
    rnnLayers=best_rnnLayers,
    bidirectional=best_bidirectional,
    dropoutRate=best_dropoutRate,
    denseUnits=best_denseUnits,
    numLabels=numLabels,
    padIndex=padIndex,
    useAttention=best_useAttention,
).to(userDevice)

if best_optName == "Adam":
    finalOptimizer = optim.Adam(
        finalModelLSTM.parameters(), lr=best_learnRate, weight_decay=best_weightDecay
    )
elif best_optName == "RMSprop":
    finalOptimizer = optim.RMSprop(
        finalModelLSTM.parameters(), lr=best_learnRate, weight_decay=best_weightDecay
    )
else:
    finalOptimizer = optim.SGD(
        finalModelLSTM.parameters(),
        lr=best_learnRate,
        momentum=best_momentum,
        weight_decay=best_weightDecay,
    )

finalCriterion = nn.BCEWithLogitsLoss(pos_weight=weightTensor)

if best_use_scheduler:
    finalScheduler = optim.lr_scheduler.ReduceLROnPlateau(
        finalOptimizer, mode="min", factor=0.5, patience=1
    )

# %%
bestValLossFinal = float("inf")
patienceCtr = 0
bestStateFinal = None

# %%
for epoch in range(1, best_epochs + 1):
    trainLoss = epochTrain(
        finalModelLSTM, finalTrainLoader, finalOptimizer, finalCriterion, userDevice
    )
    valLoss = evaluateModel(finalModelLSTM, finalValLoader, finalCriterion, userDevice)
    print(
        f"Final Epoch {epoch:02d} | Train Loss: {trainLoss:.4f} | Val Loss: {valLoss:.4f}"
    )

    if best_use_scheduler:
        finalScheduler.step(valLoss)

    if valLoss < bestValLossFinal:
        bestValLossFinal = valLoss
        patienceCtr = 0
        bestStateFinal = {k: v.cpu() for k, v in finalModelLSTM.state_dict().items()}
    else:
        patienceCtr += 1
        if patienceCtr >= 3:
            print(f"Early stopping at epoch {epoch}")
            break

# %%
finalModelLSTM.load_state_dict(bestStateFinal)

# %%
finalModelLSTM.eval()
runningLoss = 0.0
correct = 0
total = 0

allProbs = []
allTrue = []

with torch.no_grad():
    for batchInputs, batchLabels in finalValLoader:
        batchInputs = batchInputs.to(userDevice)
        batchLabels = batchLabels.to(userDevice)

        logits = finalModelLSTM(batchInputs)
        loss = finalCriterion(logits, batchLabels)
        runningLoss += loss.item() * batchInputs.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        allProbs.append(probs)
        allTrue.append(batchLabels.numpy())

        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == batchLabels).sum().item()
        total += batchLabels.numel()

allProbs = np.vstack(allProbs)
allTrue = np.vstack(allTrue)

thresholdsLSTM = []
for i in range(numLabels):
    prec, rec, thr = precision_recall_curve(allTrue[:, i], allProbs[:, i])
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    bestIndex = np.nanargmax(f1[:-1])
    thresholdsLSTM.append(thr[bestIndex])

finalValLoss = runningLoss / len(finalValDataset)
finalValAcc = correct / total
print(f"\nFinal Model -> Val Loss: {finalValLoss:.4f}, Val Accuracy: {finalValAcc:.4f}")
print("LSTM per-label thresholds:", thresholdsLSTM)

# %%
os.makedirs("saved-model", exist_ok=True)
torch.save(
    {
        "model_state_dict": finalModelLSTM.state_dict(),
        "hyperparameters": best.params,
        "wordtoindex": wordtoindex,
        "indextoword": indextoword,
        "padIndex": padIndex,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "thresholds": thresholdsLSTM,
    },
    "saved-model/rnnLSTM.pth",
)
print("Saved final model + vocab to saved-model/rnnLSTM.pth")

# %% [markdown]
# ## Recurrent Neural Network - GRU
#


# %%
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


# %%
def objectiveGRU(trial):
    embeddingDim = trial.suggest_categorical("gru_embeddingDim", [64, 128, 256, 512])
    hiddenDim = trial.suggest_int("gru_hiddenDim", 32, 256, step=16)
    rnnLayers = trial.suggest_int("gru_layers", 1, 9)
    bidirectional = trial.suggest_categorical("gru_bidirectional", [False, True])
    dropoutRate = trial.suggest_float("gru_dropoutRate", 0.2, 0.6, step=0.1)
    denseUnits = trial.suggest_int("gru_denseUnits", 32, 256, step=16)
    learnRate = trial.suggest_float("gru_learnRate", 1e-4, 1e-2, log=True)
    weightDecay = trial.suggest_float("gru_weightDecay", 1e-6, 1e-2, log=True)
    batchSize = trial.suggest_categorical("gru_batchSize", [32, 64, 128, 256])
    epochs = trial.suggest_categorical("gru_epochs", [3, 5, 7, 9])
    optimizerName = trial.suggest_categorical(
        "gru_optimizer", ["AdamW", "Adam", "RMSprop", "SGD"]
    )
    useScheduler = trial.suggest_categorical("gru_useScheduler", [False, True])
    useAttention = trial.suggest_categorical("gru_attn", [False, True])
    if optimizerName == "SGD":
        momentum = trial.suggest_float("gru_momentum", 0.1, 0.9)

    trainDataset = TextDataset(trainEncode, yTrain)
    valDataset = TextDataset(valEncode, yVal)

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True)

    model = GRUClassifier(
        vocabSize,
        embeddingDim,
        hiddenDim,
        rnnLayers,
        bidirectional,
        dropoutRate,
        denseUnits,
        numLabels,
        padIndex,
        useAttention,
    ).to(userDevice)

    if optimizerName == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=learnRate, weight_decay=weightDecay
        )
    elif optimizerName == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learnRate, weight_decay=weightDecay
        )
    elif optimizerName == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=learnRate, weight_decay=weightDecay
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=learnRate, momentum=momentum, weight_decay=weightDecay)  # type: ignore

    criterionOption = nn.BCEWithLogitsLoss(pos_weight=weightTensor)

    if useScheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1
        )

    bestValLoss = float("inf")
    counterVar = 0
    bestState = None

    for epoch in range(1, epochs + 1):
        trainLoss = epochTrain(
            model, trainLoader, optimizer, criterionOption, userDevice
        )
        valLoss = evaluateModel(model, valLoader, criterionOption, userDevice)

        if useScheduler:
            scheduler.step(valLoss)  # type: ignore

        trial.report(valLoss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if valLoss < bestValLoss:  # type: ignore
            bestValLoss = valLoss
            counterVar = 0
            bestState = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            counterVar += 1
            if counterVar >= 2:
                break

    model.load_state_dict(bestState)  # type: ignore
    return bestValLoss


# %%
def _objectiveGRU(trial):
    return objectiveGRU(trial)


# %%
optuna.delete_study(study_name="studyGRU", storage="sqlite:///db.sqlite3")

# %%
studyGRU = optuna.create_study(
    study_name="studyGRU",
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    storage="sqlite:///db.sqlite3",
)

# %%
studyGRU.optimize(_objectiveGRU, n_trials=500)

# %%
print("\nOptuna Best Trial:")
best = studyGRU.best_trial
print(f"Validation Loss: {best.value:.4f}")
for key, val in best.params.items():
    print(f"  {key}: {val}")
print("\n")

best_embeddingDim = best.params["gru_embeddingDim"]
best_hiddenDim = best.params["gru_hiddenDim"]
best_rnnLayers = best.params["gru_layers"]
best_bidirectional = best.params["gru_bidirectional"]
best_dropoutRate = best.params["gru_dropoutRate"]
best_denseUnits = best.params["gru_denseUnits"]
best_learnRate = best.params["gru_learnRate"]
best_weightDecay = best.params["gru_weightDecay"]
best_batchSize = best.params["gru_batchSize"]
best_epochs = best.params["gru_epochs"]
best_optName = best.params["gru_optimizer"]
best_use_scheduler = best.params["gru_useScheduler"]
best_useAttention = best.params["gru_attn"]
if best_optName == "SGD":
    best_momentum = best.params["gru_momentum"]

# %%
finalTrainDataset = TextDataset(trainEncode, yTrain)
finalValDataset = TextDataset(valEncode, yVal)

finalTrainLoader = DataLoader(
    finalTrainDataset, batch_size=best_batchSize, shuffle=True
)
finalValLoader = DataLoader(finalValDataset, batch_size=best_batchSize, shuffle=False)

# %%
finalModelGRU = GRUClassifier(
    vocabSize,
    best_embeddingDim,
    best_hiddenDim,
    best_rnnLayers,
    best_bidirectional,
    best_dropoutRate,
    best_denseUnits,
    numLabels,
    padIndex,
    best_useAttention,
).to(userDevice)

if best_optName == "AdamW":
    finalOptimizer = optim.AdamW(
        finalModelGRU.parameters(), lr=best_learnRate, weight_decay=best_weightDecay
    )
elif best_optName == "Adam":
    finalOptimizer = optim.Adam(
        finalModelGRU.parameters(), lr=best_learnRate, weight_decay=best_weightDecay
    )
elif best_optName == "RMSprop":
    finalOptimizer = optim.RMSprop(
        finalModelGRU.parameters(), lr=best_learnRate, weight_decay=best_weightDecay
    )
else:
    finalOptimizer = optim.SGD(finalModelGRU.parameters(), lr=best_learnRate, momentum=best_momentum, weight_decay=best_weightDecay)  # type: ignore

finalCriterion = nn.BCEWithLogitsLoss(pos_weight=weightTensor)

if best_use_scheduler:
    finalScheduler = optim.lr_scheduler.ReduceLROnPlateau(
        finalOptimizer, mode="min", factor=0.5, patience=1
    )

# %%
bestValLossFinal = float("inf")
patienceCtr = 0
bestStateFinal = None

# %%
for epoch in range(1, best_epochs + 1):
    trainLoss = epochTrain(
        finalModelGRU, finalTrainLoader, finalOptimizer, finalCriterion, userDevice
    )
    valLoss = evaluateModel(finalModelGRU, finalValLoader, finalCriterion, userDevice)
    print(
        f"Final Epoch {epoch:02d} | Train Loss: {trainLoss:.4f} | Val Loss: {valLoss:.4f}"
    )

    if best_use_scheduler:
        finalScheduler.step(valLoss)

    if valLoss < bestValLossFinal:
        bestValLossFinal = valLoss
        patienceCtr = 0
        bestStateFinal = {k: v.cpu() for k, v in finalModelGRU.state_dict().items()}
    else:
        patienceCtr += 1
        if patienceCtr >= 3:
            print(f"Early stopping at epoch {epoch}")
            break

# %%
finalModelGRU.load_state_dict(bestStateFinal)

# %%
finalModelGRU.eval()
runningLoss = 0.0
correct = 0
total = 0

allProbs = []
allTrue = []

with torch.no_grad():
    for batchInputs, batchLabels in finalValLoader:
        batchInputs = batchInputs.to(userDevice)
        batchLabels = batchLabels.to(userDevice)

        logits = finalModelGRU(batchInputs)
        loss = finalCriterion(logits, batchLabels)
        runningLoss += loss.item() * batchInputs.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        allProbs.append(probs)
        allTrue.append(batchLabels.numpy())

        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == batchLabels).sum().item()
        total += batchLabels.numel()

allProbs = np.vstack(allProbs)
allTrue = np.vstack(allTrue)

thresholdsGRU = []
for i in range(numLabels):
    prec, rec, thr = precision_recall_curve(allTrue[:, i], allProbs[:, i])
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    bestIndex = np.nanargmax(f1[:-1])
    thresholdsGRU.append(thr[bestIndex])

finalValLoss = runningLoss / len(finalValDataset)
finalValAcc = correct / total
print(f"\nFinal Model -> Val Loss: {finalValLoss:.4f}, Val Accuracy: {finalValAcc:.4f}")
print("GRU per-label thresholds:", thresholdsGRU)

# %%
os.makedirs("saved-model", exist_ok=True)
torch.save(
    {
        "model_state_dict": finalModelGRU.state_dict(),
        "hyperparameters": best.params,
        "wordtoindex": wordtoindex,
        "indextoword": indextoword,
        "padIndex": padIndex,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "thresholds": thresholdsGRU,
    },
    "saved-model/rnnGRU.pth",
)
print("Saved final model + vocab to saved-model/rnnGRU.pth")
