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
# ## Feed-forward Neural Network


# %%
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
class_weights = 1.0 / freq
class_weights = class_weights / class_weights.sum()
weight_tensor = torch.FloatTensor(class_weights).to(userDevice)


# %%
def objective(trial):
    embeddingDim = trial.suggest_categorical(
        "embeddingDim",
        [
            32,
            64,
            128,
            256,
            512,
        ],
    )
    hiddenUnit1 = trial.suggest_int("hiddenUnit1", 32, 256, step=16)
    hiddenUnit2 = trial.suggest_int("hiddenUnit2", 32, 256, step=16)
    dropoutRate = trial.suggest_float("dropoutRate", 0.2, 0.6, step=0.1)
    learnRate = trial.suggest_float("learnRate", 1e-4, 1e-2, log=True)
    weightDecay = trial.suggest_float("weightDecay", 1e-6, 1e-2, log=True)
    batchSize = trial.suggest_categorical("batchSize", [32, 64, 128, 256])
    epochs = trial.suggest_categorical("epochs", [3, 5, 7, 9, 11, 13, 15])
    optName = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
    sgdvar = 0
    if optName == "SGD":
        sgdvar = trial.suggest_float("momentum", 0.1, 0.9)

    trainDataset = TextDataset(trainEncode, yTrain)
    valDataset = TextDataset(valEncode, yVal)

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True)

    model = feedforwardNeuralNetwork(
        vocabSize,
        embeddingDim,
        hiddenUnit1,
        hiddenUnit2,
        dropoutRate,
        numLabels,
        padIndex,
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
        optimizer = optim.SGD(model.parameters(), lr=learnRate, momentum=sgdvar)

    criterionOption = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    bestvalLoss = float("inf")
    counterVar = 0
    bestState = None

    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1
        )

    for epoch in range(1, epochs + 1):
        trainLoss = epochTrain(
            model, trainLoader, optimizer, criterionOption, userDevice
        )
        valLoss = evaluateModel(model, valLoader, criterionOption, userDevice)

        if use_scheduler:
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
def _objective(trial):
    return objective(trial)


# %%
study = optuna.create_study(
    study_name="FFNN",
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    storage="sqlite:///db.sqlite3",
)

# %%
study.optimize(_objective, n_trials=500)

# %%
print("\nOptuna Best Trial:")
best = study.best_trial
print(f"Validation Loss: {best.value:.4f}")
for key, val in best.params.items():
    print(f"  {key}: {val}")
print("\n")

best_embeddingDim = best.params["embeddingDim"]
best_hiddenUnit1 = best.params["hiddenUnit1"]
best_hiddenUnit2 = best.params["hiddenUnit2"]
best_dropoutRate = best.params["dropoutRate"]
best_learnRate = best.params["learnRate"]
best_weightDecay = best.params["weightDecay"]
best_batchSize = best.params["batchSize"]
best_epochs = best.params["epochs"]
best_optimizer = best.params["optimizer"]
best_use_scheduler = best.params["use_scheduler"]
best_momentum = 0
if best_optimizer == "SGD":
    best_momentum = best.params["momentum"]

# %%
finalTrainDataset = TextDataset(trainEncode, yTrain)
finalValDataset = TextDataset(valEncode, yVal)

finalTrainLoader = DataLoader(
    finalTrainDataset, batch_size=best_batchSize, shuffle=True
)
finalValLoader = DataLoader(finalValDataset, batch_size=best_batchSize, shuffle=False)

# %%
finalModel = feedforwardNeuralNetwork(
    vocabSize=vocabSize,
    embeddingDim=best_embeddingDim,
    hiddenUnit1=best_hiddenUnit1,
    hiddenUnit2=best_hiddenUnit2,
    dropoutRate=best_dropoutRate,
    numLabels=numLabels,
    padIndex=padIndex,
).to(userDevice)

if best_optimizer == "Adam":
    finalOptimizer = optim.Adam(
        finalModel.parameters(), lr=best_learnRate, weight_decay=best_weightDecay
    )
elif best_optimizer == "RMSprop":
    finalOptimizer = optim.RMSprop(
        finalModel.parameters(), lr=best_learnRate, weight_decay=best_weightDecay
    )
else:
    finalOptimizer = optim.SGD(
        finalModel.parameters(),
        lr=best_learnRate,
        momentum=best_momentum,
        weight_decay=best_weightDecay,
    )

finalCriterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

finalScheduler = None
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
        finalModel, finalTrainLoader, finalOptimizer, finalCriterion, userDevice
    )
    valLoss = evaluateModel(finalModel, finalValLoader, finalCriterion, userDevice)
    print(
        f"Final Epoch {epoch:02d} | Train Loss: {trainLoss:.4f} | Val Loss: {valLoss:.4f}"
    )

    if best_use_scheduler and finalScheduler:
        finalScheduler.step(valLoss)

    if valLoss < bestValLossFinal:
        bestValLossFinal = valLoss
        patienceCtr = 0
        bestStateFinal = {k: v.cpu() for k, v in finalModel.state_dict().items()}
    else:
        patienceCtr += 1
        if patienceCtr >= 3:
            print(f"Early stopping at epoch {epoch}")
            break

# %%
finalModel.load_state_dict(bestStateFinal)  # type: ignore

# %%
finalModel.eval()
runningLoss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for batchInputs, batchLabels in finalValLoader:
        batchInputs = batchInputs.to(userDevice)
        batchLabels = batchLabels.to(userDevice)

        logits = finalModel(batchInputs)
        loss = finalCriterion(logits, batchLabels)
        runningLoss += loss.item() * batchInputs.size(0)

        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == batchLabels).sum().item()
        total += batchLabels.numel()

finalValLoss = runningLoss / len(finalValDataset)
finalValAcc = correct / total
print(f"\nFinal Model -> Val Loss: {finalValLoss:.4f}, Val Accuracy: {finalValAcc:.4f}")

# %%
os.makedirs("saved-model", exist_ok=True)
torch.save(
    {
        "model_state_dict": finalModel.state_dict(),
        "wordtoindex": wordtoindex,
        "indextoword": indextoword,
        "padIndex": padIndex,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    },
    "saved-model/ffnn.pth",
)
print("Saved final model + vocab to saved-model/ffnn.pth")
