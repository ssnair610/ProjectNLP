# %% [markdown]
# # Random Forest - Implementation

# %%
import numpy as np
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
import os
import optuna
import joblib

# %% [markdown]
# ## Read data

# %%
file_path = os.path.join("data", "track-a.csv")
dataframe = pd.read_csv(file_path)
dataframe.head()

# %% [markdown]
# ## Data Pre-processing

# %% [markdown]
# ### Stop word removal

# %%
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

dataframe["text"] = dataframe["text"].apply(
    lambda x: " ".join([word for word in x.split() if word not in (stop_words)])
)

# %% [markdown]
# ### Lemmatization

# %%
word_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
# TODO: Lot of punctuations (remove)


def lemmatize_text(txt):
    out = ""
    for word in word_tokenizer.tokenize(txt):
        out = out + lemmatizer.lemmatize(word) + " "
    return out


dataframe["text"] = dataframe.text.apply(lemmatize_text)


# %% [markdown]
# ### Train-Test Split

# %%
from sklearn.feature_extraction.text import CountVectorizer

X_texts = dataframe["text"]
vectorizer = CountVectorizer()
X_texts_vec = vectorizer.fit_transform(X_texts)


emotions = ["anger", "fear", "joy", "sadness", "surprise"]
Y_emotions = dataframe[emotions]


X_train_text, X_test_text, Y_train_labels, Y_test_labels = train_test_split(
    X_texts_vec, Y_emotions, test_size=0.1
)

# %% [markdown]
# ## Random Forest Classification

# %%
from sklearn.ensemble import RandomForestClassifier


# %%
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 25, 500, step=25)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    max_depth = trial.suggest_int("max_depth", 2, 64, step=2)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    random_state = trial.suggest_int("random_state", 0, 100, step=20)
    warm_start = trial.suggest_categorical("warm_start", [False, True])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
        warm_start=warm_start,
    )

    model.fit(X_train_text, Y_train_labels)

    yPred = model.predict(X_test_text)

    confusionMatrix = multilabel_confusion_matrix(Y_test_labels, yPred)
    f1s = []

    for i in range(len(confusionMatrix)):
        tn, fp = confusionMatrix[i][0]
        fn, tp = confusionMatrix[i][1]

        if (tp + fp) == 0 or (tp + fn) == 0:
            f1 = 0
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            if (prec + rec) == 0:
                f1 = 0
            else:
                f1 = 2 * (prec * rec) / (prec + rec)

        f1s.append(f1)
        trial.set_user_attr(f"f1_{emotions[i]}", f1)

    macro_f1 = np.mean(f1s)
    return macro_f1


# %%
studyRFC = optuna.create_study(
    study_name="RFC",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    storage="sqlite:///db.sqlite3",
)

# %%
# optuna.delete_study(study_name="RFC", storage = "sqlite:///db.sqlite3")


# %%
def _objective(trial):
    return objective(trial)


# %%
studyRFC.optimize(_objective, n_trials=500)

# %%
print("\nOptuna Best Trial:")
best = studyRFC.best_trial
print(f"Validation Loss: {best.value:.4f}")
for key, val in best.params.items():
    print(f"  {key}: {val}")
print("\n")

best_n_estimators = best.params["n_estimators"]
best_criterion = best.params["criterion"]
best_max_depth = best.params["max_depth"]
best_max_features = best.params["max_features"]
best_random_state = best.params["random_state"]
best_warm_start = best.params["warm_start"]

# %%
finalModel = RandomForestClassifier(
    n_estimators=best_n_estimators,
    criterion=best_criterion,
    max_depth=best_max_depth,
    max_features=best_max_features,
    random_state=best_random_state,
    warm_start=best_warm_start,
)

# %%
joblib.dump(finalModel, "saved-model/rfcmodel.joblib")

# %%
model = RandomForestClassifier(n_estimators=10, max_features=4, random_state=101)

# %%
model.fit(X_train_text, Y_train_labels)

# %%
predictions = model.predict(X_test_text)

# %% [markdown]
# ### Evaluation

# %%
confusion_mat = multilabel_confusion_matrix(Y_test_labels, predictions)


# %%
def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def f1_score(tp, fp, fn):
    precision_val = precision(tp, fp)
    recall_val = recall(tp, fn)
    return 2 * (precision_val * recall_val) / (precision_val + recall_val)


# %%
def print_eval(title, accuracy, precision, recall, f1_score):
    print(f"{title}\n")
    print(f"accuracy: {round(accuracy,2)}")
    print(f"precision: {round(precision,2)}")
    print(f"recall: {round(recall,2)}")
    print(f"f1 Score: {round(f1_score,2)}")
    print("=======\n")


def present_data(log_level):
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0

    for i in range(0, len(confusion_mat)):
        tp, fp = confusion_mat[i][0]
        fn, tn = confusion_mat[i][1]

        accuracy_val = accuracy(tp, tn, fp, fn)
        precision_val = precision(tp, fp)
        recall_val = recall(tp, fn)
        f1_score_val = f1_score(tp, fp, fn)

        total_accuracy += accuracy_val
        total_precision += precision_val
        total_recall += recall_val
        total_f1_score += f1_score_val

        if log_level == "emotions":
            print_eval(
                f"Emotion: {emotions[i]}",
                accuracy_val,
                precision_val,
                recall_val,
                f1_score_val,
            )

    avg_accuracy = total_accuracy / len(confusion_mat)
    avg_precision = total_precision / len(confusion_mat)
    avg_recall = total_recall / len(confusion_mat)
    avg_f1_score = total_f1_score / len(confusion_mat)

    if log_level == "macro":
        print_eval(
            "Macro Average:", avg_accuracy, avg_precision, avg_recall, avg_f1_score
        )


# %%
present_data("emotions")

# %%
present_data("macro")
