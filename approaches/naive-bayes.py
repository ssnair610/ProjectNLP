# %%
import pandas as pd
import numpy as np
import os
import joblib

# %%
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# %% [markdown]
# ## Read Data


# %%
def read_data(filename, type="csv"):
    if type == "csv":
        data = pd.read_csv(filename)
        data_df = pd.DataFrame(data)
        return data_df

    elif type == "excel":
        data = pd.read_excel(filename)
        data_df = pd.DataFrame(data)
        return data_df


# %%
datapath = os.path.join("data", "track-a.csv")
track_a = read_data(datapath)

# %% [markdown]
# ## Train Test Split

# %%
x = track_a.iloc[:, 1]
y = track_a.iloc[:, 2:]
test_size = 0.20

# %%
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=42
)

# %% [markdown]
# ## Training Model


# %%
def multinomialNB_classifier(X_train, y_train):
    model = make_pipeline(TfidfVectorizer(), OneVsRestClassifier(MultinomialNB()))
    model.fit(X_train, y_train)
    return model


# %%
model_NB = multinomialNB_classifier(X_train, y_train)

# %%
joblib.dump(model_NB, "saved-model/nbmodel.joblib")

# %%
y_pred = model_NB.predict(X_test)


# %%
def calculate_results(y_test, y_pred):
    result_dict = {}
    result_dict["accuracy"] = accuracy_score(y_test, y_pred)

    result_dict["f1_micro"] = f1_score(y_test, y_pred, average="micro")
    result_dict["f1_macro"] = f1_score(y_test, y_pred, average="macro")

    result_dict["report"] = classification_report(
        y_test,
        y_pred,
        target_names=["anger", "fear", "joy", "sadness", "surprise"],
        zero_division=0,
    )

    return result_dict


# %%
def accuracy_per_label(y_test, y_pred):
    accuracies = []
    labels = ["anger", "fear", "joy", "sadness", "surprise"]
    for i in range(5):
        accuracies.append(accuracy_score(y_test.iloc[:, i], y_pred[:, i]))

        print(f"Accuracy {labels[i]}: {accuracies[i] }")
    return accuracies


# %%
results = calculate_results(y_test, y_pred)

# %%
accuracies = accuracy_per_label(y_test, y_pred)

# %%
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(np.mean(accuracies))

# %%
print(results["f1_micro"])
print(results["f1_macro"])

# %%
print(results["report"])

# %% [markdown]
# ## Trying Logistic Regression

# %%
clf = MultiOutputClassifier(LogisticRegression(solver="liblinear"))
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
clf.fit(X_train_tfidf, y_train)
y_pred_lr = clf.predict(X_test_tfidf)

# %%

accuracies_lr = accuracy_per_label(y_test, y_pred_lr)

# %%
accuracy_score(y_test, y_pred_lr)
print(accuracy)
print(np.mean(accuracies_lr))

# %%
results_lr = calculate_results(y_test, y_pred_lr)
print(results_lr["f1_micro"])
print(results_lr["f1_macro"])
