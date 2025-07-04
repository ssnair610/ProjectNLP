import os
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.exceptions import InconsistentVersionWarning
import warnings

from utils.cleaner import clean_text_svm
from utils.constants import LABEL_COLUMNS
from utils.MetricCalc import MetricCalc

warnings.filterwarnings("ignore", category = InconsistentVersionWarning)

def evaluate_nn(csvPath: str, pred, predname, print_flag=False):
    df = pd.read_csv(csvPath)
    acc_list = []
    yTrue = df[LABEL_COLUMNS].values.astype(int)

    y_preds = np.array(pred(csvPath), dtype=int)

    hammingAccuracy = (y_preds == yTrue).mean()

    labelAccuracy = (y_preds == yTrue).mean(axis=0)

    for label, acc in zip(LABEL_COLUMNS, labelAccuracy):
        acc_list.append(acc)

    if print_flag == True:
        print(f"\nMetrics from {predname}")

        print(f"Hamming Accuracy : {hammingAccuracy:.4f}")
        for label, acc in zip(LABEL_COLUMNS, labelAccuracy):
            print(f"{label:10s} Accuracy : {acc:.4f}")

    return y_preds, acc_list, hammingAccuracy


def evaluate_rfc(csvPath: str, log_level: str, print_flag=False):
    rfcModel = joblib.load("saved-model/rfcmodel.joblib")
    df = pd.read_csv(csvPath)

    if log_level == "emotions":
        title = "Emotion: "
    elif log_level == "macro":
        title = "Macro Average:"
    else:
        raise Exception("Invalid log level")

    x_texts = df["text"]
    vectorizer = CountVectorizer()
    x_texts_vec = vectorizer.fit_transform(x_texts)

    y_actual = df[LABEL_COLUMNS].values.astype(int)

    x_train_text, x_test_text, y_train_labels, y_test_labels = train_test_split(
        x_texts_vec, y_actual, test_size=0.1
    )

    rfcModel.fit(x_train_text, y_train_labels)
    y_preds = rfcModel.predict(x_test_text)

    confusion_mat = multilabel_confusion_matrix(y_test_labels, y_preds)

    rfc_metric = MetricCalc(confusion_mat, log_level, title, print_flag)

    if print_flag == True:
        print("\nMetrics from Random Forest Classifier:")
    rfc_metric.report()

    return y_preds, rfc_metric


def evaluate_svm(csvPath: str, log_level: str, print_flag=False):
    clf_path = os.path.join("saved-model", "svm_model.joblib")
    vec_path = os.path.join("saved-model", "vectorizer.joblib")
    clf = joblib.load(clf_path)
    vectorizer = joblib.load(vec_path)

    df = pd.read_csv(csvPath)
    if "text" not in df.columns:
        raise ValueError("Input CSV must have a 'text' column.")

    if log_level == "emotions":
        title = "Emotion: "
    elif log_level == "macro":
        title = "Macro Average:"
    else:
        raise Exception("Invalid log level")

    df["clean_text"] = df["text"].fillna("").apply(clean_text_svm)

    x_test = vectorizer.transform(df["clean_text"])
    y_true = df[LABEL_COLUMNS].astype(int)

    y_preds = clf.predict(x_test)
    y_preds = y_preds.astype(int).tolist()

    confusion_mat = multilabel_confusion_matrix(y_true, y_preds)

    if print_flag == True:
        print("\nMetrics from Support Vector Machine:")
    svm_metric = MetricCalc(confusion_mat, log_level, title, print_flag)
    svm_metric.report()

    return y_preds, svm_metric


def evaluate_nb(csvPath: str, log_level: str, print_flag=False):
    nb_path = os.path.join("saved-model", "nbmodel.joblib")
    nb_model = joblib.load(nb_path)

    df = pd.read_csv(csvPath)

    if log_level == "emotions":
        title = "Emotion: "
    elif log_level == "macro":
        title = "Macro Average:"
    else:
        raise Exception("Invalid log level")

    x_texts = df["text"]
    y_actual = df[LABEL_COLUMNS].values.astype(int)

    x_train_text, x_test_text, y_train_labels, y_test_labels = train_test_split(
        x_texts, y_actual, test_size=0.1
    )

    y_preds = nb_model.predict(x_test_text)

    confusion_mat = multilabel_confusion_matrix(y_test_labels, y_preds)

    if print_flag == True:
        print("\nMetrics from Naive Bayes Classifier:")
    nb_metric = MetricCalc(confusion_mat, log_level, title, print_flag)
    nb_metric.report()

    return y_preds, nb_metric
