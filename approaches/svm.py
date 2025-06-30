# %%
import pandas as pd
import os
import re
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

import joblib

# %%
csv_path = os.path.join("data", "track-a.csv")

df = pd.read_csv(csv_path)

# %%
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


df["clean_text"] = df["text"].apply(clean_text)

texts = df["clean_text"]
labels = df[["anger", "fear", "joy", "sadness", "surprise"]].values

print(texts.head())


# %%
vectorizer = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95
)
X = vectorizer.fit_transform(df["clean_text"])


# %%
svm = LinearSVC(class_weight="balanced")
clf = OneVsRestClassifier(svm)

clf.fit(X, labels)
print("Training complete!")

# %%
joblib.dump(clf, "saved-model/svm_model.joblib")
joblib.dump(vectorizer, "saved-model/vectorizer.joblib")
