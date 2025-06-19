import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text.strip()

def predict(csv_path):

    clf = joblib.load('svm_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')

    
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns:
        raise ValueError("Input CSV must have a 'text' column.")

    df['clean_text'] = df['text'].fillna('').apply(clean_text)


    X_test = vectorizer.transform(df['clean_text'])


    preds = clf.predict(X_test)

    preds = preds.astype(int).tolist()

    return preds

if __name__ == "__main__":
    preds = predict('emotion_data.csv')
    print(preds)
