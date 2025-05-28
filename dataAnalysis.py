import nltk
import spacy
import numpy
import pandas as pd

track_a = pd.read_csv("Data/track-a.csv")
track_a_df = pd.DataFrame(track_a)
print(track_a_df.columns)
print(track_a_df.head(25))
print(len(track_a_df["id"]))

