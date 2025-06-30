import sys
import os
import pandas as pd                     

# 1. Setze Importpfad, damit Imports aus NLP_second_try funktionieren
nlp_folder = os.path.join(os.path.dirname(__file__), "NLP_second_try")
sys.path.insert(0, nlp_folder)

# 2. Setze das Arbeitsverzeichnis (für relative Pfade wie torch.load)
os.chdir(nlp_folder)

import NLP_second_try.main as the_main_here_takes_word_to_embedding_and_creates_classifyerNN

def main(path):

    df = pd.read_csv(path)                  
    # Erstelle Vorhersagen exakt in der Reihenfolge der Texte
    predictions = [
        the_main_here_takes_word_to_embedding_and_creates_classifyerNN.main(text)
        for text in df.iloc[:, 1]           
    ]
    return predictions                      

if __name__ == "__main__":
    preds = main("23432")                    # Path to CSV
    print(preds)


