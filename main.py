def get_global_data():

    from bert_encoder import Bert_Encoder
    model_id = "answerdotai/ModernBERT-base"
    sentence_path = "Data_here/track-a (1).csv"
    bert_encoder = Bert_Encoder(model_id)
    classification_model_path = "emotion_classifier_full.pt"

    return sentence_path, bert_encoder, classification_model_path





def main_model_creation():
    from data_pipeline import Structure_data_Pipeline
    from create_classifyer import main as create_classifier_main
    import os


    sentence_path, bert_encoder, classification_model_path = get_global_data()

    Nord_Stream_2 = Structure_data_Pipeline(sentence_path, bert_encoder)
    sentence_labels_and_embeddings = Nord_Stream_2.return_sentence_labels_and_embeddings()

    embeddings = [item["embedding"].squeeze(0).tolist() for item in sentence_labels_and_embeddings]
    labels = [
        [
            item["emotion_labels"]["anger"],
            item["emotion_labels"]["fear"],
            item["emotion_labels"]["joy"],
            item["emotion_labels"]["sadness"],
            item["emotion_labels"]["surprise"]
        ]
        for item in sentence_labels_and_embeddings
    ]


    model = create_classifier_main(embeddings=embeddings, labels=labels, input_dim=len(embeddings[0]))
    return model

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Call this function to get an emotion classification for a given text using the pre-trained model.
def main(text):

    from get_classification import get_classification
    sentence_path, bert_encoder, classification_model_path = get_global_data()

    result = get_classification(text, bert_encoder=bert_encoder, classification_model_path=classification_model_path)
    print("Classification result:", result)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# This is the main function that will be run when the script is executed.











if __name__ == "__main__":

    #If the model is not present run the main_model_creation function to create it.
    # ---> this one:

    #-------------------------#
    # main_model_creation
    #-------------------------#

    main(text="This is a test sentence to check the BERT encoder functionality. I am very happy today, but I am also a bit sad about the weather.")


