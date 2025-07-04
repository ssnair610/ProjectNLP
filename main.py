import os
from utils.plotter import plot_model_data

# Insert File-path of test file below

# ------------------------------------------------------------------------ #

FILE_PATH = os.path.join("data", "track-a-test-large.csv")

# ------------------------------------------------------------------------ #

# Save graph as image : Boolean
# ------------------------------------------------------------------------ #
save_image = True
# ------------------------------------------------------------------------ #

from utils.evaluate import evaluate_nb, evaluate_nn, evaluate_rfc, evaluate_svm
from utils.predictors import predictFNN, predictGRU, predictLSTM


def predict(testfile):
    pred_rfc, rfc_val = evaluate_rfc(testfile, "emotions")
    pred_fnn, fnn_val, fnn_total_acc = evaluate_nn(testfile, predictFNN, "FNN")
    pred_gru, gru_val, gru_total_acc = evaluate_nn(testfile, predictGRU, "GRU")
    pred_lstm, lstm_val, lstm_total_acc = evaluate_nn(testfile, predictLSTM, "LSTM")
    pred_svm, svm_val = evaluate_svm(testfile, "emotions")
    pred_nb, nb_val = evaluate_nb(testfile, "emotions")

    overall_accuracies = [
        rfc_val.overall_accuracy,
        fnn_total_acc,
        gru_total_acc,
        lstm_total_acc,
        svm_val.overall_accuracy,
        nb_val.overall_accuracy,
    ]
    predictions = [pred_rfc, pred_fnn, pred_gru, pred_lstm, pred_svm, pred_nb]

    max_index = overall_accuracies.index(max(overall_accuracies))
    best_pred = predictions[max_index]
    model_names = [
        "Random Forest Classifier",
        "Neural Network - Feed Forward",
        "Neural Network - GRU",
        "Neural Network - LSTM",
        "Support Vector Machine",
        "Naive Bayes Classifier",
    ]

    return best_pred, overall_accuracies[max_index], model_names[max_index]


def main():
    best_pred, best_accuracy, best_model = predict(FILE_PATH)
    print(f"Model: {best_model} with accuracy: {best_accuracy}")

    if save_image:
        plot_model_data(FILE_PATH, save_image)


if __name__ == "__main__":
    main()
