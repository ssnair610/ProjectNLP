import numpy as np
import plotly.graph_objects as go

from utils.evaluate import evaluate_nb, evaluate_nn, evaluate_rfc, evaluate_svm
from utils.predictors import predictFNN, predictLSTM, predictGRU


def predict_plotter(testfile):
    pred_rfc, rfc_val = evaluate_rfc(testfile, "emotions")
    pred_fnn, fnn_val, fnn_total_acc = evaluate_nn(testfile, predictFNN, "FNN")
    pred_gru, gru_val, gru_total_acc = evaluate_nn(testfile, predictGRU, "GRU")
    pred_lstm, lstm_val, lstm_total_acc = evaluate_nn(testfile, predictLSTM, "LSTM")
    pred_svm, svm_val = evaluate_svm(testfile, "emotions")
    pred_nb, nb_val = evaluate_nb(testfile, "emotions")

    return rfc_val, fnn_val, gru_val, lstm_val, svm_val, nb_val


def label_plotter(
    fnn_acc, gru_acc, lstm_acc, rfc_acc, svm_acc, nb_acc, save_image=False
):
    model_names = [
        "Neural Network: Feed Forward",
        "Neural Network: LSTM",
        "Neural Network: GRU",
        "Random Forest Classifier",
        "Support Vector Machine",
        "Naive Bayes Classifier",
    ]
    label_names = ["anger", "fear", "joy", "sadness", "surprise"]

    z_values = np.stack([fnn_acc, lstm_acc, gru_acc, rfc_acc, svm_acc, nb_acc], axis=1)

    x_values = np.arange(len(model_names))
    y_values = np.arange(len(label_names))

    fig = go.Figure(
        data=go.Surface(x=x_values, y=y_values, z=z_values, colorscale="agsunset")
    )

    fig.update_layout(
        title="Model vs Label Metric Surface",
        scene=dict(
            xaxis=dict(
                tickmode="array", tickvals=x_values, ticktext=model_names, title="Model"
            ),
            yaxis=dict(
                tickmode="array", tickvals=y_values, ticktext=label_names, title="Label"
            ),
            zaxis=dict(title="Accuracy"),
        ),
        width=1200,
        height=1200,
        margin=dict(l=50, r=50, b=50, t=50),
        scene_camera=dict(eye=dict(x=1.7, y=1.7, z=1.7)),
    )

    if save_image == True:
        fig.write_image(file="accuracy_plot.png", height=1500, width=1500, scale=1)
        print(f"Plot saved! Do check 'accuracy_plot.png'! ^_^")
    else:
        fig.show()


def plot_model_data(file_path: str, save_image: bool):
    rfc_val, fnn_val, gru_val, lstm_val, svm_val, nb_val = predict_plotter(file_path)
    label_plotter(
        fnn_val,
        gru_val,
        lstm_val,
        rfc_val.accuracy_list,
        svm_val.accuracy_list,
        nb_val.accuracy_list,
        save_image,
    )
