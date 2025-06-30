from helper import *

file_path = os.path.join("Data", "track-a-test-large.csv")
save_flag = True

best_pred, best_accuracy, best_model = predict(file_path, save_flag)

print(f"Model: {best_model} with accuracy: {best_accuracy}")

if save_flag == True:
    rfc_val, fnn_val, gru_val, lstm_val, svm_val, nb_val = predict_plotter(testfile)
    label_plotter(fnn_val, gru_val, lstm_val, rfc_val.accuracy_list, svm_val.accuracy_list, nb_val.accuracy_list, save_flag)