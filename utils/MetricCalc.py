from utils.constants import LABEL_COLUMNS


class MetricCalc:
    def __init__(
        self, confusion_mat, log_level="macro", title="Accuracy", print_flag=False
    ) -> None:
        self.confusion_mat = confusion_mat
        self.log_level = log_level
        self.title = title
        self.accuracy_list = []
        self.overall_accuracy = 0
        self.print_flag = print_flag

    @staticmethod
    def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
        denom = tp + tn + fp + fn
        return 0.0 if denom == 0 else (tp + tn) / denom

    def report(self) -> None:
        n_labels = len(self.confusion_mat)
        if n_labels == 0:
            print("No confusion matrices supplied.")
            return

        for cm in self.confusion_mat:
            tp, fp = cm[0]
            fn, tn = cm[1]
            self.accuracy_list.append(self.accuracy(tp, tn, fp, fn))

        macro_acc = sum(self.accuracy_list) / n_labels
        self.overall_accuracy = macro_acc

        if self.print_flag == True:
            print(f"Hamming Accuracy : {macro_acc:.4f}")

            if self.log_level == "emotions":
                for idx, acc in enumerate(self.accuracy_list):
                    print(f"{LABEL_COLUMNS[idx].ljust(12)} Accuracy : {acc:.4f}")
