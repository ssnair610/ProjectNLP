import pandas as pd
import re
from svm_final import predict
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('emotion_data.csv')

def extract_numeric_id(row_id):
    match = re.search(r'(\d+)$', str(row_id))
    return int(match.group(1)) if match else -1

df['num_id'] = df['id'].apply(extract_numeric_id)

filtered = df[(df['num_id'] >= 2000) & (df['num_id'] <= 2499)].copy()



filtered.to_csv('sample_test.csv', index=False)


preds = predict('sample_test.csv')

y_true = filtered[['anger', 'fear', 'joy', 'sadness', 'surprise']].values.tolist()
y_pred = preds

emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
print("Accuracy per label:")
for i, label in enumerate(emotion_labels):
    col_true = [yt[i] for yt in y_true]
    col_pred = [yp[i] for yp in y_pred]
    acc = accuracy_score(col_true, col_pred)
    print(f"{label}: {acc:.3f}")

y_true_flat = sum(y_true, [])
y_pred_flat = sum(y_pred, [])

micro_acc = accuracy_score(y_true_flat, y_pred_flat)
print(f"\nMicro accuracy (overall): {micro_acc:.3f}")

print("\nDetailed classification report:")
print(classification_report(y_true, y_pred, target_names=emotion_labels))

for idx, (text, yt, yp) in enumerate(zip(filtered['text'], y_true, y_pred)):
    print(f"Text: {text}")
    print("Actual:   ", yt)
    print("Predicted:", yp)
    print('-' * 60)
    if idx == 9:  
        break
