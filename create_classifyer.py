import torch
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    # Forward is called when we pass data through the model.
    #It is called automatically by the parent class when we call the model with input data.
    def forward(self, x):
        return self.net(x)





from torch.utils.data import Dataset, DataLoader

class EmotionDataset(Dataset):
    def __init__(self, embeddings, labels):

        #converts your input data and labels from regular Python lists or NumPy arrays ---> PyTorch tensors
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.normalize()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    
    def normalize(self):
        # Normalize the embeddings to have zero mean and unit variance
        self.embeddings = (self.embeddings - self.embeddings.mean(dim=0)) / self.embeddings.std(dim=0)






def main(embeddings, labels, input_dim=768, hidden_dim=256, output_dim=5):



    dataset = EmotionDataset(embeddings = embeddings, labels=labels)  # X = list of vectors, y = list of [0,1,0,0,1]
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = EmotionClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



    model.train()
    for epoch in range(5):  # number of passes over data
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    print("Training complete!")



    model.eval()  # switch to evaluation mode
    with torch.no_grad():
        for i, (x, true_labels) in enumerate(dataset):
            logits = model(x.unsqueeze(0))        # add batch dim
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()           # threshold

            print(f"Sample {i+1}")
            print("  True: ", true_labels.tolist())
            print("  Pred: ", preds.squeeze(0).tolist())
    
    torch.save(model, "emotion_classifier_full.pt")
    return model


