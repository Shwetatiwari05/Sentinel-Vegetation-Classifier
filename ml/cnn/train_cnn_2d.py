import os
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'indian_vegetation_patches.npz')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'cnn_model.pth')

class GrassCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(GrassCNN, self).__init__()
        # Input: (batch, 3, 5, 5)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool: (batch, 32, 2, 2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=1)
        # After conv2: (batch, 64, 3, 3)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)  # 3 output classes

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Raw logits — CrossEntropyLoss handles softmax
        return x

def main():
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found: {DATA_PATH}. Please run generate_patches.py first.")
        return

    logger.info("Loading 3-class patch dataset...")
    data = np.load(DATA_PATH)
    X = data['X'].astype(np.float32)  # (N, 5, 5, 3)
    y = data['y'].astype(np.int64)    # Labels: 0, 1, or 2

    # Normalize
    X[:, :, :, 0] = X[:, :, :, 0] / 10000.0  # Red
    X[:, :, :, 1] = X[:, :, :, 1] / 10000.0  # NIR
    # NDVI already in -1 to 1

    # PyTorch expects channels first: (N, 3, 5, 5)
    X = np.transpose(X, (0, 3, 1, 2))

    logger.info(f"Loaded {len(X)} patches. Shape: {X.shape}")
    logger.info(f"Non-Veg: {(y==0).sum()}, Grass: {(y==1).sum()}, Trees: {(y==2).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    logger.info("Training 3-class CNN...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GrassCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()  # handles multi-class
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} completed.")

    # Evaluate
    logger.info("Evaluating CNN...")
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            y_pred_list.append(preds.cpu().numpy())

    y_pred = np.concatenate(y_pred_list)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"CNN Accuracy: {acc:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Non-Veg', 'Grass', 'Tree']))

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"3-class CNN model saved: {MODEL_PATH}")

if __name__ == '__main__':
    main()