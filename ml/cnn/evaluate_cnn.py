import os
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from torch.utils.data import DataLoader, TensorDataset
import logging

from train_cnn_2d import GrassCNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'indian_vegetation_patches.npz')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'cnn_model.pth')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'metrics')

CLASS_NAMES = ['Non-Veg', 'Grass', 'Tree']

def main():
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found: {DATA_PATH}. Please run generate_patches.py first.")
        return

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}. Please run train_cnn_2d.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading 3-class patch dataset...")
    data = np.load(DATA_PATH)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)

    # Normalize — must match train_cnn_2d.py exactly
    X[:, :, :, 0] = X[:, :, :, 0] / 10000.0  # Red
    X[:, :, :, 1] = X[:, :, :, 1] / 10000.0  # NIR

    # Channels first: (N, 3, 5, 5)
    X = np.transpose(X, (0, 3, 1, 2))

    # Train/test split — must match train_cnn_2d.py exactly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    logger.info("Loading 3-class CNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GrassCNN(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    logger.info("Evaluating on test set...")
    y_pred_list = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs  = model(batch_x)
            preds    = torch.argmax(outputs, dim=1)
            y_pred_list.append(preds.cpu().numpy())

    y_pred = np.concatenate(y_pred_list)

    # Metrics
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_macro":  round(f1_score(y_test, y_pred, average='macro'), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average='macro'), 4),
        "recall_macro":    round(recall_score(y_test, y_pred, average='macro'), 4),
        "per_class": {
            cls: {
                "precision": round(precision_score(y_test, y_pred, labels=[i], average='micro'), 4),
                "recall":    round(recall_score(y_test, y_pred, labels=[i], average='micro'), 4),
                "f1":        round(f1_score(y_test, y_pred, labels=[i], average='micro'), 4),
            }
            for i, cls in enumerate(CLASS_NAMES)
        }
    }

    logger.info(f"Accuracy: {metrics['accuracy']}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Save metrics JSON
    metrics_path = os.path.join(OUTPUT_DIR, 'cnn_evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics: {metrics_path}")

    # Save text report
    report_path = os.path.join(OUTPUT_DIR, 'cnn_classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    logger.info(f"Saved report: {report_path}")

    # Confusion matrix
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Greens, ax=ax)
    plt.title('CNN 3-Class Confusion Matrix')
    cm_path = os.path.join(OUTPUT_DIR, 'cnn_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Saved confusion matrix: {cm_path}")

if __name__ == '__main__':
    main()