import os
import csv
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
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
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'indian_vegetation.csv')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'metrics')

def main():
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found: {DATA_PATH}. Please generate dataset first.")
        return
        
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}. Please train the model first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading dataset...")
    features = []
    labels = []
    
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header
        for row in reader:
            if len(row) == 4:
                red, nir, ndvi, label = row
                features.append([float(red), float(nir), float(ndvi)])
                labels.append(int(label))
                
    X = np.array(features)
    y = np.array(labels)
    
    # Train/test split (MUST MATCH train.py exactly)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info("Loading model...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }
    
    logger.info(f"Metrics: {metrics}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(OUTPUT_DIR, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Generate and save text report
    report = classification_report(y_test, y_pred, target_names=['Non-Grass', 'Grass'])
    report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Saved classification report to {report_path}")
    
    # Generate and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Grass', 'Grass'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")

if __name__ == '__main__':
    main()
