import os
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'indian_vegetation.csv')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')

def main():
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found: {DATA_PATH}. Please run generate_dataset.py first.")
        return

    logger.info("Loading localized Indian dataset...")
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
    
    logger.info(f"Total pixel samples: {len(X)}")
    logger.info(f"Grass pixels: {np.sum(y == 1)}, Non-grass pixels: {np.sum(y == 0)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest on PIXEL data (Red, NIR, NDVI)
    logger.info("Training Pixel-level Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Non-Grass', 'Grass']))
    
    # Save Model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Pixel-level Model saved: {MODEL_PATH}")

if __name__ == '__main__':
    main()