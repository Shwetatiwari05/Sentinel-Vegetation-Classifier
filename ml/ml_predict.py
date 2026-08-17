import os
import pickle
import sys
import numpy as np
import logging
import torch

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Class labels for 3-class CNN
CNN_CLASSES = {0: 'Non-Vegetation', 1: 'Grass', 2: 'Tree'}
BINARY_CLASSES = {0: 'Non-Grass', 1: 'Grass'}

def load_model(model_type: str = 'random_forest'):
    if model_type == 'random_forest':
        path = os.path.join(MODEL_DIR, 'model.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}. Run random_forest/train.py first.")
        with open(path, 'rb') as f:
            return pickle.load(f), None

    elif model_type == 'svm':
        path = os.path.join(MODEL_DIR, 'svm_model.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}. Run svm/train.py first.")
        with open(path, 'rb') as f:
            return pickle.load(f), None

    elif model_type == 'cnn_2d':
        import torch
        import sys
        ml_dir = os.path.dirname(os.path.abspath(__file__))
        if ml_dir not in sys.path:
            sys.path.insert(0, ml_dir)
        from cnn.train_cnn_2d import GrassCNN
        path = os.path.join(MODEL_DIR, 'cnn_model.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}. Run cnn/train_cnn_2d.py first.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GrassCNN(num_classes=3)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
def compute_ndvi(red, nir):
    """NDVI = (NIR - Red) / (NIR + Red)"""
    denominator = nir + red
    return np.where(denominator == 0, 0, (nir - red) / denominator)

def predict_grass(red_band: np.ndarray, nir_band: np.ndarray, model_type: str = 'random_forest') -> dict:
    """
    Accepts 2D Red and NIR band arrays and predicts pixel-by-pixel.
    For RF/SVM: binary (Grass vs Non-Grass)
    For CNN: 3-class (Non-Veg, Grass, Tree)
    """
    model, device = load_model(model_type)

    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    ndvi = compute_ndvi(red, nir)
    original_shape = red.shape

    if model_type in ('random_forest', 'svm'):
        # Flatten to (N, 3) feature matrix
        flat_red  = red.flatten()
        flat_nir  = nir.flatten()
        flat_ndvi = ndvi.flatten()
        features  = np.column_stack((flat_red, flat_nir, flat_ndvi))

        predictions  = model.predict(features)
        probabilities = model.predict_proba(features)[:, 1]

        mask_2d = predictions.reshape(original_shape)

        total_pixels = len(predictions)
        grass_pixels = np.sum(predictions == 1)
        grass_pct    = (grass_pixels / total_pixels) * 100

        is_grass = grass_pct > 20
        confidence = round(float(np.mean(probabilities)) * 100, 2)

        return {
            "prediction": "Grass" if is_grass else "Non-Grass",
            "is_grass": is_grass,
            "confidence": confidence,
            "grass_percentage": round(grass_pct, 2),
            "tree_percentage": 0.0,
            "non_veg_percentage": round(100 - grass_pct, 2),
            "ndvi_mean": round(float(np.mean(ndvi)), 4),
            "grass_mask_2d": mask_2d == 1,
            "tree_mask_2d": np.zeros(original_shape, dtype=bool),
            "ndvi_array": ndvi
        }

    elif model_type == 'cnn_2d':
        # Build 5x5 patches for every pixel using padding
        pad = 2
        red_pad  = np.pad(red,  pad, mode='edge')
        nir_pad  = np.pad(nir,  pad, mode='edge')
        ndvi_pad = np.pad(ndvi, pad, mode='edge')

        H, W = original_shape
        patches = []
        for i in range(H):
            for j in range(W):
                r_patch    = red_pad[i:i+5, j:j+5]   / 10000.0
                nir_patch  = nir_pad[i:i+5, j:j+5]   / 10000.0
                ndvi_patch = ndvi_pad[i:i+5, j:j+5]
                patch = np.stack([r_patch, nir_patch, ndvi_patch], axis=0)  # (3,5,5)
                patches.append(patch)

        patches = np.array(patches, dtype=np.float32)  # (N, 3, 5, 5)

        # Run inference in batches
        all_preds = []
        all_probs = []
        batch_size = 1024

        with torch.no_grad():
            for start in range(0, len(patches), batch_size):
                batch = torch.tensor(patches[start:start+batch_size]).to(device)
                logits = model(batch)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                preds  = np.argmax(probs, axis=1)
                all_preds.append(preds)
                all_probs.append(probs)

        predictions = np.concatenate(all_preds)        # (N,)
        probabilities = np.concatenate(all_probs)      # (N, 3)

        mask_2d = predictions.reshape(original_shape)

        total = len(predictions)
        non_veg_pct = round((np.sum(predictions == 0) / total) * 100, 2)
        grass_pct   = round((np.sum(predictions == 1) / total) * 100, 2)
        tree_pct    = round((np.sum(predictions == 2) / total) * 100, 2)

        # Overall label based on dominant class
        dominant = int(np.bincount(predictions).argmax())
        dominant_label = CNN_CLASSES[dominant]

        avg_confidence = round(float(np.mean(np.max(probabilities, axis=1))) * 100, 2)

        return {
            "prediction": dominant_label,
            "is_grass": dominant == 1,
            "confidence": avg_confidence,
            "grass_percentage": grass_pct,
            "tree_percentage": tree_pct,
            "non_veg_percentage": non_veg_pct,
            "ndvi_mean": round(float(np.mean(ndvi)), 4),
            "grass_mask_2d": mask_2d == 1,
            "tree_mask_2d": mask_2d == 2,
            "ndvi_array": ndvi
        }