import os
import ee
import numpy as np
import logging
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backend', '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SENTINEL_2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
MAX_CLOUD_COVER = 10
BAND_RED  = 'B4'
BAND_NIR  = 'B8'
BAND_RE   = 'B5'   # Red Edge — tree detection ke liye
BAND_SWIR = 'B11'  # SWIR — woody vegetation ke liye
PATCH_RADIUS = 3   # 7x7 patch (radius 3)

def initialize_gee() -> None:
    project_id = os.getenv("EE_PROJECT_ID")
    if not project_id:
        raise ValueError("Missing EE_PROJECT_ID in backend/.env")
    ee.Initialize(project=project_id)
    logger.info(f"GEE initialized with project: {project_id}")

def get_image(region: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    collection = (ee.ImageCollection(SENTINEL_2_COLLECTION)
                  .filterBounds(region)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER)))
    if collection.size().getInfo() == 0:
        raise ValueError("No images found for the given parameters.")
    image = collection.median()
    ndvi = image.normalizedDifference([BAND_NIR, BAND_RED]).rename('NDVI')
    return image.select([BAND_RED, BAND_NIR]).addBands(ndvi)

def sample_patches(image: ee.Image, region: ee.Geometry, label: int, num_pixels: int = 500):
    """Sample 5x5 image patches randomly from the image within the region."""
    kernel = ee.Kernel.square(PATCH_RADIUS, 'pixels')
    array_image = image.neighborhoodToArray(kernel)
    samples = array_image.sample(
        region=region,
        scale=10,
        numPixels=num_pixels,
        seed=42,
        geometries=False
    )
    features = samples.getInfo()['features']
    X_list = []
    y_list = []
    for feat in features:
        props = feat['properties']
        if BAND_RED in props and BAND_NIR in props and 'NDVI' in props:
            red_patch = np.array(props[BAND_RED])
            nir_patch = np.array(props[BAND_NIR])
            ndvi_patch = np.array(props['NDVI'])
            if red_patch.shape == (5, 5) and nir_patch.shape == (5, 5) and ndvi_patch.shape == (5, 5):
                patch = np.stack([red_patch, nir_patch, ndvi_patch], axis=-1)
                X_list.append(patch)
                y_list.append(label)
    return X_list, y_list

def main():
    initialize_gee()

    start_date = '2024-01-01'
    end_date = '2024-05-30'

    # --- Label 1: GRASS (open grasslands, parks, farmland) ---
    grass_delhi  = ee.Geometry.Point([77.2197, 28.5933]).buffer(300)  # Lodhi Gardens
    grass_blr    = ee.Geometry.Point([77.5946, 12.9779]).buffer(300)  # Cubbon Park
    grass_punjab = ee.Geometry.Point([75.8500, 30.9000]).buffer(500)  # Punjab fields
    grass_mumbai = ee.Geometry.Point([72.8777, 19.0760]).buffer(300)  # Oval Maidan

    # --- Label 2: TREES (dense forest / tree canopy) ---
    tree_ghats   = ee.Geometry.Point([73.5500, 17.9200]).buffer(500)  # Western Ghats forest
    tree_jim     = ee.Geometry.Point([78.9629, 29.5300]).buffer(500)  # Jim Corbett forest
    tree_coorg   = ee.Geometry.Point([75.7382, 12.3375]).buffer(500)  # Coorg coffee plantations
    tree_shillong = ee.Geometry.Point([91.8933, 25.5788]).buffer(400) # Meghalaya forest

    # --- Label 0: NON-VEGETATION (urban, desert, water) ---
    urban_delhi  = ee.Geometry.Point([77.2120, 28.6430]).buffer(300)  # Paharganj
    urban_mumbai = ee.Geometry.Point([72.8562, 19.0402]).buffer(300)  # Dharavi
    desert_thar  = ee.Geometry.Point([70.9000, 27.1000]).buffer(500)  # Thar Desert
    water_river  = ee.Geometry.Point([77.2450, 28.6130]).buffer(200)  # Yamuna River

    X_all = []
    y_all = []

    def fetch_and_append(region, name, label, num_pixels):
        logger.info(f"Fetching patches for {name} (label={label})...")
        img = get_image(region, start_date, end_date)
        X, y = sample_patches(img, region, label=label, num_pixels=num_pixels)
        logger.info(f"Sampled {len(X)} patches from {name}.")
        return X, y

    # Grass patches (Label = 1)
    for region, name, n in [
        (grass_delhi,  "Lodhi Gardens Delhi (Grass)",  400),
        (grass_blr,    "Cubbon Park Bangalore (Grass)", 400),
        (grass_punjab, "Punjab Fields (Grass)",         500),
        (grass_mumbai, "Oval Maidan Mumbai (Grass)",    300),
    ]:
        X, y = fetch_and_append(region, name, 1, n)
        X_all.extend(X); y_all.extend(y)

    # Tree patches (Label = 2)
    for region, name, n in [
        (tree_ghats,    "Western Ghats (Tree)",   500),
        (tree_jim,      "Jim Corbett (Tree)",      500),
        (tree_coorg,    "Coorg Forest (Tree)",     400),
        (tree_shillong, "Meghalaya Forest (Tree)", 400),
    ]:
        X, y = fetch_and_append(region, name, 2, n)
        X_all.extend(X); y_all.extend(y)

    # Non-vegetation patches (Label = 0)
    for region, name, n in [
        (urban_delhi,  "Delhi Urban (Non-Veg)",  400),
        (urban_mumbai, "Mumbai Urban (Non-Veg)", 400),
        (desert_thar,  "Thar Desert (Non-Veg)",  500),
        (water_river,  "Yamuna River (Non-Veg)", 250),
    ]:
        X, y = fetch_and_append(region, name, 0, n)
        X_all.extend(X); y_all.extend(y)

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, 'indian_vegetation_patches.npz')

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    np.savez(npz_path, X=X_all, y=y_all)

    logger.info(f"Saved {len(X_all)} patches → {npz_path}")
    logger.info(f"X shape: {X_all.shape}, y shape: {y_all.shape}")
    logger.info(f"Non-Veg: {(y_all==0).sum()}, Grass: {(y_all==1).sum()}, Trees: {(y_all==2).sum()}")

if __name__ == "__main__":
    main()