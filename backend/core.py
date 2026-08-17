import os
import sys
import ee
import logging
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from typing import Dict, Any

# Add ML Work directory to path to import the new ML scripts
ML_WORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml'))
sys.path.append(ML_WORK_DIR)

from ml_predict import predict_grass
from visualize import generate_prediction_visual

logger = logging.getLogger(__name__)

# Constants
SENTINEL_2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
MAX_CLOUD_COVER = 20
BAND_RED = 'B4'
BAND_NIR = 'B8'

def initialize_gee() -> None:
    project_id = os.getenv("EE_PROJECT_ID")
    if not project_id:
        logger.error("EE_PROJECT_ID environment variable is not set. Please create a .env file and add it.")
        raise ValueError("Missing EE_PROJECT_ID in environment variables.")

    try:
        ee.Initialize(project=project_id)
        logger.info(f"GEE initialized successfully in core with project: {project_id}")
    except Exception as e:
        logger.error(f"Failed to initialize GEE. Please run gee_auth.py first. Error: {e}")
        raise

def get_sentinel2_image(lat: float, lon: float, date_start: str, date_end: str) -> ee.Image:
    logger.info(f"Fetching Sentinel-2 image for coords: ({lat}, {lon}) between {date_start} and {date_end}")
    try:
        point = ee.Geometry.Point([lon, lat])
        collection = (ee.ImageCollection(SENTINEL_2_COLLECTION)
                      .filterBounds(point)
                      .filterDate(date_start, date_end)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER)))
        
        count = collection.size().getInfo()
        if count == 0:
            raise ValueError("No images found for the given parameters.")
            
        logger.info(f"Found {count} images. Creating median composite.")
        return collection.median().setDefaultProjection(crs='EPSG:3857', scale=10)
    except Exception as e:
        logger.error(f"Error fetching Sentinel-2 image: {e}")
        raise

def extract_bands_and_compute_ndvi(image: ee.Image) -> ee.Image:
    try:
        bands = image.select([BAND_RED, BAND_NIR])
        ndvi = image.normalizedDifference([BAND_NIR, BAND_RED]).rename('NDVI')
        return bands.addBands(ndvi)
    except Exception as e:
        logger.error(f"Error computing NDVI: {e}")
        raise

def get_band_arrays(image: ee.Image, lat: float, lon: float) -> tuple:
    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(1000)

        red_array = np.array(
            image.select(BAND_RED).sampleRectangle(region=region, defaultValue=0).get(BAND_RED).getInfo()
        )
        nir_array = np.array(
            image.select(BAND_NIR).sampleRectangle(region=region, defaultValue=0).get(BAND_NIR).getInfo()
        )

        logger.info(f"Band arrays downloaded — Red shape: {red_array.shape}, NIR shape: {nir_array.shape}")
        return red_array, nir_array
    except Exception as e:
        logger.error(f"Error downloading band arrays from GEE: {e}")
        raise

def process_satellite_data(lat: float, lon: float, date_start: str, date_end: str, model_type: str = "random_forest") -> Dict[str, Any]:
    initialize_gee()
    
    # 1. Fetch Image
    image = get_sentinel2_image(lat, lon, date_start, date_end)
    
    # 2. Extract bands (GEE-side)
    feature_image = extract_bands_and_compute_ndvi(image)
    
    # 3. Download band pixel arrays from GEE to numpy
    red_array, nir_array = get_band_arrays(image, lat, lon)
    
    # 4. Run Pixel-Level ML model prediction
    ml_result = predict_grass(red_array, nir_array, model_type=model_type)
    
    # 5. Generate color-coded PNG visualization using the True 2D mask
    output_image_path = generate_prediction_visual(
        ndvi_array=ml_result['ndvi_array'],
        grass_mask_2d=ml_result['grass_mask_2d'],
        grass_percentage=ml_result['grass_percentage'],
        confidence=ml_result['confidence'],
        lat=lat,
        lon=lon
    )
    
    # 6. Generate NDVI thumbnail URL from GEE (for API reference)
    point = ee.Geometry.Point([lon, lat])
    buffer_region = point.buffer(1000)
    ndvi_url = feature_image.select('NDVI').getThumbURL({
        'min': -1,
        'max': 1,
        'dimensions': 512,
        'region': buffer_region,
        'palette': ['blue', 'white', 'green']
    })
    
    # Format the image path as a public URL for the Frontend
    filename = os.path.basename(output_image_path)
    public_image_url = f"http://127.0.0.1:8000/outputs/{filename}"
    
    return {
        "status": "success",
        "message": "Satellite data processed and pixel-level ML prediction complete.",
        "prediction": ml_result['prediction'],
        "is_grass": ml_result['is_grass'],
        "confidence": ml_result['confidence'],
        "grass_percentage": ml_result['grass_percentage'],
        "ndvi_mean": ml_result['ndvi_mean'],
        "visualization_path": public_image_url,
        "ndvi_thumbnail_url": ndvi_url,
        "coordinates": {"lat": lat, "lon": lon}
    }
