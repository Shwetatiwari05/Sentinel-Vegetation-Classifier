import os
import ee
import csv
import logging
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backend', '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SENTINEL_2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
MAX_CLOUD_COVER = 10
BAND_RED = 'B4'
BAND_NIR = 'B8'

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
    
    # Compute NDVI
    ndvi = image.normalizedDifference([BAND_NIR, BAND_RED]).rename('NDVI')
    return image.select([BAND_RED, BAND_NIR]).addBands(ndvi)

def sample_pixels(image: ee.Image, region: ee.Geometry, label: int, num_pixels: int = 1000):
    """Sample pixels randomly from the image within the region."""
    samples = image.sample(
        region=region,
        scale=10, # Sentinel-2 resolution is 10m
        numPixels=num_pixels,
        seed=42,
        geometries=False
    )
    
    # Extract feature values
    features = samples.getInfo()['features']
    data = []
    for feat in features:
        props = feat['properties']
        if BAND_RED in props and BAND_NIR in props and 'NDVI' in props:
            data.append([props[BAND_RED], props[BAND_NIR], props['NDVI'], label])
    return data

def main():
    initialize_gee()
    
    # --- POSITIVES (Label = 1: Grass/Vegetation) ---
    # 1. Northern Urban Park (Lodhi Gardens, Delhi)
    grass_delhi = ee.Geometry.Point([77.2197, 28.5933]).buffer(300) 
    # 2. Southern Urban Park (Cubbon Park, Bangalore)
    grass_blr = ee.Geometry.Point([77.5946, 12.9779]).buffer(300)
    # 3. Agricultural Fields (Punjab)
    grass_punjab = ee.Geometry.Point([75.8500, 30.9000]).buffer(500)
    # 4. Dense Tropical Forest (Western Ghats, Maharashtra)
    forest_ghats = ee.Geometry.Point([73.5500, 17.9200]).buffer(500)
    
    # --- NEGATIVES (Label = 0: Non-Grass/Barren/Urban/Water) ---
    # 5. Northern Dense Urban (Paharganj, Delhi)
    urban_delhi = ee.Geometry.Point([77.2120, 28.6430]).buffer(300)
    # 6. Coastal Dense Urban (Dharavi, Mumbai)
    urban_mumbai = ee.Geometry.Point([72.8562, 19.0402]).buffer(300)
    # 7. Barren Desert (Thar Desert, Rajasthan)
    desert_thar = ee.Geometry.Point([70.9000, 27.1000]).buffer(500)
    # 8. Water (Yamuna River)
    water_river = ee.Geometry.Point([77.2450, 28.6130]).buffer(200)

    start_date = '2024-01-01'
    end_date = '2024-05-30'
    
    all_data = []
    
    def fetch_and_append(region, name, label, num_pixels):
        logger.info(f"Fetching Sentinel-2 data for {name}...")
        img = get_image(region, start_date, end_date)
        data = sample_pixels(img, region, label=label, num_pixels=num_pixels)
        logger.info(f"Sampled {len(data)} pixels from {name}.")
        return data

    # Fetch all positives
    all_data.extend(fetch_and_append(grass_delhi, "Delhi Park (Grass)", 1, 800))
    all_data.extend(fetch_and_append(grass_blr, "Bangalore Park (Grass)", 1, 800))
    all_data.extend(fetch_and_append(grass_punjab, "Punjab Fields (Vegetation)", 1, 1000))
    all_data.extend(fetch_and_append(forest_ghats, "Western Ghats (Forest)", 1, 1000))
    
    # Fetch all negatives
    all_data.extend(fetch_and_append(urban_delhi, "Delhi Urban (Concrete)", 0, 800))
    all_data.extend(fetch_and_append(urban_mumbai, "Mumbai Urban (Concrete)", 0, 800))
    all_data.extend(fetch_and_append(desert_thar, "Thar Desert (Barren Sand)", 0, 1000))
    all_data.extend(fetch_and_append(water_river, "River (Water)", 0, 500))
    
    # Save to CSV
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'indian_vegetation.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Red', 'NIR', 'NDVI', 'Label'])
        writer.writerows(all_data)
        
    logger.info(f"Successfully saved {len(all_data)} total pixels to {csv_path}")

if __name__ == "__main__":
    main()
