import csv
import os
import logging
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

logger = logging.getLogger(__name__)

CSV_FILE_PATH = "history.csv"

# Initialize Geocoder with a custom user agent (required by Nominatim terms of service)
geolocator = Nominatim(user_agent="sentinel-grass-classifier-internship")

def get_location_name(lat: float, lon: float) -> str:
    """
    Reverse geocodes the coordinates into a human-readable location name.
    """
    try:
        # Nominatim expects "lat, lon"
        location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True, timeout=5)
        if location:
            return location.address
        return "Unknown Location"
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        logger.warning(f"Geocoding service unavailable or timed out: {e}")
        return "Geocoding Unavailable"
    except Exception as e:
        logger.error(f"Unexpected error during reverse geocoding: {e}")
        return "Error resolving location"

def log_query_to_csv(lat: float, lon: float, date_start: str, date_end: str) -> None:
    """
    Reverse geocodes the coordinates and appends the query details to a CSV file.
    """
    location_name = get_location_name(lat, lon)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    file_exists = os.path.isfile(CSV_FILE_PATH)
    
    try:
        with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write the header if the file doesn't exist yet
            if not file_exists:
                writer.writerow(["Timestamp", "Location Name", "Latitude", "Longitude", "Date Start", "Date End"])
                
            writer.writerow([timestamp, location_name, lat, lon, date_start, date_end])
            logger.info(f"Successfully logged query for {location_name} to CSV.")
            
    except Exception as e:
        logger.error(f"Failed to write to CSV log: {e}")
