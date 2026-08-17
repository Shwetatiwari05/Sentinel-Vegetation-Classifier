import logging
from datetime import date
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from core import process_satellite_data
from utils import log_query_to_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Initialize FastAPI app
app = FastAPI(
    title="Sentinel-2 Grass Classifier API",
    description="API for fetching satellite imagery and classifying grass vs. non-grass",
    version="1.0.0"
)

import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- CORS & STATIC FILES ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount the outputs directory so the frontend can load the images
outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml', 'outputs'))
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

# --- SECURITY HARDENING: Rate Limiting ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- SECURITY HARDENING: Reject oversized payloads (> 1MB) ---
MAX_PAYLOAD_SIZE = 1048576 # 1 MB in bytes

@app.middleware("http")
async def enforce_payload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_PAYLOAD_SIZE:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"detail": "Payload too large. Maximum allowed size is 1MB."}
        )
    return await call_next(request)

from typing import Union
import re

# Pydantic models for Input Validation
class PredictionRequest(BaseModel):
    latitude: Union[float, str] = Field(..., description="Latitude (Decimal Float or DMS String like 40°46'52.81\"N)")
    longitude: Union[float, str] = Field(..., description="Longitude (Decimal Float or DMS String like 73°57'57.12\"W)")
    date_start: date = Field(..., description="Start date in YYYY-MM-DD format")
    date_end: date = Field(..., description="End date in YYYY-MM-DD format")
    model_type: str = Field(default="random_forest", description="Machine learning model to use (random_forest, cnn_2d, or svm)")

    # SECURITY HARDENING: Reject unknown or extra fields in the JSON payload
    model_config = {
        "extra": "forbid"
    }

class PredictionResponse(BaseModel):
    status: str
    message: str
    prediction: str
    is_grass: bool
    confidence: float
    grass_percentage: float
    ndvi_mean: float
    visualization_path: str
    ndvi_thumbnail_url: str
    coordinates: dict

def parse_coordinate(coord: Union[str, float]) -> float:
    """Converts DMS string (e.g., 40°46'52.81\"N) to a decimal float."""
    if isinstance(coord, (float, int)):
        return float(coord)
    
    # SECURITY HARDENING: Prevent ReDoS by capping string length
    coord_str = str(coord).upper().strip()
    if len(coord_str) > 50:
        raise ValueError("Coordinate string is too long.")
        
    # Match format like 40°46'52.81"N
    match = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", coord_str)
    if match:
        degrees = float(match.group(1))
        minutes = float(match.group(2))
        seconds = float(match.group(3))
        direction = match.group(4)
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if direction in ['S', 'W']:
            decimal = -decimal
        return decimal
    return float(coord_str)

@app.get("/health", status_code=status.HTTP_200_OK)
@limiter.limit("10/minute")
async def health_check(request: Request):
    """Health check endpoint to ensure API is running."""
    return {"status": "healthy", "service": "sentinel-classifier-api"}

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
@limiter.limit("5/15minute")  # Extremely strict: Max 5 attempts per 15 mins
def predict_grass(request: Request, payload: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Accepts coordinates and dates, triggers the core data pipeline, 
    and returns the classification results.
    Logs the query to a CSV file in the background.
    """
    try:
        parsed_lat = parse_coordinate(payload.latitude)
        parsed_lon = parse_coordinate(payload.longitude)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid coordinate format. Use decimal (40.78) or DMS (40°46'52.81\"N)."
        )

    logger.info(f"Received prediction request for coordinates: ({parsed_lat}, {parsed_lon})")
    
    # Send the logging task to the background so it doesn't block the API response
    background_tasks.add_task(
        log_query_to_csv,
        lat=parsed_lat,
        lon=parsed_lon,
        date_start=payload.date_start.isoformat(),
        date_end=payload.date_end.isoformat()
    )
    
    try:
        # Separate Business Logic from the Route Handler
        # We pass validation-checked data to our core processing module
        result = process_satellite_data(
            lat=parsed_lat,
            lon=parsed_lon,
            date_start=payload.date_start.isoformat(),
            date_end=payload.date_end.isoformat(),
            model_type=payload.model_type
        )
        return result
        
    except ValueError as ve:
        # Handle known business logic errors (e.g., no images found)
        logger.warning(f"Validation/Data error during processing: {ve}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve)
        )
    except Exception as e:
        # Never expose internal errors publicly. Log the real error, return generic 500.
        logger.error(f"Unexpected internal server error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the satellite data."
        )

# Instructions to run:
# uvicorn api:app --reload
