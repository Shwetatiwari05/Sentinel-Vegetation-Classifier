import os
import ee
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gee_auth")

def authenticate_and_initialize() -> None:
    """
    Authenticates the user with Google Earth Engine and initializes the API.
    This script should be run once interactively.
    """
    project_id = os.getenv("EE_PROJECT_ID")
    if not project_id:
        logger.error("EE_PROJECT_ID environment variable is not set. Please create a .env file and add it.")
        return

    try:
        logger.info(f"Attempting to initialize Google Earth Engine with project: {project_id}")
        ee.Initialize(project=project_id)
        logger.info("Google Earth Engine initialized successfully.")
    except Exception as e:
        logger.warning("Initialization failed. Attempting to authenticate...")
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            logger.info("Authentication and initialization successful.")
        except Exception as auth_e:
            logger.error(f"Failed to authenticate with GEE: {auth_e}")
            raise

if __name__ == "__main__":
    logger.info("Starting GEE Authentication process.")
    authenticate_and_initialize()
