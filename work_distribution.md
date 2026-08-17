# Project Plan and Work Distribution
**Project Title:** Sentinel-2 Satellite Grass & Vegetation Classifier
**Team Size:** 2 Members (Member A & Member B)

## 1. Project Phases (Timeline & Milestones)

**Phase 1: Project Setup & Research (Day 1-2)**
- Finalize tech stack (Python, Streamlit/FastAPI, Google Earth Engine).
- Research Google Earth Engine (GEE) Python API for Sentinel-2 access and set up authentication.
- Set up GitHub repository and local development environments.

**Phase 2: Data Acquisition & Core Logic (Day 3-4)**
- Build the connection to Google Earth Engine.
- **Deliverable 1: Fetch Sentinel-2 image** using latitude, longitude, and date range.
- **Deliverable 2: Extract bands** (specifically Band 4 Red and Band 8 Near-Infrared).

**Phase 3: Machine Learning & Classification (Day 5-7)**
- Collect/format a training dataset (e.g., using labeled Sentinel-2 data like EuroSAT).
- **Deliverable 3: Compute NDVI** (Normalized Difference Vegetation Index) to feed into the model alongside raw spectral bands.
- **Deliverable 4: Train Random Forest** classifier to distinguish grass vs. non-grass.
- **Deliverable 5: Predict grass/not grass** on new unseen satellite images.
- **Deliverable 6: Visualize prediction** by generating a color-mapped visual output.

**Phase 4: API / Application Integration (Day 8-10)**
- Wrap the core processing logic in a web backend (e.g., FastAPI) or an interactive UI (e.g., Streamlit).
- Add error handling for cloudy images or invalid coordinates.

**Phase 5: Testing, Documentation, & Demo Prep (Day 11-12)**
- Test the system with known grassy and non-grassy coordinates (e.g., parks vs. deserts).
- Write the final README documentation (Architecture, API specs, Setup instructions).
- Prepare the presentation and live demo.

---

## 2. Work Distribution

### Member A: Backend, API Integration, & Architecture
*Focuses on fetching the data and serving the application.*
* **Data Fetching:** Implement the Google Earth Engine Python API (`earthengine-api`) to retrieve Sentinel-2 imagery.
* **Backend Development:** Build the REST API (FastAPI) or Streamlit framework to take user inputs (coords/dates).
* **Infrastructure:** Manage the GitHub repo, Python environment, and any Docker configurations.
* **Documentation:** Write the Architecture, Tech Stack, and Setup Instructions in the README.

### Member B: Machine Learning & Geospatial Processing
*Focuses on handling the satellite data and training the ML model.*
* **Data Processing:** Handle loading Google Earth Engine exports into Python arrays and formatting data for the ML model.
* **ML Model Training:** Build and train the Machine Learning classifier (Random Forest/CNN) using libraries like `scikit-learn` or `PyTorch`.
* **Visualization:** Generate the output imagery (PNG/JPEG) showing the model's predictions with color-coded grass areas.
* **Testing & Tuning:** Evaluate model accuracy on a validation set and tune hyperparameters to reduce false positives (like trees or water).
