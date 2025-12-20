# Agnisutra API Documentation

This document provides details about the API endpoints available in the Agnisutra backend.

## Base URL

`http://localhost:8000`

## Authentication

### Register User

- **URL**: `/auth/register`
- **Method**: `POST`
- **Description**: Registers a new user.
- **Request Body**:
  ```json
  {
    "name": "John Doe",
    "email": "john@example.com",
    "password": "securepassword"
  }
  ```
- **Response**:
  ```json
  {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "role": "farmer"
  }
  ```

### Login User

- **URL**: `/auth/login`
- **Method**: `POST`
- **Description**: Authenticates a user and returns an access token.
- **Request Body** (Form Data):
  - `username`: User's email (e.g., `john@example.com`)
  - `password`: User's password
- **Response**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer"
  }
  ```

### Get User Profile

- **URL**: `/auth/me`
- **Method**: `GET`
- **Description**: Retrieves the profile of the currently logged-in user.
- **Headers**:
  - `Authorization`: `Bearer <access_token>`
- **Response**:
  ```json
  {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "role": "farmer",
    "city": "Rourkela",
    "latitude": 22.26,
    "longitude": 84.85,
    "profile_photo": "/static/uploads/profile_1_123456.jpg",
    "cover_photo": "/static/uploads/cover_1_123456.jpg"
  }
  ```

### Update User Profile

- **URL**: `/auth/me`
- **Method**: `PUT`
- **Description**: Updates the profile details of the currently logged-in user.
- **Headers**:
  - `Authorization`: `Bearer <access_token>`
- **Request Body**:
  ```json
  {
    "name": "John Updated",
    "city": "Bhubaneswar",
    "latitude": 20.29,
    "longitude": 85.82
  }
  ```
- **Response**:
  ```json
  {
    "id": 1,
    "name": "John Updated",
    "email": "john@example.com",
    "role": "farmer",
    "city": "Bhubaneswar",
    "latitude": 20.29,
    "longitude": 85.82,
    "profile_photo": null,
    "cover_photo": null
  }
  ```

### Upload Profile Photo

- **URL**: `/auth/me/profile-photo`
- **Method**: `POST`
- **Description**: Uploads a profile photo for the user.
- **Headers**:
  - `Authorization`: `Bearer <access_token>`
- **Request Body** (Multipart Form Data):
  - `file`: The image file to upload.
- **Response**:
  ```json
  {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "role": "farmer",
    "profile_photo": "/static/uploads/profile_1_123456.jpg",
    ...
  }
  ```

### Upload Cover Photo

- **URL**: `/auth/me/cover-photo`
- **Method**: `POST`
- **Description**: Uploads a cover photo for the user.
- **Headers**:
  - `Authorization`: `Bearer <access_token>`
- **Request Body** (Multipart Form Data):
  - `file`: The image file to upload.
- **Response**:
  ```json
  {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "role": "farmer",
    "cover_photo": "/static/uploads/cover_1_123456.jpg",
    ...
  }
  ```

## IoT Data

### Update Sensor Data

- **URL**: `/iot/update`
- **Method**: `POST`
- **Description**: Receives sensor data from IoT devices.
- **Request Body**:
  ```json
  {
    "moisture": 45.5,
    "nitrogen": 50.0,
    "phosphorus": 30.0,
    "potassium": 20.0
  }
  ```
- **Response**:
  ```json
  {
    "moisture": 45.5,
    "nitrogen": 50.0,
    "phosphorus": 30.0,
    "potassium": 20.0,
    "id": 101,
    "timestamp": "2023-10-27T10:00:00"
  }
  ```

## Yield Prediction

### Predict Yield

- **URL**: `/yield/predict`
- **Method**: `POST`
- **Description**: Predicts crop yield based on environmental factors.
- **Request Body**:
  ```json
  {
    "nitrogen": 120.0,
    "rainfall": 200.0,
    "temp": 30.0
  }
  ```
- **Response**:
  ```json
  {
    "predicted_yield": 4.5,
    "unit": "tons/hectare"
  }
  ```

## Admin

### Get Statistics

- **URL**: `/admin/stats`
- **Method**: `GET`
- **Description**: Retrieves system statistics (admin only).
- **Headers**:
  - `Authorization`: `Bearer <access_token>`
- **Response**:
  ```json
  {
    "total_users": 150,
    "active_sensors": 45,
    "alerts_today": 3
  }
  ```

## Krishi Saathi (Advanced Yield & Advisory)

### Predict Yield (Advanced)

- **URL**: `/krishi/predict`
- **Method**: `POST`
- **Description**: Predicts yield using the advanced CatBoost model and provides weather-based alerts.
- **Request Body**:
  ```json
  {
    "crop": "soybean",
    "variety_group": "medium",
    "maturity_days": 110,
    "base_yield_potential_t_ha": 2.0,
    "mean_temp_gs_C": 26.5,
    "temp_flowering_C": 28.0,
    "seasonal_rain_mm": 600.0,
    "rain_flowering_mm": 100.0,
    "humidity_mean_pct": 70.0,
    "solar_MJ_m2_day": 18.0,
    "soil_pH": 6.5,
    "soil_oc_pct": 0.8,
    "clay_pct": 30.0,
    "soil_N_status_kg_ha": 150.0,
    "soil_P_status_kg_ha": 30.0,
    "soil_K_status_kg_ha": 200.0,
    "soil_texture": "clay",
    "soil_depth_cm": 100.0,
    "season_length_days": 110,
    "plant_density_plants_m2": 30.0,
    "irrigation_events": 2,
    "herbicide_apps": 1,
    "insecticide_apps": 1,
    "fungicide_apps": 0,
    "weed_pressure_index": 1.0,
    "pest_pressure_index": 1.0,
    "disease_pressure_index": 1.0,
    "ndvi_early": 0.3,
    "ndvi_flowering": 0.7,
    "ndvi_peak": 0.8,
    "ndvi_late": 0.5,
    "ndvi_veg_slope": 0.01,
    "seed_moisture_pct": 10.0,
    "fert_N_kg_ha": 80.0,
    "fert_P_kg_ha": 40.0,
    "fert_K_kg_ha": 20.0,
    "sowing_doy": 170
  }
  ```
- **Response**:
  ```json
  {
    "predicted_yield": 1.85,
    "unit": "t/ha",
    "alerts": ["✅ No major weather red-flags detected."],
    "benchmark_comparison": "Typical: 1.80 t/ha. Your yield is 102.8% of benchmark."
  }
  ```

### Chat with Advisor

- **URL**: `/krishi/chat`
- **Method**: `POST`
- **Description**: Chat with the Krishi Saathi LLM advisor.
- **Request Body**:
  ```json
  {
    "session_id": "user123",
    "query": "How can I improve my soybean yield?",
    "yield_context": {
      "crop": "soybean",
      "yield": 1.85,
      "unit": "t/ha",
      "features": { ... }
    },
    "language": "English"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "Based on your soil parameters..."
  }
  ```
