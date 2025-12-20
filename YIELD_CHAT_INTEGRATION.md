# AgniSutra: Yield Prediction & Krishi Saathi Integration Guide

This guide details how to integrate the **Yield Prediction** (ML) and **Krishi Saathi Chat** (LLM) features into your frontend application.

## 1. Yield Prediction

This endpoint uses a CatBoost ML model to predict crop yield based on soil, weather, and crop parameters.

### Endpoint

- **URL**: `POST /yield/predict`
- **Auth**: Required (`Authorization: Bearer <token>`)
- **Content-Type**: `application/json`

### Request Body

You must collect these inputs from the user or fetch them from weather/soil APIs.

```json
{
 {
  "crop": "sunflower",
  "maturity_days": 120,
  "mean_temp_gs_C": 0,
  "temp_flowering_C": 0,
  "seasonal_rain_mm": 0,
  "rain_flowering_mm": 0,
  "humidity_mean_pct": 0,
  "soil_pH": 0,
  "clay_pct": 0,
  "soil_N_status_kg_ha": 0,
  "soil_P_status_kg_ha": 0,
  "soil_K_status_kg_ha": 0,
  "irrigation_events": 2,
  "ndvi_flowering": 0,
  "ndvi_peak": 0,
  "ndvi_veg_slope": 0,
  "seed_moisture_pct": 40,
  "fert_N_kg_ha": 0,
  "fert_P_kg_ha": 0,
  "fert_K_kg_ha": 0
}
}
```

### Response

```json
{
  "predicted_yield": 1.85,
  "unit": "tons/hectare"
}
```

---

## 2. Krishi Saathi Chat (AI Advisor)

This endpoint provides an AI agronomy advisor that uses the context of the yield prediction to give specific advice.

### Endpoint

- **URL**: `POST /krishi/chat`
- **Auth**: Required (`Authorization: Bearer <token>`)
- **Content-Type**: `application/json`

### Request Body

You need to pass the **entire input payload** used for yield prediction inside `yield_data`. This allows the AI to "see" the farm conditions.

```json
{
  "session_id": "unique_session_id_123",
  "query": "My yield is low. How can I improve it?",
  "language": "en",
  "yield_data": {
      "features": {
          "crop": "soybean",
          "soil_pH": 6.5,
          "soil_N_status_kg_ha": 150.0,
          ... (include all fields from Yield Prediction input) ...
      },
      "predicted_yield": 1.85
  }
}
```

- **`session_id`**: Generate a unique ID for the chat session (or use User ID) to maintain conversation history.
- **`language`**: "en", "hi" (Hindi), "or" (Odia), or "auto".
- **`yield_data`**: Must contain a `features` object (the inputs) and the `predicted_yield` (the output from the previous step).

### Response

```json
{
  "response": "Based on your soil nitrogen levels (150 kg/ha), your soybean yield is slightly below potential. I recommend..."
}
```

---

## 3. Frontend Flow Example (Flutter)

1.  **Step 1: Predict Yield**

    - User fills form -> App calls `POST /yield/predict`.
    - App receives `predicted_yield`.
    - **Store** the input form data and the result in a variable (e.g., `currentFarmContext`).

2.  **Step 2: Chat with Advisor**
    - User opens Chat screen.
    - User types: "What fertilizer should I use?"
    - App calls `POST /krishi/chat` with:
      - `query`: "What fertilizer should I use?"
      - `yield_data`: `currentFarmContext`
    - App displays the AI's response.

### Flutter Code Snippet (Chat)

```dart
Future<void> sendMessage(String message) async {
  final response = await dio.post(
    '$baseUrl/krishi/chat',
    data: {
      "session_id": "user_${currentUser.id}",
      "query": message,
      "language": "en",
      "yield_data": {
        "features": yieldInputData, // Map<String, dynamic> of form inputs
        "predicted_yield": predictedYieldValue // double
      }
    },
    options: Options(headers: {'Authorization': 'Bearer $token'}),
  );

  print(response.data['response']);
}
```
