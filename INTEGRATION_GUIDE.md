# AgniSutra Integration Guide

This guide explains how to connect your Frontend (Mobile/Web) and IoT devices to the AgniSutra Backend.

## 1. General Configuration

### Base URL

Since the backend runs on your local machine, you cannot use `localhost` from other devices (like a phone or ESP32). You must use your PC's Local IP Address.

1.  **Find your IP**: Run `ipconfig` (Windows) or `ifconfig` (Mac/Linux) in a terminal. Look for **IPv4 Address** (e.g., `192.168.1.5`).
2.  **Backend URL**: `http://<YOUR_IP>:8000`

---

## 2. Frontend Integration (Flutter Example)

### A. Authentication

**1. Login**

- **Endpoint**: `POST /auth/login`
- **Content-Type**: `application/x-www-form-urlencoded`
- **Body**: `username=<email>&password=<password>`
- **Response**: `{"access_token": "...", "token_type": "bearer"}`

**Flutter Code (using Dio):**

```dart
final dio = Dio();
final response = await dio.post(
  'http://192.168.1.5:8000/auth/login',
  data: {'username': email, 'password': password},
  options: Options(contentType: Headers.formUrlEncodedContentType),
);
String token = response.data['access_token'];
// Save 'token' securely
```

**2. Register**

- **Endpoint**: `POST /auth/register`
- **Body (JSON)**:
  ```json
  {
    "name": "Farmer John",
    "email": "john@example.com",
    "password": "securepassword"
  }
  ```

**3. Logout**

- **Endpoint**: `POST /auth/logout`
- **Headers**: `Authorization: Bearer <token>`
- **Action**: Call this API, then delete the token from your app's storage.

### B. Authenticated Requests

For all other endpoints (Profile, Yield Prediction, Chat), you must send the token in the header.

**Header Format**: `Authorization: Bearer <your_access_token>`

**Example: Generic Authenticated Request**

```dart
final response = await dio.get(
  'http://192.168.1.5:8000/some/protected/route',
  options: Options(headers: {'Authorization': 'Bearer $token'}),
);
```

### C. User Profile Management

**1. Get Profile**

- **Endpoint**: `GET /auth/me`
- **Headers**: `Authorization: Bearer <token>`

```dart
try {
  final response = await dio.get(
    'http://192.168.1.5:8000/auth/me',
    options: Options(headers: {'Authorization': 'Bearer $token'}),
  );
  print(response.data);
  // Response: { "id": 1, "name": "...", "profile_photo": "/static/uploads/...", ... }
} catch (e) {
  print('Error fetching profile: $e');
}
```

**2. Update Profile**

- **Endpoint**: `PUT /auth/me`
- **Headers**: `Authorization: Bearer <token>`
- **Body**: JSON with fields to update (name, city, latitude, longitude).

```dart
try {
  final response = await dio.put(
    'http://192.168.1.5:8000/auth/me',
    data: {
      "name": "New Name",
      "city": "New City",
      "latitude": 22.5,
      "longitude": 85.2
    },
    options: Options(headers: {'Authorization': 'Bearer $token'}),
  );
  print('Profile updated: ${response.data}');
} catch (e) {
  print('Error updating profile: $e');
}
```

**3. Upload Profile/Cover Photo**

- **Endpoint**: `POST /auth/me/profile-photo` or `/auth/me/cover-photo`
- **Headers**: `Authorization: Bearer <token>`
- **Body**: Multipart Form Data with key `file`.

```dart
// Pick file using image_picker package first
// File imageFile = ...;

String fileName = imageFile.path.split('/').last;
FormData formData = FormData.fromMap({
  "file": await MultipartFile.fromFile(imageFile.path, filename: fileName),
});

try {
  final response = await dio.post(
    'http://192.168.1.5:8000/auth/me/profile-photo', // or /auth/me/cover-photo
    data: formData,
    options: Options(headers: {'Authorization': 'Bearer $token'}),
  );
  print('Photo uploaded: ${response.data}');
  // The response contains the new URL, e.g., "profile_photo": "/static/uploads/..."
  // Construct full URL: http://192.168.1.5:8000/static/uploads/...
} catch (e) {
  print('Error uploading photo: $e');
}
```

### D. Krishi Saathi Chat (LLM)

- **Endpoint**: `POST /krishi/chat`
- **Body**: `{"message": "How to grow wheat?", "history": []}`
- **Response**: `{"response": "To grow wheat..."}`

---

## 3. IoT Integration (ESP32 / NodeMCU)

Your IoT device will send sensor data to the backend.

### A. Data Endpoint

- **URL**: `http://<YOUR_IP>:8000/iot/update`
- **Method**: `POST`
- **Content-Type**: `application/json`

### B. Payload Format

```json
{
  "user_id": 1,
  "moisture": 45.5,
  "nitrogen": 120,
  "phosphorus": 40,
  "potassium": 60,
  "temperature": 28.5,
  "humidity": 65.0
}
```

_Note: `user_id` should be hardcoded or configured on the device for the specific farmer._

### C. Example ESP32 Code (Arduino IDE)

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ======================= USER CONFIGURATION =======================
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Updated URL: Ensure this matches your FastAPI endpoint
const char* serverUrl = "http://192.168.1.5:8000/iot/update";

#define MOISTURE_PIN 34    // Analog Pin (AO)
#define MOISTURE_POWER 15  // Digital Pin (VCC)

// Calibration for Resistive Sensor
const int dryValue = 4095;
const int wetValue = 1728;

void setup() {
  Serial.begin(115200);
  pinMode(MOISTURE_POWER, OUTPUT);
  digitalWrite(MOISTURE_POWER, LOW);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

int readSoilMoisture() {
  digitalWrite(MOISTURE_POWER, HIGH);
  delay(100);
  int val = analogRead(MOISTURE_PIN);
  digitalWrite(MOISTURE_POWER, LOW);
  return val;
}

void loop() {
  // 1. Read Moisture
  int rawMoisture = readSoilMoisture();
  int moisturePct = map(rawMoisture, dryValue, wetValue, 0, 100);
  moisturePct = constrain(moisturePct, 0, 100);

  Serial.print("Moisture: "); Serial.print(moisturePct); Serial.println("%");

  // 2. Simulate NPK Values (Randomized for Demo)
  float nitrogen = random(20, 50) + (random(0, 100) / 10.0);
  float phosphorus = random(10, 40) + (random(0, 100) / 10.0);
  float potassium = random(30, 60) + (random(0, 100) / 10.0);

  // 3. Send Data via HTTP POST
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    // JSON Payload
    StaticJsonDocument<200> doc;

    // ⚠️ IMPORTANT: Use the unique Device ID (e.g., MAC Address)
    // This must match the 'device_id' registered for the user in the database
    doc["device_id"] = "A1:B2:C3:D4:E5:F6";

    doc["moisture"] = moisturePct;
    doc["nitrogen"] = nitrogen;
    doc["phosphorus"] = phosphorus;
    doc["potassium"] = potassium;

    String requestBody;
    serializeJson(doc, requestBody);

    int httpResponseCode = http.POST(requestBody);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("Server Response: " + response);
    } else {
      Serial.print("HTTP Error: "); Serial.println(httpResponseCode);
    }
    http.end();
  } else {
    Serial.println("Wi-Fi Disconnected!");
  }

  delay(5000); // Send every 5 seconds
}
```

## 4. Fetching Sensor Data (Frontend)

To see the data sent by the IoT device, the frontend must request it using the authenticated user's token.

### A. Get My Logs

- **Endpoint**: `GET /iot/my-logs`
- **Headers**: `Authorization: Bearer <token>`
- **Response**: List of sensor readings for the logged-in user.

### B. Get Latest Reading

- **Endpoint**: `GET /iot/latest`
- **Headers**: `Authorization: Bearer <token>`
- **Response**: The single most recent reading.

---

## 5. Real-time Alerts (WebSockets)

The backend broadcasts alerts as JSON. The frontend must parse this JSON and check if the `user_id` matches the logged-in user.

### A. WebSocket URL

- **URL**: `ws://<YOUR_IP>:8000/ws/alerts`

### B. Flutter Implementation (web_socket_channel)

```dart
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';

void listenToAlerts(int myUserId) {
  final channel = WebSocketChannel.connect(
    Uri.parse('ws://192.168.1.5:8000/ws/alerts'),
  );

  channel.stream.listen((message) {
    try {
      final data = jsonDecode(message);
      // Check if the alert is for THIS user
      if (data['user_id'] == myUserId) {
        List<dynamic> alerts = data['messages'];
        String alertText = alerts.join("\n");
        print("New Alert for Me: $alertText");
        // Show Notification
      }
    } catch (e) {
      print("Error parsing alert: $e");
    }
  });
}
```

## 4. Real-time Alerts (WebSockets)

The backend automatically checks the sensor data every time it receives it (every 5 seconds). If moisture is low (< 30) or temperature is high (> 40), it broadcasts an alert immediately.

### A. WebSocket URL

- **URL**: `ws://<YOUR_IP>:8000/ws/alerts`

### B. Flutter Implementation (web_socket_channel)

Add `web_socket_channel: ^2.4.0` to `pubspec.yaml`.

```dart
import 'package:web_socket_channel/web_socket_channel.dart';

void listenToAlerts() {
  // Replace with your PC's IP
  final channel = WebSocketChannel.connect(
    Uri.parse('ws://192.168.1.5:8000/ws/alerts'),
  );

  channel.stream.listen((message) {
    print("New Alert Received: $message");
    // Show a local notification or a SnackBar here
    // Example: ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }, onError: (error) {
    print("WebSocket Error: $error");
  });
}
```
