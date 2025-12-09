# AgniSutra – AI-Driven Yield Optimization Platform for Oilseed Crops

**Team Name:** Sadhguna  
**Problem Statement ID:** SIH252XX  
**Theme:** Agriculture, FoodTech & Rural Development

---

## 📖 Project Overview

**AgniSutra** is an AI-enabled platform designed to empower oilseed farmers with real-time yield forecasts, personalized advisories, and data-driven insights. By integrating machine learning, weather data, satellite imagery, and IoT sensor networks, AgniSutra aims to optimize crop yields, reduce import dependency, and stabilize farmer income.

### 🚀 Key Features

- **Yield Prediction Engine:** AI models (Linear Regression/XGBoost) to forecast crop yield based on soil nutrients, rainfall, and temperature.
- **Real-Time IoT Monitoring:** Ingests data from soil moisture sensors to track farm health.
- **Smart Alerts:** Real-time WebSocket alerts for critical conditions (e.g., low soil moisture).
- **Personalized Advisory:** Recommendations for irrigation and resource management.
- **Admin Dashboard:** Aggregated view of total farms, users, and yield statistics with geospatial visualization.
- **Secure Authentication:** JWT-based secure login and registration for farmers and administrators.

---

## 🛠️ Tech Stack

### Frontend

- **Framework:** Next.js (React)
- **Styling:** Tailwind CSS
- **Language:** TypeScript

### Backend

- **Framework:** FastAPI (Python)
- **Database:** PostgreSQL (via SQLAlchemy)
- **ML Libraries:** Scikit-learn, NumPy, Pandas
- **Real-Time:** WebSockets
- **Authentication:** OAuth2 with JWT (Passlib, Python-Jose)

### IoT & Data

- **Simulation:** Python-based IoT simulator
- **Data Sources:** User inputs, Simulated Sensors (extensible to IMD/ISRO APIs)

---

## 📂 Project Structure

```
agnisutra/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── routers/         # API Endpoints (Auth, IoT, Yield, Admin)
│   │   ├── models.py        # SQLAlchemy Database Models
│   │   ├── schemas.py       # Pydantic Data Schemas
│   │   ├── database.py      # DB Connection (PostgreSQL)
│   │   ├── main.py          # App Entrypoint & WebSocket Manager
│   │   ├── manager.py       # WebSocket Connection Manager
│   │   └── ml.py            # ML Model Loader
│   ├── requirements.txt     # Python Dependencies
│   ├── simulator.py         # IoT Sensor Simulator
│   └── .env                 # Environment Variables
│
├── frontend/                # Next.js Frontend
│   ├── app/                 # App Router Pages
│   ├── public/              # Static Assets
│   └── package.json         # Node Dependencies
│
└── README.md                # Project Documentation
```

---

## ⚙️ Setup Instructions

### Prerequisites

- **Python 3.9+**
- **Node.js 18+** & **pnpm**
- **PostgreSQL** (Running locally or via Docker)

### 1. Database Setup

Ensure PostgreSQL is running and create a database named `agnisutra`.

```sql
CREATE DATABASE agnisutra;
```

### 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure Environment Variables:
Create a `.env` file in `backend/` (or edit the existing one) and set your DB credentials:

```ini
DATABASE_URL=postgresql://postgres:password@localhost:5432/agnisutra
SECRET_KEY=your_secret_key
```

Run the Server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

_The API will be available at `http://127.0.0.1:8000`_

### 3. Frontend Setup

Navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
pnpm install
```

Run the development server:

```bash
pnpm dev
```

_The app will be available at `http://localhost:3000`_

---

## 🧪 Testing & Usage

### API Documentation (Swagger UI)

Visit **`http://127.0.0.1:8000/docs`** to explore and test all API endpoints interactively.

### IoT Simulation

To simulate live sensor data flowing into the system:

1.  Ensure the backend is running.
2.  Open a new terminal in `backend/`.
3.  Run the simulator:
    ```bash
    python simulator.py
    ```
4.  Observe real-time logs in the backend console or connect via WebSocket to receive alerts.

### ML Model

The backend loads a pre-trained model (`model.pkl`) at startup. If not found, it falls back to a heuristic formula.
To retrain/generate the dummy model:

```bash
# Run this python one-liner in the backend directory
python -c "import pickle; from sklearn.linear_model import LinearRegression; import numpy as np; X = np.array([[100, 500, 25], [150, 600, 30]]); y = np.array([2000, 2500]); model = LinearRegression().fit(X, y); pickle.dump(model, open('app/model.pkl', 'wb'))"
```

---

## 👥 Team Sadhguna

- **Ashutosh Mishra** - Lead Developer
- _(Add other team members here)_

---

_Built for Smart India Hackathon 2025_
