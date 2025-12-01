# AgniSutra

AI-Driven Yield Optimization Platform for Oilseed Crops.

## Project Structure

- **frontend/**: Next.js application (React)
- **backend/**: FastAPI application (Python)

## Setup Instructions

### Frontend

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies (if not already installed):
   ```bash
   pnpm install
   ```
3. Run the development server:
   ```bash
   pnpm dev
   ```

### Backend

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Activate the virtual environment:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
3. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

## Technologies

- **Frontend:** Next.js, Tailwind CSS, TypeScript
- **Backend:** FastAPI, PostgreSQL, TensorFlow, Scikit-learn
