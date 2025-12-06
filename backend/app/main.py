from contextlib import asynccontextmanager
import pickle
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .routers import auth, yield_pred, iot, admin, krishi_saathi
from .ml import ml_models
from .manager import manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    try:
        with open("backend/app/model.pkl", "rb") as f:
            ml_models["yield_model"] = pickle.load(f)
        print("ML Model loaded successfully.")
    except FileNotFoundError:
        # Fallback for local dev if running from inside backend/app or similar
        try:
            with open("app/model.pkl", "rb") as f:
                ml_models["yield_model"] = pickle.load(f)
            print("ML Model loaded successfully (fallback path).")
        except Exception as e:
            print(f"Warning: Could not load ML model: {e}")
            ml_models["yield_model"] = None
    except Exception as e:
        print(f"Error loading ML model: {e}")
        ml_models["yield_model"] = None
    
    yield
    
    # Clean up
    ml_models.clear()

app = FastAPI(title="AgniSutra API", version="1.0.0", lifespan=lifespan)

# Configure CORS - allow web frontend and common mobile emulator hosts
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://10.0.2.2:3000",  # Android emulator -> host machine
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
def read_root():
    return {"message": "Welcome to AgniSutra API (app.main)"}


@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(yield_pred.router, prefix="/yield", tags=["yield"])
app.include_router(iot.router, prefix="/iot", tags=["iot"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(krishi_saathi.router, prefix="/krishi", tags=["krishi"])

