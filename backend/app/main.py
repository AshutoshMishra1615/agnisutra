from contextlib import asynccontextmanager
import pickle
import joblib
import json
import tensorflow as tf
from tensorflow.keras.layers import InputLayer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from .scheduler import check_conditions_job

from .routers import auth, iot, krishi_saathi, disease
from .ml import ml_models
from .manager import manager

# Fix for Keras Version Mismatch (batch_shape vs batch_input_shape)
class PatchedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    try:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        
        # 1. Load Keras Model (Fertilizer Recommender)
        model_path = os.path.join(current_dir, "final_model.keras")
        if os.path.exists(model_path):
            # compile=False is often safer for inference-only loading to avoid optimizer version mismatches
            # custom_objects={'InputLayer': PatchedInputLayer} handles the version mismatch
            ml_models["fertilizer_model"] = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={'InputLayer': PatchedInputLayer}
            )
            print(f"✅ Keras Fertilizer Model loaded successfully from {model_path}")
        else:
            print(f"❌ Model file not found at {model_path}")
            
        # 2. Load Preprocessor (Required for Keras model)
        # Look in current dir first, then project root
        preprocessor_path = os.path.join(current_dir, "dl_preprocessor.joblib")
        if not os.path.exists(preprocessor_path):
             # Fallback to project root (../../dl_preprocessor.joblib)
             preprocessor_path = os.path.abspath(os.path.join(current_dir, "../../dl_preprocessor.joblib"))
        
        if os.path.exists(preprocessor_path):
            ml_models["preprocessor"] = joblib.load(preprocessor_path)
            print(f"✅ Preprocessor loaded successfully from {preprocessor_path}")
        else:
            print(f"❌ Preprocessor file not found at {preprocessor_path}")
            ml_models["preprocessor"] = None

        # 3. Load Plant Disease CNN Model
        # Try backend/app/models/plant_disease_model.h5 or CNN_PLANT_DISEASE/plant_disease_prediction_model.h5
        cnn_paths = [
            os.path.join(current_dir, "models", "plant_disease_prediction_model.h5"),
            os.path.join(current_dir, "models", "plant_disease_model.h5"),
            os.path.join(project_root, "CNN_PLANT_DISEASE", "plant_disease_prediction_model.h5"),
            os.path.join(current_dir, "plant_disease_model.h5")
        ]
        
        ml_models["disease_cnn"] = None
        for path in cnn_paths:
            if os.path.exists(path):
                try:
                    ml_models["disease_cnn"] = tf.keras.models.load_model(path)
                    print(f"✅ Disease CNN Model loaded from {path}")
                    break
                except Exception as e:
                    print(f"❌ Failed to load Disease CNN from {path}: {e}")
        
        if not ml_models["disease_cnn"]:
            print("⚠️ Disease CNN Model not found. Using GPT-4 Vision fallback.")

        # 4. Load Class Indices
        indices_paths = [
            os.path.join(current_dir, "models", "class_indices.json"),
            os.path.join(project_root, "CNN_PLANT_DISEASE", "class_indices.json")
        ]
        
        ml_models["class_indices"] = None
        for path in indices_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        ml_models["class_indices"] = json.load(f)
                    print(f"✅ Class Indices loaded from {path}")
                    break
                except Exception as e:
                    print(f"❌ Failed to load Class Indices from {path}: {e}")
        
        if not ml_models["class_indices"]:
            print("❌ Class Indices not found in any location.")

        # 5. Load FAISS Index for Disease RAG
        faiss_path = os.path.join(current_dir, "faiss_disease_index")
        if os.path.exists(os.path.join(faiss_path, "index.faiss")):
            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                ml_models["disease_vectorstore"] = FAISS.load_local(
                    faiss_path, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Disease FAISS Index loaded from {faiss_path}")
            except Exception as e:
                print(f"❌ Failed to load Disease FAISS Index: {e}")
                ml_models["disease_vectorstore"] = None
        else:
            print(f"⚠️ Disease FAISS Index not found at {faiss_path}")
            ml_models["disease_vectorstore"] = None
            
    except Exception as e:
        print(f"Error loading ML model: {e}")
        ml_models["fertilizer_model"] = None
        ml_models["preprocessor"] = None
    
    # Start Scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(check_conditions_job, 'interval', minutes=0.5) # Run every 30 mins
    scheduler.start()
    print("✅ Scheduler started: Running every 30 minutes.")
    
    yield
    
    # Clean up
    scheduler.shutdown()
    ml_models.clear()

app = FastAPI(title="AgniSutra API", version="1.0.0", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configure CORS - allow web frontend and common mobile emulator hosts
origins = ["*"]

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
app.include_router(iot.router, prefix="/iot", tags=["iot"])
app.include_router(krishi_saathi.router, prefix="/krishi-saathi", tags=["krishi-saathi"])
app.include_router(disease.router, prefix="/disease", tags=["disease"])

