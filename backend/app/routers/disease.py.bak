import os
import json
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
from typing import Optional

# RAG / LangChain imports
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
try:
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except ImportError:
    from langchain_classic.chains import RetrievalQA
    from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

router = APIRouter()

# ==========================================================
# 1. CONFIG & PATHS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # backend/app
MODELS_DIR = os.path.join(BASE_DIR, "models")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_disease_index")

MODEL_PATH = os.path.join(MODELS_DIR, "plant_disease_prediction_model.h5")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ==========================================================
# 2. GLOBAL RESOURCES (Lazy Loading)
# ==========================================================
_cnn_model = None
_class_indices = None
_qa_chain = None

def get_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ CNN Model not found at {MODEL_PATH}")
            return None
        try:
            print(f"🔄 Loading CNN Model from {MODEL_PATH}...")
            _cnn_model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ CNN Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading CNN Model: {e}")
            return None
    return _cnn_model

def get_class_indices():
    global _class_indices
    if _class_indices is None:
        if not os.path.exists(CLASS_INDICES_PATH):
            print(f"❌ Class Indices not found at {CLASS_INDICES_PATH}")
            return {}
        try:
            with open(CLASS_INDICES_PATH, "r") as f:
                _class_indices = json.load(f)
            # Ensure keys are integers (if they are stored as strings in JSON)
            # The original app.py didn't convert keys, but usually json keys are strings.
            # Let's check the structure. If it's {"0": "Apple___Apple_scab", ...}
            # We might need to map index -> name.
            # The original app.py used: prediction = np.argmax(...); predicted_class_name = class_indices[str(prediction)]
            print("✅ Class Indices loaded.")
        except Exception as e:
            print(f"❌ Error loading Class Indices: {e}")
            return {}
    return _class_indices

def get_qa_chain():
    global _qa_chain
    if _qa_chain is None:
        if not os.path.exists(FAISS_INDEX_DIR):
            print(f"❌ FAISS Index not found at {FAISS_INDEX_DIR}")
            return None
        
        try:
            print(f"🔄 Loading FAISS Index from {FAISS_INDEX_DIR}...")
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            llm = ChatOpenAI(
                model_name=OPENAI_MODEL,
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            prompt_template = """
            You are an expert plant pathologist.
            Use the following pieces of context to recommend preventive measures and treatments for the plant disease identified.
            
            Disease Context: {context}
            
            Question: {question}
            
            Provide a concise, actionable list of preventive measures and treatments.
            Format your response using Markdown:
            - Use bullet points for the list.
            - Use **bold** for key terms.
            - Keep it easy to read.
            
            If you don't know the answer based on the context, say "I don't have specific information for this disease in my database, but general good agricultural practices include..."
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            
            _qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT}
            )
            print("✅ RAG QA Chain initialized.")
        except Exception as e:
            print(f"❌ Error initializing RAG Chain: {e}")
            return None
    return _qa_chain

# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

# ==========================================================
# 4. ENDPOINTS
# ==========================================================

@router.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Upload a plant leaf image to detect disease and get preventive measures.
    """
    # 1. Load Resources
    model = get_cnn_model()
    indices = get_class_indices()
    qa_chain = get_qa_chain()
    
    if not model or not indices:
        raise HTTPException(status_code=503, detail="Model or Class Indices not available.")
    
    # 2. Process Image
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        processed_image = preprocess_image(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    
    # 3. Predict
    try:
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        # Map index to class name
        # JSON keys are strings, so convert index to string
        predicted_class_name = indices.get(str(predicted_index), "Unknown Disease")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    # 4. Get Preventive Measures (RAG)
    preventive_measures = "Preventive measures not available (RAG system offline)."
    if qa_chain:
        try:
            query = f"What are the preventive measures for {predicted_class_name}?"
            result = qa_chain.invoke(query)
            # RetrievalQA returns a dict with 'result' or 'answer' depending on version/config
            # Usually 'result' for RetrievalQA
            preventive_measures = result.get("result", result.get("answer", "No answer generated."))
        except Exception as e:
            print(f"⚠️ RAG Error: {e}")
            preventive_measures = f"Could not fetch preventive measures: {e}"
            
    return {
        "predicted_class": predicted_class_name,
        "confidence": confidence,
        "preventive_measures": preventive_measures
    }
