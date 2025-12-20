import os
import base64
import json
import io
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ..ml import ml_models

load_dotenv()

router = APIRouter()

MODEL = "gpt-4o-mini"  # Vision + Text model


def encode_image(img_file) -> str:
    img_file.seek(0)
    return base64.b64encode(img_file.read()).decode("utf-8")


def _get_llm(max_tokens: int = 4000) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing in .env")

    return ChatOpenAI(
        model=MODEL,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=0.25,
    )

# --- CNN & RAG Helper Functions ---

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for CNN model (224x224, normalized).
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

def get_rag_advice(disease_name: str, language: str = "en") -> str:
    """
    Use RAG to get preventive measures for the predicted disease.
    """
    vectorstore = ml_models.get("disease_vectorstore")
    if not vectorstore:
        return "Detailed advice not available (RAG system offline)."

    try:
        llm = _get_llm(max_tokens=3000)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        prompt_template = """
        You are an expert plant pathologist.
        Use the following context to answer the question.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Provide a detailed response in {language} language.
        Structure the answer as:
        1. Disease Description & Symptoms
        2. Preventive Measures (Cultural)
        3. Chemical Control (if applicable)
        4. Biological Control
        
        If the context doesn't have specific info, use your general knowledge but mention it.
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"],
            partial_variables={"language": language}
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        query = f"What are the preventive measures and treatments for {disease_name}?"
        result = chain.invoke(query)
        return result.get("result", "No advice generated.")
        
    except Exception as e:
        print(f"RAG Error: {e}")
        return f"Error generating advice: {str(e)}"


def ask_about_image(img_file, crop_name: str, query: str):
    image_b64 = encode_image(img_file)
    llm = _get_llm()

    enhanced_prompt = (
        f"Crop: {crop_name}\n"
        f"Farmer Query: {query}\n\n"
        "You are a world-class Plant Pathologist & Agronomist.\n"
        "Analyze the uploaded image and generate a **very detailed expert advisory**.\n"
        "Do NOT use bullet template from other queries — tailor the response to THIS case.\n"
        "Provide:\n\n"
        "1️⃣ Disease Identification & Deep Reasoning\n"
        "- Explain why the disease is diagnosed: lesion morphology, color, margins, pattern, necrosis, sporulation\n"
        "- Compare with 1–2 look-alike diseases (brief)\n"
        "- Expected yield loss range based on current severity\n\n"

        "2️⃣ Causes & Epidemiology\n"
        "- Pathogen biology, spread (wind/rain/seed/soil/insects)\n"
        "- Weather triggers (humidity %, rain pattern, leaf wetness hours, temp °C)\n"
        "- Field microclimate influence\n\n"

        "3️⃣ Resistant / Tolerant Varieties (MOST IMPORTANT — Provide 4–8 real examples)\n"
        "- For each: name, maturity group (early/medium/late), suitability zones, partial/full resistance level\n"
        "- Expected performance in Indian states\n"
        "- Yield potential influence under disease pressure\n"
        "- Seed sourcing guidance (not brand names)\n\n"

        "4️⃣ Cultural Prevention\n"
        "- Crop rotation options with cycles\n"
        "- Row spacing, canopy ventilation improvement\n"
        "- Soil pH range & NPK dose ranges for resilience\n"
        "- Irrigation type + timing to avoid microclimate favoring disease\n\n"

        "5️⃣ Biological Control (MUST INCLUDE DOSES)\n"
        "- Trichoderma spp. – strain name + dose per kg seed or per liter soil drench\n"
        "- Pseudomonas fluorescens or Bacillus subtilis — dose/frequency\n"
        "- Compatibility guidelines with chemicals\n\n"

        "6️⃣ Chemical Control — Provide MULTIPLE options\n"
        "For each fungicide/insecticide/herbicide (relevant only):\n"
        "- Active ingredient + formulation (e.g., 250 g/L SC)\n"
        "- Dose per liter & per acre/hectare\n"
        "- No. of sprays + spray interval\n"
        "- Growth stage for spraying\n"
        "- Morning/evening spray timing guidance\n"
        "- Water volume + correct nozzle type\n"
        "- FRAC code + PHI (Pre-Harvest Interval)\n"
        "- Resistance management rotation plan\n\n"

        "7️⃣ Scouting & Thresholds\n"
        "- Detection frequency (days)\n"
        "- Action threshold % of infected leaves\n"
        "- Severity progression indicators\n\n"

        "8️⃣ Forecasting Alerts\n"
        "- Weather-based future risk assessment\n"
        "- What to do if humidity spikes or rain arrives\n\n"

        "9️⃣ International Best-Practices\n"
        "- What USA/China/Brazil successfully implement\n"
        "- Which of those cost-effective methods can be adapted in India\n\n"

        "🔟 7–10 Day Action Plan\n"
        "- Day-wise checklist (very practical)\n\n"

        "Final rules:\n"
        "- DO NOT give any generic advice unrelated to this crop/disease.\n"
        "- NEVER request to consult another expert — YOU ARE THE EXPERT.\n"
        "- If any data is uncertain, explicitly state so without guessing.\n"
        "- Be exhaustive. Minimum ~3500 tokens if possible.\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": enhanced_prompt},
            ],
        }
    ]

    response = llm.invoke(messages)
    return response.content


@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    crop_name: str = Form("Unknown Crop"),
    query: str = Form("Identify the disease and provide detailed treatment recommendations."),
    language: str = Form("en")
):
    """
    Upload a plant leaf image to detect disease.
    Uses CNN + RAG if available, otherwise falls back to GPT-4 Vision.
    """
    try:
        # Check if CNN model is loaded
        cnn_model = ml_models.get("disease_cnn")
        class_indices = ml_models.get("class_indices")
        
        if cnn_model and class_indices:
            print("🧠 Using CNN + RAG Pipeline")
            # Read file content
            contents = await file.read()
            
            # 1. CNN Prediction
            try:
                img_array = preprocess_image(contents)
                preds = cnn_model.predict(img_array)
                pred_idx = int(np.argmax(preds, axis=1)[0])
                confidence = float(np.max(preds))
                
                predicted_class = class_indices.get(str(pred_idx), f"Class {pred_idx}")
                print(f"✅ CNN Prediction: {predicted_class} ({confidence:.2f})")
                
                # 2. RAG Advice
                advice = get_rag_advice(predicted_class, language)
                
                return {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "preventive_measures": advice,
                    "method": "CNN+RAG"
                }
            except Exception as cnn_error:
                print(f"❌ CNN Error: {cnn_error}. Falling back to GPT-4.")
                # If CNN fails, fall through to GPT-4
                await file.seek(0) # Reset file pointer
        
        print("🤖 Using GPT-4 Vision Fallback")
        # Fallback to GPT-4 Vision
        # Ensure file pointer is at start
        await file.seek(0)
        result = ask_about_image(file.file, crop_name, query)
        
        return {
            "predicted_class": "AI Analysis (GPT-4 Vision)",
            "confidence": 1.0,
            "preventive_measures": result,
            "method": "GPT-4-Vision"
        }
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
