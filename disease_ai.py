import os
import base64
import json
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

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
