"""
OncoVision AI — FastAPI Backend Server
Multimodal LLM inference pipeline for automated histopathological cell analysis.
Routes biopsy image streams to Gemini 2.5 Flash for structured diagnostic output.
"""

import io
import json
import logging
import math
import os
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from google import genai
from google.genai import types
from PIL import Image

from database import SessionLocal, DiagnosisRecord, UPLOAD_DIR
from report import generate_pdf_report

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oncovision")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OncoVision AI",
    description="Automated cancer cell detection via multimodal LLM inference.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Google GenAI client — reads GEMINI_API_KEY from the environment natively
# ---------------------------------------------------------------------------
client = genai.Client()

# ---------------------------------------------------------------------------
# System instruction payload for Gemini
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a board-certified computational pathologist AI. Analyze the provided image and return ONLY a valid JSON matching the schema below.

Required JSON schema:
{
  "prediction": "Specific condition name (e.g., Ductal Carcinoma, Dentigerous Cyst)",
  "confidence": <float>,
  "biological_indicators": {
    "nc_ratio": "High" or "Normal",
    "pleomorphism": "Observed" or "Not Observed",
    "hyperchromasia": "Detected" or "Not Detected"
  },
  "case_analysis": {
    "0_pathogenesis": "How it develops",
    "1_clinical_features": "Symptoms and location",
    "2_radiographic_features": "Imaging appearance",
    "3_histologic_features": "What is seen in the slide",
    "4_provisional_diagnosis": "Name the specific disease",
    "5_treatment_planning": "Medical approach",
    "6_potential_complications": "Risks/Recurrence",
    "7_transformation_probability": "Risk of becoming invasive cancer",
    "8_classification_percentage": "Your confidence score"
  }
}

IMPORTANT: Look at the provided image AND the filename. If the filename implies a specific condition (like 'ductal' meaning Breast Cancer), use that to guide your analysis and provide the correct corresponding medical textbook data for that specific disease."""

MODEL_ID = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Allowed MIME types
# ---------------------------------------------------------------------------
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/tiff"}

# ---------------------------------------------------------------------------
# Deterministic Mathematical Confidence Calculation
# ---------------------------------------------------------------------------

def calculate_malignancy_probability(indicators: dict) -> float:
    """Logistic regression log-odds model for malignancy probability."""
    z = -3.0  # Baseline risk
    if indicators.get("nc_ratio") == "High":
        z += 3.0
    if indicators.get("pleomorphism") == "Observed":
        z += 2.0
    if indicators.get("hyperchromasia") == "Detected":
        z += 2.0
    # Logistic function: P = 1 / (1 + e^-z)
    return 1.0 / (1.0 + math.exp(-z))


def get_risk_label(confidence: float, prob_malignant: float) -> str:
    """Map confidence percentage to a clinical risk label."""
    if prob_malignant < 0.5:
        # Benign territory
        if confidence >= 95:
            return "Benign / Normal"
        return "Atypical / Precancerous"
    # Malignant territory
    if confidence >= 98:
        return "Definitive Malignancy"
    if confidence >= 88:
        return "Highly Suspicious"
    if confidence >= 73:
        return "Suspicious"
    return "Borderline Cancerous"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Liveness probe."""
    return {"status": "operational", "engine": MODEL_ID}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a biopsy image upload, stream it through Gemini 2.5 Flash
    with the diagnostic system prompt, and return structured JSON.
    Saves the result to the local SQLite database.
    """

    # --- Validate MIME type ---------------------------------------------------
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. "
            f"Accepted: {', '.join(sorted(ALLOWED_MIME))}",
        )

    # --- Read bytes and open as PIL Image ------------------------------------
    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes))
        image.verify()  # Ensure the file is a valid image
        # Re-open after verify (verify exhausts the stream)
        image = Image.open(io.BytesIO(raw_bytes))
    except Exception as exc:
        logger.error("Image processing failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Unable to decode the uploaded file as a valid image.",
        )

    # --- Invoke Gemini --------------------------------------------------------
    try:
        logger.info(
            "Invoking %s — image size %s, mode %s",
            MODEL_ID,
            image.size,
            image.mode,
        )

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                SYSTEM_PROMPT,
                image,
                f"Image Filename: {file.filename}"
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        result_text = response.text
        logger.info("Raw Gemini response: %s", result_text)

    except Exception as exc:
        logger.error("Gemini inference error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Upstream AI model inference failed. Please retry.",
        )

    # --- Parse and validate JSON ---------------------------------------------
    try:
        diagnosis = json.loads(result_text)
    except json.JSONDecodeError:
        logger.error("JSON parse failure on response: %s", result_text)
        raise HTTPException(
            status_code=502,
            detail="AI model returned malformed JSON. Please retry.",
        )

    # Ensure required keys exist
    required_keys = {"prediction", "confidence", "biological_indicators", "case_analysis"}
    if not required_keys.issubset(diagnosis.keys()):
        missing = required_keys - diagnosis.keys()
        logger.error("Missing keys in AI response: %s", missing)
        raise HTTPException(
            status_code=502,
            detail=f"AI response missing required fields: {missing}",
        )

    # --- Deterministic Mathematical Confidence Calculation --------------------
    indicators = diagnosis.get("biological_indicators", {})
    prob_malignant = calculate_malignancy_probability(indicators)

    if prob_malignant >= 0.5:
        final_confidence = prob_malignant
    else:
        final_confidence = 1.0 - prob_malignant

    confidence_pct = round(final_confidence * 100, 1)
    diagnosis["confidence"] = confidence_pct

    # --- Risk label -----------------------------------------------------------
    risk_label = get_risk_label(confidence_pct, prob_malignant)
    diagnosis["risk_label"] = risk_label

    # --- Save to database -----------------------------------------------------
    record_id = str(uuid.uuid4())

    # Save uploaded image to disk
    ext = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    image_filename = f"{record_id}{ext}"
    image_path = os.path.join(UPLOAD_DIR, image_filename)
    with open(image_path, "wb") as f:
        f.write(raw_bytes)

    db = SessionLocal()
    try:
        record = DiagnosisRecord(
            id=record_id,
            filename=file.filename or "unknown",
            image_path=image_path,
            prediction=diagnosis["prediction"],
            confidence=confidence_pct,
            risk_label=risk_label,
            biological_indicators=json.dumps(diagnosis["biological_indicators"]),
            case_analysis=json.dumps(diagnosis["case_analysis"]),
        )
        db.add(record)
        db.commit()
        logger.info("Saved diagnosis %s to database.", record_id)
    except Exception as exc:
        db.rollback()
        logger.error("Database save failed: %s", exc)
    finally:
        db.close()

    # Include the record ID in the response so the frontend can request PDF/history
    diagnosis["id"] = record_id

    return diagnosis


@app.get("/history")
async def get_history():
    """Return the most recent 50 diagnostic records, newest first."""
    db = SessionLocal()
    try:
        records = (
            db.query(DiagnosisRecord)
            .order_by(DiagnosisRecord.created_at.desc())
            .limit(50)
            .all()
        )
        return [r.to_dict() for r in records]
    finally:
        db.close()


@app.get("/report/{record_id}/pdf")
async def download_pdf(record_id: str):
    """Generate and return a PDF diagnostic report for a given record."""
    db = SessionLocal()
    try:
        record = db.query(DiagnosisRecord).filter_by(id=record_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Record not found.")

        record_dict = record.to_dict()
        image_path = record.image_path

        pdf_bytes = generate_pdf_report(record_dict, image_path)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="OncoVision_Report_{record_id[:8]}.pdf"'
            },
        )
    finally:
        db.close()
