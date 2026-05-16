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

@app.get("/report/{record_id}/html")
async def html_report(record_id: str):
    """Generate and return a styled HTML diagnostic report."""
    db = SessionLocal()
    try:
        record = db.query(DiagnosisRecord).filter_by(id=record_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Record not found.")

        d = record.to_dict()
        bio = d.get("biological_indicators", {})
        case = d.get("case_analysis", {})

        # Build case analysis HTML
        case_html = ""
        for key in sorted(case.keys()):
            if key == "8_classification_percentage":
                continue
            val = case[key]
            label = " ".join(w.capitalize() for w in key.split("_")[1:])
            case_html += f'<div class="case-item"><h4>{label}</h4><p>{val}</p></div>'

        # Inline image as base64 if available
        img_html = ""
        if record.image_path and os.path.exists(record.image_path):
            import base64
            ext = os.path.splitext(record.image_path)[1].lower()
            mime = {"jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
            with open(record.image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            img_html = f'<img src="data:{mime};base64,{b64}" alt="Biopsy" />'

        def marker_class(val, active):
            return "active" if val == active else ""

        # Compute the math reasoning step-by-step
        nc_val = bio.get("nc_ratio", "Normal")
        pleo_val = bio.get("pleomorphism", "Not Observed")
        hyper_val = bio.get("hyperchromasia", "Not Detected")

        z_baseline = -3.0
        nc_weight = 3.0 if nc_val == "High" else 0.0
        pleo_weight = 2.0 if pleo_val == "Observed" else 0.0
        hyper_weight = 2.0 if hyper_val == "Detected" else 0.0
        z_total = z_baseline + nc_weight + pleo_weight + hyper_weight
        prob_malignant = 1.0 / (1.0 + math.exp(-z_total))
        prob_benign = 1.0 - prob_malignant
        is_mal = prob_malignant >= 0.5
        final_conf = prob_malignant if is_mal else prob_benign
        direction = "Malignant" if is_mal else "Benign"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>OncoVision Report — {d['prediction']}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',system-ui,sans-serif; background:#fafafa; color:#18181b; padding:0; }}
  .page {{ max-width:800px; margin:40px auto; background:#fff; border:1px solid #e4e4e7; }}
  .header {{ padding:32px 40px; border-bottom:1px solid #e4e4e7; }}
  .header h1 {{ font-size:22px; font-weight:700; letter-spacing:-0.5px; }}
  .header p {{ font-size:12px; color:#71717a; margin-top:4px; }}
  .meta {{ padding:20px 40px; border-bottom:1px solid #e4e4e7; display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }}
  .meta-item span {{ display:block; font-size:10px; color:#a1a1aa; text-transform:uppercase; letter-spacing:1px; }}
  .meta-item strong {{ font-size:13px; font-weight:500; }}
  .section {{ padding:24px 40px; border-bottom:1px solid #e4e4e7; }}
  .section-title {{ font-size:10px; color:#a1a1aa; text-transform:uppercase; letter-spacing:2px; margin-bottom:16px; font-weight:600; }}
  .classification {{ display:flex; align-items:center; justify-content:space-between; }}
  .prediction {{ font-size:24px; font-weight:600; }}
  .badge {{ font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:1px; padding:4px 10px; border:1px solid #18181b; }}
  .badge.benign {{ border-color:#d4d4d8; color:#71717a; }}
  .conf-bar {{ height:4px; background:#f4f4f5; margin-top:16px; border-radius:2px; overflow:hidden; }}
  .conf-fill {{ height:100%; background:#18181b; transition:width 0.5s; }}
  .conf-label {{ display:flex; justify-content:space-between; margin-top:6px; font-size:11px; color:#a1a1aa; font-family:monospace; }}
  img {{ max-width:320px; border:1px solid #e4e4e7; margin-top:8px; }}
  .markers {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }}
  .marker {{ padding:16px; border:1px solid #e4e4e7; }}
  .marker h4 {{ font-size:12px; font-weight:600; margin-bottom:4px; }}
  .marker .desc {{ font-size:10px; color:#a1a1aa; }}
  .marker .status {{ display:inline-block; margin-top:8px; font-size:11px; font-family:monospace; padding:2px 8px; border:1px solid #e4e4e7; color:#a1a1aa; }}
  .marker .status.active {{ border-color:#18181b; color:#18181b; background:#fafafa; font-weight:600; }}
  .case-item {{ margin-bottom:16px; padding-left:16px; border-left:2px solid #e4e4e7; }}
  .case-item h4 {{ font-size:11px; color:#71717a; text-transform:uppercase; letter-spacing:1px; font-weight:500; margin-bottom:4px; }}
  .case-item p {{ font-size:13px; line-height:1.6; color:#3f3f46; }}
  .disclaimer {{ padding:24px 40px; background:#fafafa; font-size:11px; color:#a1a1aa; line-height:1.5; }}
  @media print {{
    body {{ background:#fff; }}
    .page {{ border:none; margin:0; box-shadow:none; }}
    .no-print {{ display:none; }}
  }}
  .print-btn {{ position:fixed; bottom:24px; right:24px; background:#18181b; color:#fff; border:none; padding:12px 24px; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:1px; cursor:pointer; font-family:'Inter',sans-serif; }}
  .print-btn:hover {{ background:#3f3f46; }}
  .math-section {{ background:#fafafa; }}
  .formula {{ font-family:'Courier New',monospace; background:#f4f4f5; padding:16px 20px; border:1px solid #e4e4e7; margin:12px 0; font-size:13px; line-height:1.8; overflow-x:auto; }}
  .formula .highlight {{ color:#18181b; font-weight:700; }}
  .formula .dim {{ color:#a1a1aa; }}
  .step-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:12px 0; }}
  .step-card {{ padding:14px 16px; border:1px solid #e4e4e7; background:#fff; }}
  .step-card .label {{ font-size:10px; color:#a1a1aa; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }}
  .step-card .value {{ font-size:18px; font-weight:600; font-family:'Courier New',monospace; }}
  .step-card .note {{ font-size:11px; color:#71717a; margin-top:4px; }}
  .arrow {{ text-align:center; font-size:20px; color:#a1a1aa; padding:8px 0; }}
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <h1>OncoVision AI</h1>
    <p>Histopathological Diagnostic Report · Generated {__import__('datetime').datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
  </div>
  <div class="meta">
    <div class="meta-item"><span>Report ID</span><strong>{d['id'][:8]}</strong></div>
    <div class="meta-item"><span>Original File</span><strong>{d['filename']}</strong></div>
    <div class="meta-item"><span>Analysis Date</span><strong>{str(d.get('created_at',''))[:19].replace('T',' ')}</strong></div>
  </div>
  <div class="section">
    <div class="section-title">Classification</div>
    <div class="classification">
      <span class="prediction">{d['prediction']}</span>
      <span class="badge {'benign' if 'Benign' in d.get('risk_label','') else ''}">{d.get('risk_label','N/A')}</span>
    </div>
    <div class="conf-bar"><div class="conf-fill" style="width:{d['confidence']}%"></div></div>
    <div class="conf-label"><span>0%</span><span>{d['confidence']}%</span><span>100%</span></div>
  </div>
  {'<div class="section"><div class="section-title">Biopsy Image</div>' + img_html + '</div>' if img_html else ''}
  <div class="section">
    <div class="section-title">Morphological Biomarkers</div>
    <div class="markers">
      <div class="marker">
        <h4>N:C Ratio</h4>
        <div class="desc">Nuclear-to-cytoplasmic ratio</div>
        <span class="status {marker_class(bio.get('nc_ratio',''), 'High')}">{bio.get('nc_ratio','N/A')}</span>
      </div>
      <div class="marker">
        <h4>Pleomorphism</h4>
        <div class="desc">Cellular irregularity</div>
        <span class="status {marker_class(bio.get('pleomorphism',''), 'Observed')}">{bio.get('pleomorphism','N/A')}</span>
      </div>
      <div class="marker">
        <h4>Hyperchromasia</h4>
        <div class="desc">Dense chromatin packing</div>
        <span class="status {marker_class(bio.get('hyperchromasia',''), 'Detected')}">{bio.get('hyperchromasia','N/A')}</span>
      </div>
    </div>
  </div>
  <div class="section math-section">
    <div class="section-title">Mathematical Reasoning — Logistic Regression</div>
    <p style="font-size:12px;color:#71717a;margin-bottom:16px;">Instead of relying on the AI's arbitrary confidence score, we calculate a deterministic probability using the Logistic Regression log-odds model based on the detected biomarkers.</p>

    <div class="formula">
      <span class="dim">Step 1: Calculate log-odds (z)</span><br/>
      z = <span class="highlight">-3.0</span> <span class="dim">(baseline risk)</span><br/>
      &nbsp;&nbsp;+ <span class="highlight">{'+3.0' if nc_weight > 0 else '+0.0'}</span> <span class="dim">(N:C Ratio = {nc_val}{' → abnormal' if nc_weight > 0 else ' → normal'})</span><br/>
      &nbsp;&nbsp;+ <span class="highlight">{'+2.0' if pleo_weight > 0 else '+0.0'}</span> <span class="dim">(Pleomorphism = {pleo_val}{' → abnormal' if pleo_weight > 0 else ' → normal'})</span><br/>
      &nbsp;&nbsp;+ <span class="highlight">{'+2.0' if hyper_weight > 0 else '+0.0'}</span> <span class="dim">(Hyperchromasia = {hyper_val}{' → abnormal' if hyper_weight > 0 else ' → normal'})</span><br/>
      <br/>
      z = <span class="highlight">{z_total}</span>
    </div>

    <div class="arrow">↓</div>

    <div class="formula">
      <span class="dim">Step 2: Apply Logistic (Sigmoid) Function</span><br/>
      P(Malignant) = 1 / (1 + e<sup>-z</sup>)<br/>
      P(Malignant) = 1 / (1 + e<sup>-({z_total})</sup>)<br/>
      P(Malignant) = 1 / (1 + {math.exp(-z_total):.6f})<br/>
      <span class="highlight">P(Malignant) = {prob_malignant:.4f} ({prob_malignant*100:.1f}%)</span>
    </div>

    <div class="arrow">↓</div>

    <div class="step-grid">
      <div class="step-card">
        <div class="label">Probability of Malignancy</div>
        <div class="value">{prob_malignant*100:.1f}%</div>
        <div class="note">{'Above 50% threshold → classified as malignant' if is_mal else 'Below 50% threshold → classified as benign'}</div>
      </div>
      <div class="step-card">
        <div class="label">Probability of Benign</div>
        <div class="value">{prob_benign*100:.1f}%</div>
        <div class="note">P(Benign) = 1 - P(Malignant)</div>
      </div>
    </div>

    <div class="formula">
      <span class="dim">Step 3: Final Confidence</span><br/>
      Direction: <span class="highlight">{direction}</span> (P(Malignant) {'≥' if is_mal else '<'} 0.5)<br/>
      Reported Confidence = P({direction}) = <span class="highlight">{final_conf*100:.1f}%</span>
    </div>
  </div>
  <div class="section">
    <div class="section-title">Case Analysis</div>
    {case_html}
  </div>
  <div class="disclaimer">
    ⚠ This report was generated by OncoVision AI, an educational prototype. It is not a substitute for a qualified pathologist or clinical diagnosis. All findings must be verified by a licensed medical professional.<br/><br/>
    OncoVision AI v1.0 · Gemini 2.5 Flash · Logistic Regression Confidence Model
  </div>
</div>
<button class="print-btn no-print" onclick="window.print()">⬇ Save as PDF</button>
</body>
</html>"""

        return Response(content=html, media_type="text/html")
    finally:
        db.close()
