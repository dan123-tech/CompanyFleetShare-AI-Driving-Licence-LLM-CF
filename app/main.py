"""
Gemini AI - Driving Licence Validator Service
Receives a driving licence image, analyses it with Google Gemini,
and returns whether the driver has 2+ years of experience.
"""

import os
import io
import base64
import logging
from datetime import datetime, date
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# google-generativeai / PIL are imported lazily so /health works on Vercel and cold
# starts do not fail if those stacks misbehave during import.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("licence-validator")

# ---------------------------------------------------------------------------
# Gemini configuration (lazy init — supports Vercel build & cold start without import-time env)
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_gemini_model = None


def _get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="GEMINI_API_KEY is not configured on the server",
            )
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    return _gemini_model

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
fastapi_app = FastAPI(
    title="Driving Licence Validator",
    description="Validates driving licences using Gemini AI – checks 2+ years of experience",
    version="1.0.0",
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class ValidationResult(BaseModel):
    approved: bool
    years_of_experience: float | None = None
    issue_date: str | None = None
    expiry_date: str | None = None
    licence_number: str | None = None
    holder_name: str | None = None
    licence_categories: str | None = None
    reason: str
    raw_ai_response: str


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    gemini_configured: bool = False


# ---------------------------------------------------------------------------
# Prompt sent to Gemini
# ---------------------------------------------------------------------------
VALIDATION_PROMPT = """You are an expert driving licence document analyst.

Analyse the driving licence image provided and extract the following information:

1. **Holder name** – full name on the licence
2. **Licence number** – the unique licence identifier
3. **Issue date** – date the licence was first issued (dd/mm/yyyy)
4. **Expiry date** – date the licence expires (dd/mm/yyyy)
5. **Licence categories** – vehicle categories the holder is allowed to drive
6. **Years of driving experience** – calculate from the issue date to today ({today}).
   If the issue date is not clearly visible, estimate from any available dates.

Then decide:
- **APPROVED** if the holder has **2 or more years** of driving experience.
- **REJECTED** if the holder has **less than 2 years** of driving experience.
- If the image is **not a driving licence**, respond with REJECTED and explain.

Respond ONLY with the following JSON (no markdown fences, no extra text):
{{
  "approved": true or false,
  "years_of_experience": <number or null>,
  "issue_date": "<dd/mm/yyyy or null>",
  "expiry_date": "<dd/mm/yyyy or null>",
  "licence_number": "<string or null>",
  "holder_name": "<string or null>",
  "licence_categories": "<string or null>",
  "reason": "<brief explanation>"
}}
"""


# ---------------------------------------------------------------------------
# Helper – call Gemini with the image
# ---------------------------------------------------------------------------
async def analyse_licence(image_bytes: bytes, content_type: str) -> dict:
    """Send the licence image to Gemini and parse the structured response."""
    from google.api_core import exceptions as google_exceptions
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    today_str = date.today().strftime("%d/%m/%Y")
    prompt = VALIDATION_PROMPT.replace("{today}", today_str)

    logger.info("Sending image to Gemini for analysis …")
    try:
        response = _get_gemini_model().generate_content([prompt, image])
    except google_exceptions.ResourceExhausted as e:
        logger.warning("Gemini quota exceeded: %s", e)
        raise HTTPException(
            status_code=429,
            detail="Gemini API quota exceeded. Try again later or check your plan at https://ai.google.dev/gemini-api/docs/rate-limits",
        ) from e

    raw_text = response.text.strip()
    logger.info("Gemini raw response: %s", raw_text)

    # Try to parse the JSON response from Gemini
    import json

    # Strip markdown code fences if Gemini adds them anyway
    cleaned = raw_text
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # If parsing fails, return a conservative rejection
        return {
            "approved": False,
            "years_of_experience": None,
            "issue_date": None,
            "expiry_date": None,
            "licence_number": None,
            "holder_name": None,
            "licence_categories": None,
            "reason": "Could not parse AI response. Manual review required.",
            "raw_ai_response": raw_text,
        }

    data["raw_ai_response"] = raw_text
    return data


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@fastapi_app.get("/health", response_model=HealthResponse)
async def health():
    """Health-check endpoint – useful for Docker / orchestrators."""
    return HealthResponse(
        status="healthy",
        service="driving-licence-validator",
        timestamp=datetime.utcnow().isoformat(),
        gemini_configured=bool(os.getenv("GEMINI_API_KEY")),
    )


@fastapi_app.post("/validate", response_model=ValidationResult)
async def validate_licence(file: UploadFile = File(...)):
    """
    Upload a driving licence image.
    Returns approval status based on 2+ years of experience.
    """
    # Validate content type
    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {', '.join(sorted(allowed))}",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(image_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=400, detail="File too large (max 20 MB)")

    logger.info(
        "Received file: %s (%s, %d bytes)",
        file.filename,
        file.content_type,
        len(image_bytes),
    )

    result = await analyse_licence(image_bytes, file.content_type)
    return ValidationResult(**result)


@fastapi_app.post("/validate-base64", response_model=ValidationResult)
async def validate_licence_base64(payload: dict):
    """
    Alternative endpoint – accepts a base64-encoded image in JSON body.
    Useful for container-to-container communication.

    Expected body:
    {
        "image_base64": "<base64 string>",
        "content_type": "image/jpeg"   (optional, defaults to image/jpeg)
    }
    """
    image_b64 = payload.get("image_base64")
    if not image_b64:
        raise HTTPException(status_code=400, detail="'image_base64' field is required")

    content_type = payload.get("content_type", "image/jpeg")

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data")

    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 20 MB)")

    result = await analyse_licence(image_bytes, content_type)
    return ValidationResult(**result)
