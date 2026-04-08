"""
Cloudflare AI - Driving Licence Validator Service
Receives a driving licence image, analyses it with Cloudflare Workers AI,
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

# Third-party libs are imported lazily so /health works on Vercel and cold starts
# do not fail if optional stacks misbehave during import.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("licence-validator")

# ---------------------------------------------------------------------------
# Cloudflare Workers AI configuration
# ---------------------------------------------------------------------------
# Default model is vision-capable (image + prompt). You can override via env.
CF_AI_MODEL = os.getenv("CF_AI_MODEL", "@cf/llava-hf/llava-1.5-7b-hf")


def _cloudflare_configured() -> bool:
    return bool(os.getenv("CF_ACCOUNT_ID")) and bool(os.getenv("CF_API_TOKEN"))


def _cloudflare_ai_url() -> str:
    account_id = os.getenv("CF_ACCOUNT_ID")
    if not account_id:
        raise HTTPException(status_code=503, detail="CF_ACCOUNT_ID is not configured on the server")
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{CF_AI_MODEL}"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
fastapi_app = FastAPI(
    title="Driving Licence Validator",
    description="Validates driving licences using Cloudflare Workers AI – checks 2+ years of experience",
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
    family_name: str | None = None
    given_name: str | None = None
    birth_date: str | None = None
    birth_place: str | None = None
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
    cloudflare_configured: bool = False


# ---------------------------------------------------------------------------
# Prompt sent to the model
# ---------------------------------------------------------------------------
VALIDATION_PROMPT = """You are an expert driving licence document analyst.

Analyse the driving licence image provided and extract the standard EU-style fields:

- **Field 1**: family name
- **Field 2**: given name
- **Field 3**: date and place of birth (split into birth_date and birth_place)
- **Field 4a**: issue date (licence emitted date)
- **Field 4b**: expiry date
- **Field 9**: categories allowed to drive
- **Licence number**: unique licence identifier
- **Years of driving experience** – calculate from field 4a to today ({today}).
   If the issue date is not clearly visible, estimate from any available dates.

Then decide:
- **APPROVED** if the holder has **2 or more years** of driving experience.
- **REJECTED** if the holder has **less than 2 years** of driving experience.
- If the image is **not a driving licence**, respond with REJECTED and explain.

Rules:
- Do not guess. If a field is not clearly readable, set it to null.
- Dates must be exactly dd/mm/yyyy or null.
- Keep strings short (max 64 chars); never output long repeated text.
- Output MUST be valid JSON (double quotes), with ONLY the keys below.
- You may receive an OCR_TEXT block. Prefer OCR_TEXT over guessing from the image.
- Only output a field if it is supported by OCR_TEXT or is clearly visible.

Respond ONLY with the following JSON (no markdown fences, no extra text):
{{
  "approved": true or false,
  "years_of_experience": <number or null>,
  "family_name": "<string or null>",
  "given_name": "<string or null>",
  "birth_date": "<dd/mm/yyyy or null>",
  "birth_place": "<string or null>",
  "issue_date": "<dd/mm/yyyy or null>",
  "expiry_date": "<dd/mm/yyyy or null>",
  "licence_number": "<string or null>",
  "licence_categories": "<string or null>",
  "reason": "<brief explanation>"
}}
Keep text fields concise. Do not output excessively long strings.
"""


def _parse_ddmmyyyy(value: str | None) -> date | None:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    # Accept common separators produced by OCR/LLMs, normalize to dd/mm/yyyy.
    for sep in (".", "-", " "):
        raw = raw.replace(sep, "/")
    raw = "/".join([p for p in raw.split("/") if p])
    try:
        return datetime.strptime(raw, "%d/%m/%Y").date()
    except Exception:
        return None


def _cap_str(value: str | None, max_len: int = 64) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        return None
    return value[:max_len]


def _normalize_categories(raw: str | None) -> str | None:
    """
    Keep only known driving licence categories and discard issuer text/noise.
    """
    if not raw:
        return None
    if not isinstance(raw, str):
        raw = str(raw)

    allowed = {"AM", "A1", "A2", "A", "B1", "B", "BE", "C1", "C", "CE", "D1", "D", "DE"}

    # Normalize separators and tokenize.
    up = raw.upper()
    for sep in (";", "|", "/", "\\", "-", "_"):
        up = up.replace(sep, " ")
    tokens = [t.strip(",. ") for t in up.split() if t.strip(",. ")]

    # Keep tokens that are exact categories
    picked: list[str] = []
    for tok in tokens:
        if tok in allowed and tok not in picked:
            picked.append(tok)

    if not picked:
        return None

    # Stable, human-friendly order
    order = ["AM", "A1", "A2", "A", "B1", "B", "BE", "C1", "C", "CE", "D1", "D", "DE"]
    picked_sorted = [x for x in order if x in picked]
    return ", ".join(picked_sorted) if picked_sorted else None


def _postprocess_ai_result(data: dict) -> dict:
    """
    Make output more reliable:
    - Enforce expected types/lengths
    - Recompute years_of_experience from issue_date when possible
    - Ensure approved matches the computed years when possible
    """
    issue = _parse_ddmmyyyy(data.get("issue_date"))
    expiry = _parse_ddmmyyyy(data.get("expiry_date"))

    # Normalize strings
    data["holder_name"] = _cap_str(data.get("holder_name"))
    data["family_name"] = _cap_str(data.get("family_name"))
    data["given_name"] = _cap_str(data.get("given_name"))
    data["birth_place"] = _cap_str(data.get("birth_place"))
    data["licence_number"] = _cap_str(data.get("licence_number"))
    data["licence_categories"] = _normalize_categories(_cap_str(data.get("licence_categories")))
    data["reason"] = _cap_str(data.get("reason"), max_len=200) or "No reason provided."

    birth = _parse_ddmmyyyy(data.get("birth_date"))
    data["birth_date"] = birth.strftime("%d/%m/%Y") if birth else None

    # Build/normalize holder_name for backward compatibility
    if data.get("given_name") or data.get("family_name"):
        full = " ".join([x for x in [data.get("given_name"), data.get("family_name")] if x])
        data["holder_name"] = _cap_str(full)

    # If categories looks like it duplicated another field, drop it.
    if data.get("licence_categories") and (
        data["licence_categories"] == data.get("licence_number")
        or data["licence_categories"] == data.get("holder_name")
    ):
        data["licence_categories"] = None

    # Normalize dates to dd/mm/yyyy or null
    data["issue_date"] = issue.strftime("%d/%m/%Y") if issue else None
    data["expiry_date"] = expiry.strftime("%d/%m/%Y") if expiry else None

    # Recompute years if we have issue date
    if issue:
        today = date.today()
        # If issue date is in the future, treat as invalid
        if issue > today:
            data["years_of_experience"] = 0.0
            data["approved"] = False
            data["reason"] = "Issue date is in the future; manual review required."
            return data

        years = (today - issue).days / 365.25
        years = round(years, 1)
        data["years_of_experience"] = years
        data["approved"] = bool(years >= 2.0)

    # Ensure years is numeric or null
    y = data.get("years_of_experience")
    if y is None:
        pass
    else:
        try:
            data["years_of_experience"] = float(y)
        except Exception:
            data["years_of_experience"] = None

    # Ensure approved is boolean
    data["approved"] = bool(data.get("approved"))
    return data


def _extract_first_json_object(text: str) -> str | None:
    """
    Extract the first complete JSON object from text using a simple brace scanner
    that is aware of JSON strings and escapes.
    """
    if not text or not isinstance(text, str):
        return None

    start = text.find("{")
    if start == -1:
        return None

    in_str = False
    escape = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_str = False
        else:
            if ch == "\"":
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Helper – call Gemini with the image
# ---------------------------------------------------------------------------
async def analyse_licence(image_bytes: bytes, content_type: str) -> dict:
    """Send the licence image to Cloudflare Workers AI and parse the structured response."""
    import httpx
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    today_str = date.today().strftime("%d/%m/%Y")
    prompt = VALIDATION_PROMPT.replace("{today}", today_str)

    if not _cloudflare_configured():
        raise HTTPException(status_code=503, detail="Cloudflare Workers AI is not configured on the server")

    # Ensure the image is safely representable as bytes for Workers AI input.
    # We'll send a compact list of uint8 ints (matches the common Workers AI schema for images).
    ocr_text = ""
    normalized_bytes = image_bytes
    try:
        image = image.convert("RGB")

        # OCR pass to reduce hallucinations on IDs/dates.
        try:
            import pytesseract

            # Resize a bit to improve OCR for small text.
            w, h = image.size
            scale = 2 if max(w, h) < 1400 else 1
            if scale != 1:
                image_for_ocr = image.resize((w * scale, h * scale))
            else:
                image_for_ocr = image

            ocr_text = pytesseract.image_to_string(image_for_ocr) or ""
            ocr_text = ocr_text.strip()
        except Exception:
            ocr_text = ""

        # Normalize image bytes (keeps payload consistent)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        normalized_bytes = buf.getvalue()
    except Exception:
        ocr_text = ""
        normalized_bytes = image_bytes

    headers = {
        "Authorization": f"Bearer {os.getenv('CF_API_TOKEN')}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": (prompt + ("\n\nOCR_TEXT:\n" + ocr_text if ocr_text else "")),
        "image": list(normalized_bytes),
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "frequency_penalty": 0.2,
    }

    logger.info("Sending image to Cloudflare Workers AI for analysis …")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(_cloudflare_ai_url(), headers=headers, json=payload)
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail="Cloudflare AI request timed out") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail="Cloudflare AI request failed") from e

    if resp.status_code == 429:
        raise HTTPException(status_code=429, detail="Cloudflare AI rate limit exceeded. Try again later.")
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Cloudflare AI error ({resp.status_code}): {resp.text}")

    try:
        cf = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail="Cloudflare AI returned non-JSON response") from e

    # Workers AI typically returns:
    # { "success": true, "result": { "response": "..." } }
    # Some models may return { "result": { "description": "..." } } etc.
    result = (cf or {}).get("result") if isinstance(cf, dict) else None
    raw_text = ""
    if isinstance(result, dict):
        raw_text = (
            result.get("response")
            or result.get("text")
            or result.get("output")
            or result.get("description")
            or ""
        )
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    raw_text = raw_text.strip()

    logger.info("Cloudflare AI raw response: %s", raw_text)

    # Try to parse the JSON response from Gemini
    import json

    # Strip markdown code fences if Gemini adds them anyway
    cleaned = raw_text
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\\_", "_")
    while cleaned.startswith("{{") and cleaned.endswith("}}"):
        cleaned = cleaned[1:-1].strip()

    extracted = _extract_first_json_object(cleaned)
    if extracted:
        cleaned = extracted

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # If parsing fails, return a conservative rejection
        return {
            "approved": False,
            "years_of_experience": None,
            "family_name": None,
            "given_name": None,
            "birth_date": None,
            "birth_place": None,
            "issue_date": None,
            "expiry_date": None,
            "licence_number": None,
            "holder_name": None,
            "licence_categories": None,
            "reason": "Could not parse AI response. Manual review required.",
            "raw_ai_response": raw_text,
        }

    data["raw_ai_response"] = raw_text
    return _postprocess_ai_result(data)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@fastapi_app.get("/")
async def root():
    """So the bare deployment URL (e.g. on Vercel) is not a 404."""
    return {
        "service": "driving-licence-validator",
        "health": "/health",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "validate": "POST /validate (multipart image)",
            "validate_base64": "POST /validate-base64 (JSON)",
        },
    }


@fastapi_app.get("/health", response_model=HealthResponse)
async def health():
    """Health-check endpoint – useful for Docker / orchestrators."""
    return HealthResponse(
        status="healthy",
        service="driving-licence-validator",
        timestamp=datetime.utcnow().isoformat(),
            cloudflare_configured=_cloudflare_configured(),
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
