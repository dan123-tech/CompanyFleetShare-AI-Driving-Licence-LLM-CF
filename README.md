# Driving Licence Validator – Cloudflare Workers AI

A Dockerised microservice that uses **Cloudflare Workers AI** to analyse driving licence images and determine whether the holder has **2+ years of driving experience**.

---

## Architecture

```
┌──────────────┐        image         ┌──────────────────┐
│              │  ─────────────────►  │                  │
│  Client App  │                      │  AI Validator    │
│  (future)    │  ◄─────────────────  │  (this service)  │
│              │   approved / rejected│                  │
└──────────────┘                      └──────────────────┘
       Both containers share a Docker bridge network (validator-net)
```

## Quick Start

### 0. Configure environment variables

Copy `.env.example` to `.env` and fill your Cloudflare values:

```bash
cp .env.example .env
```

### 1. Build & Run with Docker Compose

```bash
docker compose up --build
```

The API will be available at **http://localhost:8080**.

### 2. Test the health endpoint

```bash
curl http://localhost:8080/health
```

### 3. Validate a driving licence (file upload)

```bash
curl -X POST http://localhost:8080/validate \
  -F "file=@/path/to/driving_licence.jpg"
```

### 5. Smoke test from PowerShell (Windows)

```powershell
.\scripts\smoke-test.ps1 -ImagePath "C:\path\to\driving_licence.jpg"
```

### 4. Validate via base64 (container-to-container)

```bash
curl -X POST http://localhost:8080/validate-base64 \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-encoded-image>",
    "content_type": "image/jpeg"
  }'
```

## API Endpoints

| Method | Path               | Description                          |
|--------|--------------------|--------------------------------------|
| GET    | `/health`          | Health check                         |
| POST   | `/validate`        | Upload licence image (multipart)     |
| POST   | `/validate-base64` | Send base64-encoded image (JSON)     |
| GET    | `/docs`            | Interactive Swagger UI               |

## Response Example

```json
{
  "approved": true,
  "years_of_experience": 5.3,
  "issue_date": "15/06/2020",
  "expiry_date": "15/06/2030",
  "licence_number": "SMITH906152AB1CD",
  "holder_name": "John Smith",
  "licence_categories": "B, BE",
  "reason": "Licence issued on 15/06/2020, giving 5.3 years of experience (>= 2 years). APPROVED.",
  "raw_ai_response": "..."
}
```

## Future: Connecting the Client Container

When you build the second Docker container, it can communicate with this service using:

```
http://ai-validator:8000/validate-base64
```

Uncomment the `client-app` section in `docker-compose.yml` and set the environment variable `AI_VALIDATOR_URL=http://ai-validator:8000`.

Both containers share the `validator-net` bridge network, so they can reach each other by service name.

## Environment Variables

| Variable         | Description              |
|------------------|--------------------------|
| `CF_ACCOUNT_ID`  | Cloudflare account id (Workers AI) |
| `CF_API_TOKEN`   | Cloudflare API token with Workers AI permissions |
| `CF_AI_MODEL`    | Workers AI model name (default: `@cf/llava-hf/llava-1.5-7b-hf`) |

## Project Structure

```
AI_cloudflare/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI application
├── .dockerignore
├── .env                     # API key (not committed to git)
├── docker-compose.yml       # Multi-container orchestration
├── Dockerfile               # AI validator image
├── requirements.txt         # Python dependencies
└── README.md
```
