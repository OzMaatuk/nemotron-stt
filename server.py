"""
Nemotron 3.5 — Batch API Server (corrected)
--------------------------------------------
Drop-in OpenAI Whisper-compatible endpoint.
Fixes vs. naive version:
  1. tempfile delete=False + explicit unlink (Linux open-file unlink bug)
  2. Safe .transcribe() return: handles both str and Hypothesis objects
  3. Single model load at startup, reused across requests

Install:
    pip install torch torchaudio nemo_toolkit[asr] fastapi uvicorn python-multipart soundfile

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST http://localhost:8000/v1/audio/transcriptions -F file=@sample.wav
"""

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
import nemo.collections.asr as nemo_asr

app = FastAPI()

# ──────────────────────────────────────────────────────────────
# Load once at startup
# ──────────────────────────────────────────────────────────────
MODEL_NAME = "nvidia/nemotron-3.5-asr-streaming-0.6b"

print(f"Loading {MODEL_NAME} …")
model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
model.eval()
print("Model ready.")


def _extract_text(result) -> str:
    """
    .transcribe() returns either:
      - List[str]            (older NeMo builds)
      - List[Hypothesis]     (newer NeMo builds, has .text attribute)
    Handle both safely.
    """
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):       # Hypothesis object
        return result.text
    return str(result)


# ──────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────
@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...)):
    # Write to temp file with delete=False so NeMo can open it by path
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())
        # File is now closed — NeMo can safely open it by name
        results = model.transcribe([tmp_path])
        text = _extract_text(results[0])
        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)          # always clean up
