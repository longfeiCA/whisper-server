import tempfile
import os
import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
from faster_whisper import WhisperModel

app = FastAPI(title="Whisper API Server")

# Load fixed model: always 'small' (ignore all incoming model parameters)
# Use int8 quantization on CPU for minimal memory usage
print("Loading Whisper model 'small' with faster-whisper (int8)...")
model_name = "small"
model = WhisperModel(model_name, device="cpu", compute_type="int8")
print(f"Model {model_name} loaded!")


@app.get("/")
def root():
    return {"status": "ok", "model": model_name}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "model_name": model_name}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None)
):
    """
    Transcribe audio file (ignores incoming model/model_param etc parameters).
    Only uses the fixed backend model: small (int8 quantization via faster-whisper)
    """
    start_time = time.time()
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe with faster-whisper
        segments, info = model.transcribe(tmp_path, language=language)
        os.unlink(tmp_path)

        # Collect segments into full text
        text_parts = []
        segment_list = list(segments)
        for seg in segment_list:
            text_parts.append(seg.text)
        full_text = "".join(text_parts)

        processing_time = round(time.time() - start_time, 3)
        return JSONResponse(content={
            "text": full_text,
            "language": info.language if info.language else (language or "auto"),
            "segments": len(segment_list),
            "processing_time": processing_time,
            "model": model_name
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)