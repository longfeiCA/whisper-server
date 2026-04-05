FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    openai-whisper

# Create app directory
WORKDIR /app

# Create the FastAPI server script
RUN cat > server.py << 'EOF'
import whisper
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Whisper API Server")

# Load model
print("Loading Whisper model...")
model_name = "small"
model = whisper.load_model(model_name)
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
    model_param: str = Form("base"),
    language: str = Form(None)
):
    """Transcribe audio file"""
    import time
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Transcribe
        result = model.transcribe(tmp_path, language=language, fp16=False)
        
        # Cleanup
        os.unlink(tmp_path)
        
        # Calculate processing time
        processing_time = round(time.time() - start_time, 3)
        
        return JSONResponse(content={
            "text": result["text"],
            "language": result.get("language", language or "auto"),
            "segments": len(result.get("segments", [])),
            "processing_time": processing_time
        })
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        return JSONResponse(
            content={"error": str(e), "detail": error_detail},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget -q --method=GET -O /dev/null http://localhost:8000/health || exit 1

CMD ["python", "server.py"]
