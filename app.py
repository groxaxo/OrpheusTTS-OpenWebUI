"""
Orpheus-FastAPI Server for Text-to-Speech

A high-performance, FastAPI-based TTS server with OpenAI-compatible API endpoints.
Based on patterns from Lex-au/Orpheus-FastAPI with text sanitization from Kokoro-FastAPI.

Features:
- OpenAI-compatible /v1/audio/speech endpoint
- Support for any Orpheus GGUF model loaded in LMStudio
- Text sanitization and normalization
- Web UI for testing
- Long-form audio generation with batching
- Multiple voice support
"""

import os
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

# Ensure .env file exists from example
def ensure_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        try:
            default_env = {}
            with open(".env.example", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key = line.split("=")[0].strip()
                        default_env[key] = line.split("=", 1)[1].strip()
            
            final_env = default_env.copy()
            for key in default_env:
                if key in os.environ:
                    final_env[key] = os.environ[key]
            
            with open(".env", "w") as f:
                for key, value in final_env.items():
                    f.write(f"{key}={value}\n")
            
            print("✅ Created .env file from .env.example")
        except Exception as e:
            print(f"⚠️ Error creating .env file: {e}")

ensure_env_file()
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tts_engine import (
    generate_speech_from_api,
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    VOICE_TO_LANGUAGE,
    AVAILABLE_LANGUAGES,
    SAMPLE_RATE,
    TextSanitizer,
    NormalizationOptions,
)

# Create FastAPI app
app = FastAPI(
    title="Orpheus-FastAPI",
    description="High-performance Text-to-Speech server using Orpheus TTS with LMStudio integration",
    version="1.0.0",
)

# Ensure output directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
try:
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
except RuntimeError:
    pass

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    pass


# Request/Response models
class SpeechRequest(BaseModel):
    """OpenAI-compatible speech request."""
    input: str = Field(..., description="The text to convert to speech")
    model: str = Field(default="orpheus", description="Model identifier (ignored, uses loaded model)")
    voice: str = Field(default=DEFAULT_VOICE, description="Voice to use for synthesis")
    response_format: str = Field(default="wav", description="Audio format (currently only wav supported)")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed multiplier")


class TTSRequest(BaseModel):
    """Extended TTS request with additional options."""
    text: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default=DEFAULT_VOICE, description="Voice to use")
    temperature: float = Field(default=0.6, ge=0.1, le=2.0, description="Generation temperature")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p sampling")
    max_tokens: int = Field(default=8192, ge=128, le=16384, description="Maximum tokens")
    sanitize: bool = Field(default=True, description="Apply text sanitization")


class VoiceInfo(BaseModel):
    """Voice information."""
    name: str
    language: str


# OpenAI-compatible endpoint
@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """
    Generate speech from text using the Orpheus TTS model.
    Compatible with OpenAI's /v1/audio/speech endpoint.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{request.voice}_{timestamp}.wav"
    
    # Use batching for long texts
    use_batching = len(request.input) > 1000
    if use_batching:
        print(f"Using batched generation for long text ({len(request.input)} characters)")
    
    try:
        start = time.time()
        generate_speech_from_api(
            prompt=request.input,
            voice=request.voice,
            output_file=output_path,
            use_batching=use_batching,
            max_batch_chars=1000
        )
        end = time.time()
        
        print(f"Generated speech in {end - start:.2f} seconds")
        
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=f"{request.voice}_{timestamp}.wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")


@app.get("/v1/audio/voices")
async def list_voices():
    """Return list of available voices."""
    voices = [
        VoiceInfo(name=voice, language=VOICE_TO_LANGUAGE.get(voice, "unknown"))
        for voice in AVAILABLE_VOICES
    ]
    
    return JSONResponse(
        content={
            "status": "ok",
            "default_voice": DEFAULT_VOICE,
            "voices": [v.dict() for v in voices],
            "languages": AVAILABLE_LANGUAGES
        }
    )


# Legacy endpoint for compatibility
@app.post("/speak")
async def speak(request: TTSRequest):
    """Legacy endpoint for compatibility with existing clients."""
    if not request.text:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'text'"}
        )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{request.voice}_{timestamp}.wav"
    
    use_batching = len(request.text) > 1000
    
    try:
        start = time.time()
        generate_speech_from_api(
            prompt=request.text,
            voice=request.voice,
            output_file=output_path,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            use_batching=use_batching,
            max_batch_chars=1000
        )
        end = time.time()
        
        return JSONResponse(content={
            "status": "ok",
            "voice": request.voice,
            "output_file": output_path,
            "generation_time": round(end - start, 2)
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Speech generation failed: {str(e)}"}
        )


@app.post("/sanitize")
async def sanitize_text(text: str, normalize: bool = True):
    """
    Preview text sanitization without generating speech.
    
    Args:
        text: Input text to sanitize
        normalize: Whether to apply normalization
        
    Returns:
        Original and sanitized text
    """
    options = NormalizationOptions(normalize=normalize)
    sanitizer = TextSanitizer(options)
    sanitized = sanitizer.sanitize(text)
    
    return JSONResponse(content={
        "original": text,
        "sanitized": sanitized,
        "normalize_enabled": normalize
    })


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "ok",
        "service": "orpheus-fastapi",
        "sample_rate": SAMPLE_RATE
    })


@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple web UI for testing."""
    voices_html = "\n".join([
        f'<option value="{v}" {"selected" if v == DEFAULT_VOICE else ""}>{v} ({VOICE_TO_LANGUAGE.get(v, "unknown")})</option>'
        for v in AVAILABLE_VOICES
    ])
    
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Orpheus TTS</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{ color: #6c63ff; }}
        textarea {{
            width: 100%;
            height: 150px;
            padding: 12px;
            border: 1px solid #333;
            border-radius: 8px;
            background: #16213e;
            color: #eee;
            font-size: 14px;
            resize: vertical;
        }}
        select, button {{
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 14px;
            cursor: pointer;
            margin: 10px 5px 10px 0;
        }}
        select {{
            background: #16213e;
            color: #eee;
            border: 1px solid #333;
        }}
        button {{
            background: #6c63ff;
            color: white;
        }}
        button:hover {{ background: #5a52e0; }}
        button:disabled {{
            background: #444;
            cursor: not-allowed;
        }}
        .status {{
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            background: #16213e;
        }}
        audio {{
            width: 100%;
            margin-top: 15px;
        }}
        .emotions {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .emotions code {{
            background: #0f3460;
            padding: 3px 8px;
            border-radius: 4px;
            margin-right: 5px;
        }}
    </style>
</head>
<body>
    <h1>🎙️ Orpheus TTS</h1>
    <p>Text-to-Speech with OpenAI-compatible API</p>
    
    <textarea id="text" placeholder="Enter text to convert to speech...">Hello! This is Orpheus text-to-speech. I can speak with natural emotion and expression.</textarea>
    
    <div>
        <select id="voice">
            {voices_html}
        </select>
        <button id="generate" onclick="generateSpeech()">🔊 Generate Speech</button>
    </div>
    
    <div class="status" id="status">
        <p>Ready to generate speech.</p>
    </div>
    
    <div id="audioContainer"></div>
    
    <div class="emotions">
        <h3>Emotion Tags</h3>
        <p>Add emotion to your speech:</p>
        <p>
            <code>&lt;laugh&gt;</code>
            <code>&lt;chuckle&gt;</code>
            <code>&lt;sigh&gt;</code>
            <code>&lt;cough&gt;</code>
            <code>&lt;sniffle&gt;</code>
            <code>&lt;groan&gt;</code>
            <code>&lt;yawn&gt;</code>
            <code>&lt;gasp&gt;</code>
        </p>
    </div>
    
    <script>
        async function generateSpeech() {{
            const text = document.getElementById('text').value;
            const voice = document.getElementById('voice').value;
            const btn = document.getElementById('generate');
            const status = document.getElementById('status');
            const container = document.getElementById('audioContainer');
            
            if (!text.trim()) {{
                status.innerHTML = '<p style="color: #ff6b6b;">Please enter some text.</p>';
                return;
            }}
            
            btn.disabled = true;
            btn.textContent = '⏳ Generating...';
            status.innerHTML = '<p>Generating speech, please wait...</p>';
            
            try {{
                const response = await fetch('/v1/audio/speech', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ input: text, voice: voice }})
                }});
                
                if (!response.ok) {{
                    throw new Error(`HTTP ${{response.status}}`);
                }}
                
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                
                container.innerHTML = `<audio controls autoplay src="${{url}}"></audio>`;
                status.innerHTML = '<p style="color: #4ade80;">✅ Speech generated successfully!</p>';
            }} catch (error) {{
                status.innerHTML = `<p style="color: #ff6b6b;">❌ Error: ${{error.message}}</p>`;
                console.error('Error:', error);
            }} finally {{
                btn.disabled = false;
                btn.textContent = '🔊 Generate Speech';
            }}
        }}
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("ORPHEUS_HOST", "0.0.0.0")
    port = int(os.environ.get("ORPHEUS_PORT", "5005"))
    
    print(f"🚀 Starting Orpheus-FastAPI server on {host}:{port}")
    print(f"📖 API docs available at http://{host}:{port}/docs")
    print(f"🌐 Web UI available at http://{host}:{port}/")
    
    uvicorn.run(app, host=host, port=port)
