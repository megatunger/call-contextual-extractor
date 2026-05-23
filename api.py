import os
import uuid
import shutil
import concurrent.futures
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Pipeline imports
from pipeline.config import PipelineConfig
from pipeline.vad_segment import load_silero_vad, segment_one_call
from pipeline.asr_recognize import recognize_segment
from pipeline.transcript import merge_transcript_lines
from pipeline.inference import load_model, generate_extraction

# Global state to hold our models
class AppState:
    vad_model = None
    llm_model = None
    llm_tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup - load models
    print("Loading VAD model...")
    AppState.vad_model = load_silero_vad()
    
    print("Loading LLM model...")
    try:
        AppState.llm_model, AppState.llm_tokenizer = load_model("data/models/Qwen3.5-2B_lora")
    except FileNotFoundError as e:
        print(f"Warning: Could not load LLM. {e}")
    
    yield
    
    # Cleanup (if needed)
    pass

app = FastAPI(title="Call Contextual Extractor API", lifespan=lifespan)

@app.post("/extract")
async def extract_context(file: UploadFile = File(...)):
    if AppState.llm_model is None or AppState.llm_tokenizer is None:
        raise HTTPException(status_code=500, detail="LLM model not loaded. Check server logs.")

    # Generate a unique ID for this call
    call_id = str(uuid.uuid4())
    
    # Create PipelineConfig for this specific call
    cfg = PipelineConfig.from_repo()
    
    # Ensure raw audio and segments directories exist
    cfg.raw_audio_dir.mkdir(parents=True, exist_ok=True)
    cfg.segment_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded file
    file_extension = Path(file.filename).suffix
    if not file_extension:
        file_extension = ".wav"
    
    raw_audio_path = cfg.raw_audio_dir / f"{call_id}{file_extension}"
    
    try:
        # 1. Save uploaded audio
        with open(raw_audio_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        print(f"[{call_id}] Saved audio to {raw_audio_path}")

        # 2. VAD: Segment the audio
        print(f"[{call_id}] Running VAD...")
        segments = segment_one_call(raw_audio_path, AppState.vad_model, cfg)
        
        if not segments:
            return JSONResponse(status_code=400, content={"error": "No speech detected in the audio file."})

        # 3. ASR: Recognize each segment in parallel
        print(f"[{call_id}] Running ASR on {len(segments)} segments...")
        asr_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_seg = {
                executor.submit(recognize_segment, seg["path"], cfg): seg
                for seg in segments
            }
            for future in concurrent.futures.as_completed(future_to_seg):
                seg = future_to_seg[future]
                try:
                    asr_out = future.result()
                except Exception as exc:
                    asr_out = {"success": False, "transcription": "", "error": str(exc)}
                asr_results.append({**seg, **asr_out})

        # 4. Merge transcripts
        print(f"[{call_id}] Merging transcripts...")
        transcript_lines = merge_transcript_lines(asr_results)
        full_transcript = "\n".join(transcript_lines)
        
        if not full_transcript.strip():
            return JSONResponse(status_code=400, content={"error": "ASR could not recognize any text."})

        print(f"[{call_id}] Full Transcript:\n{full_transcript}")

        # 5. LLM Inference
        print(f"[{call_id}] Generating extraction...")
        extraction_result = generate_extraction(AppState.llm_model, AppState.llm_tokenizer, full_transcript)
        
        # 6. Return Result
        return {
            "call_id": call_id,
            "transcript": full_transcript,
            "extraction": extraction_result
        }

    except Exception as e:
        print(f"[{call_id}] Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 7. Cleanup
        print(f"[{call_id}] Cleaning up temporary files...")
        if raw_audio_path.exists():
            os.remove(raw_audio_path)
            
        call_dir = cfg.segment_output_dir / call_id
        if call_dir.exists():
            shutil.rmtree(call_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
