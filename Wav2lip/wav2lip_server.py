from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
from uuid import uuid4
import wave
import io
import logging

app = FastAPI()

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


def run_wav2lip(checkpoint_path: str, video_path: str, audio_path: str, output_path: str):
    try:
        command = [
            'python', 'inference_trt.py',
            '--checkpoint_path', checkpoint_path,
            '--face', video_path,
            '--audio', audio_path,
            '--outfile', output_path
        ]
        subprocess.run(command, check=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"Wav2Lip command failed: {e.output}")
        raise

@app.post("/process-audio-bytes/")
async def process_audio_bytes(request: Request):
    audio_filename = f"{uuid4()}.wav"
    audio_cache_dir = "static/audio_cache"
    audio_path = os.path.join(audio_cache_dir, audio_filename)

    try:
        audio_bytes = await request.body()
        
        os.makedirs(audio_cache_dir, exist_ok=True)
        
        with open(audio_path, 'wb') as f_out:
            f_out.write(audio_bytes)
        
        logger.info(f"Audio file saved: {audio_path}")

        checkpoint_path = "/root/wav2lip/Wav2Lip/wav2lip_dynamic.engine"
        video_path = "/root/wav2lip/Wav2Lip/data/real_nurse.mp4"
        
        processed_video_filename = f"{uuid4()}.mp4"
        processed_video_path = f"static/{processed_video_filename}"
        
        run_wav2lip(checkpoint_path, video_path, audio_path, processed_video_path)
        
        if not os.path.exists(processed_video_path):
            raise HTTPException(status_code=500, detail="视频处理失败，文件未生成。")
        
        video_url = f"/static/{processed_video_filename}"
        return {"message": "视频处理成功", "video_url": video_url}
    except Exception as e:
        logger.error(f"Error processing audio bytes: {e}")
        raise HTTPException(status_code=500, detail=f"处理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5090)