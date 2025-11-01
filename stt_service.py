import whisper
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("ai-service")

class StreamingSTTService:
    def __init__(self, model_name="tiny"):
        self.model = whisper.load_model(model_name)
        self.audio_buffer = bytearray()
        self.SAMPLE_RATE = 16000
        self.BUFFER_SECONDS = 5
        self.BUFFER_SIZE = self.SAMPLE_RATE * 2 * self.BUFFER_SECONDS

    def _process_buffer(self) -> str:
        if not self.audio_buffer:
            return ""

        try:
            audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            result = self.model.transcribe(audio_np, fp16=False)
            transcribed_text = result.get("text", "").strip()
            
            self.audio_buffer.clear()
            
            return transcribed_text
        except Exception as e:
            logger.error(f"Whisper transkripsiyon hatası: {e}", exc_info=True)
            self.audio_buffer.clear()
            return ""

    def process_audio_chunk(self, audio_chunk: bytes) -> Optional[str]:
        self.audio_buffer.extend(audio_chunk)
        
        if len(self.audio_buffer) >= self.BUFFER_SIZE:
            return self._process_buffer()
            
        return None

    def finalize_stream(self) -> str:
        return self._process_buffer()