import os
import uuid
import json
import logging
from typing import List, Dict, Any, Optional

import httpx
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from .config import ayarlar
from .logging_setup import loglamayi_kur

loglamayi_kur()
logger = logging.getLogger("ai-service")

app = FastAPI(
    title="AI Meeting Summarization & Transcription Service",
    description="Toplantı dökümlerini özetlemek ve gerçek zamanlı transkripsiyon yapmak için bir API.",
    version="1.4.0"
)

class Location(BaseModel):
    city: str
    country: str

class Participant(BaseModel):
    participant_id: str
    full_name: str
    email: Optional[str] = None
    role: Optional[str] = None
    is_moderator: Optional[bool] = False
    timezone: Optional[str] = None
    utc_offset: Optional[str] = None
    device: Optional[str] = None
    location: Optional[Location] = None

class TranscriptEntry(BaseModel):
    start_time: float
    end_time: float
    speaker_id: str
    text: str

class MeetingPayload(BaseModel):
    participants: List[Participant]
    transcript: List[TranscriptEntry]
    metadata: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    message_id: int
    timestamp: str
    sender_id: str
    message: str

class ChatPayload(BaseModel):
    participants: List[Participant]
    chat_history: List[ChatMessage]
    metadata: Optional[Dict[str, Any]] = None

class DetailedSummaryResponse(BaseModel):
    summary: str
    participants: List[Participant]

def format_transcript(payload: MeetingPayload) -> str:
    transcript_lines = []
    participant_names = {p.participant_id: p.full_name for p in payload.participants}
    for entry in payload.transcript:
        speaker_name = participant_names.get(entry.speaker_id, "Bilinmeyen Katılımcı")
        transcript_lines.append(f"{speaker_name}: {entry.text}")
    return "\n".join(transcript_lines)

def format_chat_history(payload: ChatPayload) -> str:
    chat_lines = []
    participant_names = {p.participant_id: p.full_name for p in payload.participants}
    for message in payload.chat_history:
        sender_name = participant_names.get(message.sender_id, "Bilinmeyen Katılımcı")
        chat_lines.append(f"{sender_name}: {message.message}")
    return "\n".join(chat_lines)

async def generate_summary_with_ollama(transcript: str, conversation_id: str) -> str:
    system_prompt = (
        "Sen uzman bir toplantı asistanısın. Lütfen aşağıdaki konuşmanın kısa ve profesyonel bir özetini çıkar. "
        "Konuşulan ana konulara, alınan kararlara ve belirlenen eylem maddelerine odaklan. "
        "Özet en fazla 4 cümle olmalıdır. İşte konuşma dökümü:"
    )
    full_prompt = f"{system_prompt}\n\n---\n\n{transcript}"
    ollama_url = ayarlar.OLLAMA_BASE_URL + "/generate"
    model_name = ayarlar.OLLAMA_MODEL_ADI
    logger.info(f"Ollama'ya istek gönderiliyor. URL: {ollama_url}, Model: {model_name}, conversationId: {conversation_id}")
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                ollama_url,
                json={"model": model_name, "prompt": full_prompt, "stream": False}
            )
            response.raise_for_status()
            response_data = response.json()
            summary_text = response_data.get("response", "").strip()
            if not summary_text:
                logger.error(f"Ollama'dan boş bir özet yanıtı alındı. conversationId: {conversation_id}")
                raise ValueError("Ollama'dan boş bir özet yanıtı alındı.")
            logger.info(f"Ozet başarılı bir şekilde oluşturuldu. conversationId: {conversation_id}")
            return summary_text
        except httpx.ConnectError:
            logger.error(f"Ollama servisine bağlanırken ağ hatası oluştu (503). conversationId: {conversation_id}", exc_info=True)
            raise HTTPException(status_code=503, detail="Özetleme servisine ulaşılamıyor.")
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API'den hata yanıtı alındı (API Status: {e.response.status_code}). Detay: {e.response.text}. conversationId: {conversation_id}", exc_info=True)
            raise HTTPException(status_code=500, detail="Özetleme servisinde bir hata oluştu.")
        except Exception:
            logger.error(f"Özetleme sırasında beklenmedik bir hata oluştu (500). conversationId: {conversation_id}", exc_info=True)
            raise HTTPException(status_code=500, detail="Özetleme sırasında beklenmedik bir sunucu hatası oluştu.")

@app.get("/")
async def root():
    return {"message": "AI Service is running"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/summarize", response_model=DetailedSummaryResponse, summary="Bir Sesli Konuşmayı Özetler")
async def summarize_conversation_endpoint(payload: MeetingPayload):
    conversation_id = str(uuid.uuid4())
    logger.info(f"Yeni sesli transkript özet isteği alındı. conversationId: {conversation_id}")
    transcript = format_transcript(payload)
    if not transcript:
        raise HTTPException(status_code=400, detail="Geçerli bir konuşma dökümü bulunamadı.")
    
    dynamic_summary = await generate_summary_with_ollama(transcript, conversation_id)
    return DetailedSummaryResponse(summary=dynamic_summary, participants=payload.participants)

@app.post("/summarize-chat", response_model=DetailedSummaryResponse, summary="Bir Metin Sohbetini Özetler")
async def summarize_chat_endpoint(payload: ChatPayload):
    conversation_id = str(uuid.uuid4())
    logger.info(f"Yeni sohbet geçmişi özet isteği alındı. conversationId: {conversation_id}")
    formatted_chat = format_chat_history(payload)
    if not formatted_chat:
        raise HTTPException(status_code=400, detail="Geçerli bir sohbet geçmişi bulunamadı.")
    
    dynamic_summary = await generate_summary_with_ollama(formatted_chat, conversation_id)
    return DetailedSummaryResponse(summary=dynamic_summary, participants=payload.participants)

@app.websocket("/transcribe")
async def transcribe_endpoint(websocket: WebSocket):
    connection_id = str(uuid.uuid4())
    await websocket.accept()
    logger.info(f"WebSocket bağlantısı kuruldu: {connection_id}")
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            logger.info(f"Bağlantı {connection_id} üzerinden {len(audio_data)} byte ses verisi alındı.")
    except WebSocketDisconnect:
        logger.warning(f"WebSocket bağlantısı istemci tarafından kapatıldı: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket hatası {connection_id}: {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket bağlantısı sonlandırılıyor: {connection_id}")