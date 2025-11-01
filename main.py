import os
import uuid
import json
import logging
from typing import List, Dict, Any, Optional

import httpx
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from config import ayarlar
from logging_setup import loglamayi_kur
from stt_service import StreamingSTTService

loglamayi_kur()
logger = logging.getLogger("ai-service")

app = FastAPI(
    title="AI Meeting Summarization & Transcription Service",
    description="Toplantı dökümlerini özetlemek ve gerçek zamanlı transkripsiyon yapmak için bir API.",
    version="1.8.0"
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

class SimpleSummaryResponse(BaseModel):
    summary: str
    participants: List[Participant]

class ExtractedEntity(BaseModel):
    event: str
    date: Optional[str] = None
    time: Optional[str] = None

class StructuredSummary(BaseModel):
    summary: str
    extracted_entities: List[ExtractedEntity]

class DetailedSummaryResponse(BaseModel):
    structured_summary: StructuredSummary
    participants: List[Participant]

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

async def generate_simple_summary_with_ollama(transcript: str, conversation_id: str) -> str:
    system_prompt = (
        "Sen uzman bir toplantı asistanısın. Lütfen aşağıdaki konuşmanın kısa ve profesyonel bir özetini çıkar. "
        "Konuşulan ana konulara, alınan kararlara ve belirlenen eylem maddelerine odaklan. "
        "Özet en fazla 4 cümle olmalıdır. İşte konuşma dökümü:"
    )
    full_prompt = f"{system_prompt}\n\n---\n\n{transcript}"
    ollama_url = ayarlar.OLLAMA_BASE_URL + "/generate"
    model_name = ayarlar.OLLAMA_MODEL_ADI
    logger.info(f"Ollama'ya basit özet isteği gönderiliyor. conversationId: {conversation_id}")
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
                raise ValueError("Ollama'dan boş bir özet yanıtı alındı.")
            logger.info(f"Basit özet başarılı bir şekilde oluşturuldu. conversationId: {conversation_id}")
            return summary_text
        except Exception:
            logger.error(f"Basit özetleme sırasında beklenmedik bir hata oluştu. conversationId: {conversation_id}", exc_info=True)
            raise HTTPException(status_code=500, detail="Basit özetleme sırasında beklenmedik bir sunucu hatası oluştu.")

async def generate_structured_summary_with_ollama(transcript: str, conversation_id: str) -> StructuredSummary:
    system_prompt = (
        "Sen, toplantı dökümlerini analiz eden uzman bir AI asistanısın. Görevin, aşağıdaki konuşmayı analiz edip iki bilgi çıkarmaktır:\n"
        "1. Konuşmanın ana noktalarını, kararlarını ve eylem maddelerini içeren kısa ve profesyonel bir özet (en fazla 4 cümle).\n"
        "2. Konuşmada bahsedilen tüm belirli etkinlikleri, randevuları veya son tarihleri, ilgili tarih ve saatleriyle birlikte bir liste halinde.\n\n"
        "Yanıtını SADECE geçerli bir JSON formatında vermelisin. JSON nesnesinden önce veya sonra kesinlikle hiçbir metin ekleme.\n"
        "Kullanman gereken JSON yapısı şu şekildedir:\n"
        "{\n"
        '  "summary": "Buraya toplantının kısa özeti gelecek.",\n'
        '  "extracted_entities": [\n'
        "    {\n"
        '      "event": "Etkinliğin açıklaması, örn. \'Proje demo toplantısı\'",\n'
        '      "date": "Çıkarılan tarih, örn. \'2025-11-02\'",\n'
        '      "time": "Çıkarılan saat, örn. \'15:00\'"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Eğer konuşmada hiçbir etkinlik, tarih veya saat belirtilmemişse, `extracted_entities` için boş bir liste `[]` döndür.\n"
        "İşte analiz edilecek konuşma dökümü:"
    )
    full_prompt = f"{system_prompt}\n\n---\n\n{transcript}"
    ollama_url = ayarlar.OLLAMA_BASE_URL + "/generate"
    model_name = ayarlar.OLLAMA_MODEL_ADI
    logger.info(f"Ollama'ya yapılandırılmış özet isteği gönderiliyor. conversationId: {conversation_id}")
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                ollama_url,
                json={"model": model_name, "prompt": full_prompt, "stream": False}
            )
            response.raise_for_status()
            response_data = response.json()
            llm_output_str = response_data.get("response", "").strip()
            if not llm_output_str:
                raise ValueError("Ollama'dan boş bir yanıt alındı.")
            json_data = json.loads(llm_output_str)
            parsed_summary = StructuredSummary.model_validate(json_data)
            logger.info(f"Yapılandırılmış özet başarılı bir şekilde oluşturuldu ve doğrulandı. conversationId: {conversation_id}")
            return parsed_summary
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Ollama'dan gelen yanıt JSON formatında değil veya modelle uyuşmuyor. Yanıt: '{llm_output_str}'. Hata: {e}. conversationId: {conversation_id}", exc_info=True)
            raise HTTPException(status_code=500, detail="Özetleme servisinden geçersiz formatlı bir yanıt alındı.")
        except Exception:
            logger.error(f"Yapılandırılmış özetleme sırasında beklenmedik bir hata oluştu. conversationId: {conversation_id}", exc_info=True)
            raise HTTPException(status_code=500, detail="Yapılandırılmış özetleme sırasında beklenmedik bir sunucu hatası oluştu.")

@app.get("/")
async def root():
    return {"message": "AI Service is running"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/summarize", response_model=SimpleSummaryResponse, summary="Bir Konuşmanın Basit Özetini Yapar (Geriye Uyumlu)")
async def summarize_endpoint_simple(payload: MeetingPayload):
    conversation_id = str(uuid.uuid4())
    logger.info(f"Yeni basit özet isteği alındı. conversationId: {conversation_id}")
    transcript = format_transcript(payload)
    if not transcript:
        raise HTTPException(status_code=400, detail="Geçerli bir konuşma dökümü bulunamadı.")
    simple_summary = await generate_simple_summary_with_ollama(transcript, conversation_id)
    return SimpleSummaryResponse(summary=simple_summary, participants=payload.participants)

@app.post("/summarize-structured", response_model=DetailedSummaryResponse, summary="Bir Konuşmayı Özetler ve Bilgileri Çıkarır")
async def summarize_endpoint_structured(payload: MeetingPayload):
    conversation_id = str(uuid.uuid4())
    logger.info(f"Yeni yapılandırılmış özet isteği alındı. conversationId: {conversation_id}")
    transcript = format_transcript(payload)
    if not transcript:
        raise HTTPException(status_code=400, detail="Geçerli bir konuşma dökümü bulunamadı.")
    structured_summary = await generate_structured_summary_with_ollama(transcript, conversation_id)
    return DetailedSummaryResponse(structured_summary=structured_summary, participants=payload.participants)

@app.post("/summarize-chat", response_model=SimpleSummaryResponse, summary="Bir Metin Sohbetinin Basit Özetini Yapar")
async def summarize_chat_endpoint(payload: ChatPayload):
    conversation_id = str(uuid.uuid4())
    logger.info(f"Yeni sohbet geçmişi özet isteği alındı. conversationId: {conversation_id}")
    formatted_chat = format_chat_history(payload)
    if not formatted_chat:
        raise HTTPException(status_code=400, detail="Geçerli bir sohbet geçmişi bulunamadı.")
    simple_summary = await generate_simple_summary_with_ollama(formatted_chat, conversation_id)
    return SimpleSummaryResponse(summary=simple_summary, participants=payload.participants)

@app.websocket("/transcribe")
async def transcribe_endpoint(websocket: WebSocket):
    connection_id = str(uuid.uuid4())
    await websocket.accept()
    logger.info(f"WebSocket bağlantısı kuruldu: {connection_id}")
    stt_service = StreamingSTTService(model_name="tiny")
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            transcribed_text = stt_service.process_audio_chunk(audio_data)
            if transcribed_text:
                logger.info(f"[Transkript - {connection_id}]: {transcribed_text}")
                await websocket.send_json({"text": transcribed_text})
    except WebSocketDisconnect:
        logger.warning(f"WebSocket bağlantısı istemci tarafından kapatıldı: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket hatası {connection_id}: {e}", exc_info=True)
    finally:
        final_text = stt_service.finalize_stream()
        if final_text:
            logger.info(f"[Son Transkript - {connection_id}]: {final_text}")
            await websocket.send_json({"text": final_text})
        logger.info(f"WebSocket bağlantısı sonlandırılıyor: {connection_id}")