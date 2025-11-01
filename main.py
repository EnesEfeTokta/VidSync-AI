import os
import uuid
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

import httpx
from pydantic import BaseModel, ValidationError, parse_obj_as
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from config import ayarlar
from logging_setup import loglamayi_kur
from stt_service import StreamingSTTService

loglamayi_kur()
logger = logging.getLogger("ai-service")

app = FastAPI(
    title="AI Meeting Summarization & Transcription Service",
    description="Toplantı dökümlerini özetlemek ve gerçek zamanlı transripsiyon yapmak için bir API.",
    version="1.9.0"
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

class Reminder(BaseModel):
    event: str
    date: Optional[str] = None
    time: Optional[str] = None

class ParsedSummaryResponse(BaseModel):
    summary: str
    reminders: List[Reminder]
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

def parse_llm_response(llm_response_text: str) -> Tuple[str, List[Reminder]]:
    summary = ""
    reminders = []
    try:
        json_match = re.search(r'\[.*\]', llm_response_text, re.DOTALL)
        if not json_match:
            return (llm_response_text.strip(), [])
        json_string = json_match.group(0)
        summary = llm_response_text.replace(json_string, "").strip()
        json_data = json.loads(json_string)
        reminders = parse_obj_as(List[Reminder], json_data)
        return (summary, reminders)
    except (json.JSONDecodeError, ValidationError):
        return (llm_response_text.strip(), [])
    except Exception as e:
        logger.error(f"Ayrıştırma sırasında beklenmedik hata: {e}", exc_info=True)
        return (llm_response_text.strip(), [])

def format_transcript(payload: MeetingPayload) -> str:
    transcript_lines = []
    participant_names = {p.participant_id: p.full_name for p in payload.participants}
    for entry in payload.transcript:
        speaker_name = participant_names.get(entry.speaker_id, "Bilinmeyen Katılımcı")
        transcript_lines.append(f"{speaker_name}: {entry.text}")
    return "\n".join(transcript_lines)

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
        response = await client.post(ollama_url, json={"model": model_name, "prompt": full_prompt, "stream": False})
        response.raise_for_status()
        response_data = response.json()
        return response_data.get("response", "").strip()

async def generate_structured_summary_with_ollama(transcript: str, conversation_id: str) -> StructuredSummary:
    system_prompt = (
        "Sen, toplantı dökümlerini analiz eden uzman bir AI asistanısın. Görevin, aşağıdaki konuşmayı analiz edip iki bilgi çıkarmaktır:\n"
        "1. Konuşmanın ana noktalarını içeren kısa bir özet.\n"
        "2. Konuşmada bahsedilen tüm belirli etkinlikleri, ilgili tarih ve saatleriyle birlikte bir liste halinde.\n\n"
        "Yanıtını SADECE geçerli bir JSON formatında vermelisin. JSON nesnesinden önce kesinlikle hiçbir metin ekleme.\n"
        'Kullanman gereken JSON yapısı şu şekildedir:\n'
        '{"summary": "...", "extracted_entities": [{"event": "...", "date": "...", "time": "..."}]}'
    )
    full_prompt = f"{system_prompt}\n\n---\n\n{transcript}"
    ollama_url = ayarlar.OLLAMA_BASE_URL + "/generate"
    model_name = ayarlar.OLLAMA_MODEL_ADI
    logger.info(f"Ollama'ya yapılandırılmış özet isteği gönderiliyor. conversationId: {conversation_id}")
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(ollama_url, json={"model": model_name, "prompt": full_prompt, "stream": False})
        response.raise_for_status()
        response_data = response.json()
        llm_output_str = response_data.get("response", "").strip()
        json_data = json.loads(llm_output_str)
        return StructuredSummary.model_validate(json_data)

async def generate_unstructured_summary_for_parsing(transcript: str, conversation_id: str) -> str:
    system_prompt = (
        "Aşağıdaki konuşmanın kısa bir özetini yaz. Ardından, konuşmada geçen tüm etkinlikleri, tarihleri ve saatleri içeren bir JSON listesi ekle. JSON listesinin formatı şu şekilde olmalı: "
        '[{"event": "...", "date": "...", "time": "..."}]. Eğer etkinlik yoksa, boş bir liste [] ekle.'
    )
    full_prompt = f"{system_prompt}\n\n---\n\n{transcript}"
    ollama_url = ayarlar.OLLAMA_BASE_URL + "/generate"
    model_name = ayarlar.OLLAMA_MODEL_ADI
    logger.info(f"Ollama'ya ayrıştırılacak özet isteği gönderiliyor. conversationId: {conversation_id}")
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(ollama_url, json={"model": model_name, "prompt": full_prompt, "stream": False})
        response.raise_for_status()
        response_data = response.json()
        return response_data.get("response", "").strip()

@app.get("/")
async def root():
    return {"message": "AI Service is running"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/summarize", response_model=SimpleSummaryResponse, summary="Bir Konuşmanın Basit Özetini Yapar (Geriye Uyumlu)")
async def summarize_endpoint_simple(payload: MeetingPayload):
    conversation_id = str(uuid.uuid4())
    transcript = format_transcript(payload)
    simple_summary = await generate_simple_summary_with_ollama(transcript, conversation_id)
    return SimpleSummaryResponse(summary=simple_summary, participants=payload.participants)

@app.post("/summarize-structured", response_model=DetailedSummaryResponse, summary="Bir Konuşmayı Özetler ve Bilgileri Çıkarır (Sadece JSON)")
async def summarize_endpoint_structured(payload: MeetingPayload):
    conversation_id = str(uuid.uuid4())
    transcript = format_transcript(payload)
    structured_summary = await generate_structured_summary_with_ollama(transcript, conversation_id)
    return DetailedSummaryResponse(structured_summary=structured_summary, participants=payload.participants)

@app.post("/summarize-and-parse", response_model=ParsedSummaryResponse, summary="Bir Konuşmayı Özetler ve Yanıtı Ayrıştırır")
async def summarize_endpoint_with_parsing(payload: MeetingPayload):
    conversation_id = str(uuid.uuid4())
    transcript = format_transcript(payload)
    raw_response = await generate_unstructured_summary_for_parsing(transcript, conversation_id)
    summary, reminders = parse_llm_response(raw_response)
    return ParsedSummaryResponse(summary=summary, reminders=reminders, participants=payload.participants)

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
                await websocket.send_json({"text": transcribed_text})
    except WebSocketDisconnect:
        logger.warning(f"WebSocket bağlantısı istemci tarafından kapatıldı: {connection_id}")
    finally:
        final_text = stt_service.finalize_stream()
        if final_text:
            await websocket.send_json({"text": final_text})
        logger.info(f"WebSocket bağlantısı sonlandırılıyor: {connection_id}")