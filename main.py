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
    description="Toplantı dökümlerini özetlemek, transkripsiyon yapmak ve soruları yanıtlamak için bir API.",
    version="2.1.0"
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

class Reminder(BaseModel):
    event: str
    date: Optional[str] = None
    time: Optional[str] = None

class SummaryWithRemindersResponse(BaseModel):
    summary: str
    reminders: List[Reminder]
    participants: List[Participant]

class SimpleSummaryResponse(BaseModel):
    summary: str
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

class ChatbotQuery(BaseModel):
    query: str

class ChatbotResponse(BaseModel):
    response: str

def parse_llm_response(llm_response_text: str) -> Tuple[str, List[Reminder]]:
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
        "Özet, konuşmanın ana temalarını, önemli noktalarını ve sonuçlarını yansıtmalıdır."
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

async def generate_summary_and_reminders(transcript: str, conversation_id: str) -> str:
    system_prompt = (
        "Aşağıdaki konuşmanın kısa bir özetini yaz. Ardından, konuşmada geçen tüm etkinlikleri, tarihleri ve saatleri içeren bir JSON listesi ekle. JSON listesinin formatı şu şekilde olmalı: "
        '[{"event": "...", "date": "...", "time": "..."}]. Eğer etkinlik yoksa, boş bir liste [] ekle.'
    )
    full_prompt = f"{system_prompt}\n\n---\n\n{transcript}"
    ollama_url = ayarlar.OLLAMA_BASE_URL + "/generate"
    model_name = ayarlar.OLLAMA_MODEL_ADI
    logger.info(f"Ollama'ya ayrıştırılacak özet isteği gönderiliyor. conversationId: {conversation_id}")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                ollama_url,
                json={"model": model_name, "prompt": full_prompt, "stream": False}
            )
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "").strip()
        except Exception as e:
            logger.error(f"LLM isteği sırasında hata oluştu: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Özetleme servisiyle iletişimde bir hata oluştu.")

async def generate_chatbot_response(user_query: str, conversation_id: str) -> str:
    system_prompt = (
        "Sen VidSync Assistant'sın; VidSync platformu hakkında soruları yanıtlamak için tasarlanmış yardımsever ve dost canlısı bir yapay zekasın. "
        "VidSync, ekiplerin uzaktan birlikte çalışmasına yardımcı olan bir video konferans ve iş birliği aracıdır. Yüksek kaliteli video, gerçek zamanlı transkripsiyon ve yapay zeka destekli özetler gibi özelliklere sahiptir. "
        "Cevaplarını kısa ve anlaşılır tut. Her zaman pozitif ve teşvik edici bir ton kullan."
    )
    full_prompt = f"{system_prompt}\n\n---\n\nKullanıcı Sorusu: {user_query}\n\nAsistan:"
    ollama_url = ayarlar.OLLAMA_BASE_URL + "/generate"
    model_name = ayarlar.OLLAMA_MODEL_ADI
    logger.info(f"Ollama'ya chatbot isteği gönderiliyor. conversationId: {conversation_id}")

    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(
                ollama_url,
                json={"model": model_name, "prompt": full_prompt, "stream": False}
            )
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "Üzgünüm, bir sorun oluştu ve bir yanıt üretemedim.").strip()
        except Exception as e:
            logger.error(f"Chatbot LLM isteği sırasında hata oluştu: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Chatbot servisiyle iletişimde bir hata oluştu.")

@app.get("/")
async def root():
    return {"message": "AI Service is running"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/summarize", response_model=SummaryWithRemindersResponse, summary="Bir Konuşmayı Özetler ve Hatırlatıcıları Çıkarır")
async def summarize_endpoint(payload: MeetingPayload):
    conversation_id = str(uuid.uuid4())
    logger.info(f"Yeni özet ve hatırlatıcı isteği alındı. conversationId: {conversation_id}")
    transcript = format_transcript(payload)
    if not transcript:
        raise HTTPException(status_code=400, detail="Geçerli bir konuşma dökümü bulunamadı.")
    
    raw_response = await generate_summary_and_reminders(transcript, conversation_id)
    summary, reminders = parse_llm_response(raw_response)
    
    return SummaryWithRemindersResponse(summary=summary, reminders=reminders, participants=payload.participants)

@app.post("/summarize-chat", response_model=SimpleSummaryResponse, summary="Bir Metin Sohbetinin Basit Özetini Yapar")
async def summarize_chat_endpoint(payload: ChatPayload):
    conversation_id = str(uuid.uuid4())
    logger.info(f"Yeni sohbet geçmişi özet isteği alındı. conversationId: {conversation_id}")
    formatted_chat = format_chat_history(payload)
    if not formatted_chat:
        raise HTTPException(status_code=400, detail="Geçerli bir sohbet geçmişi bulunamadı.")
    simple_summary = await generate_simple_summary_with_ollama(formatted_chat, conversation_id)
    return SimpleSummaryResponse(summary=simple_summary, participants=payload.participants)

@app.post("/chatbot", response_model=ChatbotResponse, summary="VidSync Chatbot ile Etkileşime Girer")
async def chatbot_endpoint(payload: ChatbotQuery):
    conversation_id = str(uuid.uuid4())
    logger.info(f"Yeni chatbot sorgusu alındı: '{payload.query}'. conversationId: {conversation_id}")
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Sorgu boş olamaz.")

    llm_response = await generate_chatbot_response(payload.query, conversation_id)
    return ChatbotResponse(response=llm_response)

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