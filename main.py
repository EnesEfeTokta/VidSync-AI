import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import httpx
import json
import uuid 
import logging
from config import ayarlar 
from logging_setup import loglamayi_kur 

from Models.Participant import Participant
from Models.Message import Message
from Models.ConversationPayload import ConversationPayload
from Models.SummaryResponse import SummaryResponse

loglamayi_kur()
logger = logging.getLogger("ai-service")

app = FastAPI(
    title="AI Meeting Summarization Service",
    description="Toplantı dökümlerini özetlemek için Ollama ve Mistral modelini kullanan bir API.",
    version="1.0.0"
)

def format_transcript(payload: ConversationPayload) -> str:
    transcript_lines = []
    participant_names = {p.participant_id: p.full_name for p in payload.participants}

    for msg in payload.chat_history:
        sender_name = participant_names.get(msg.sender_id, "Bilinmeyen Katılımcı")
        transcript_lines.append(f"[{msg.timestamp}] {sender_name}: {msg.message}")

    return "\n".join(transcript_lines)

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
    
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(
                ollama_url,
                json={
                    "model": model_name,
                    "prompt": full_prompt,
                    "stream": False
                }
            )
            
            response.raise_for_status() 

            response_data = response.json()
            summary_text = response_data.get("response", "").strip()

            if not summary_text:
                logger.error(f"Ollama'dan boş bir özet yanıtı alındı. conversationId: {conversation_id}")
                raise ValueError("Ollama'dan boş bir özet yanıtı alındı.")
            
            logger.info(f"Ozet başarılı bir şekilde oluşturuldu. conversationId: {conversation_id}")
            return summary_text

        except httpx.ConnectError as e:
            logger.error(f"Ollama servisine bağlanırken ağ hatası oluştu (503). conversationId: {conversation_id}", exc_info=True)
            raise HTTPException(status_code=503, detail="Özetleme servisine ulaşılamıyor. Lütfen Ollama konteynerinin çalıştığından emin olun.")
        
        except httpx.HTTPStatusError as e:
            error_details = e.response.text
            logger.error(f"Ollama API'den hata yanıtı alındı (API Status: {e.response.status_code}). Detay: {error_details}. conversationId: {conversation_id}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Özetleme servisinde bir hata oluştu (Ollama API hatası).")
        
        except Exception as e:
            logger.error(f"Özetleme sırasında beklenmedik bir hata oluştu (500). conversationId: {conversation_id}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Özetleme sırasında beklenmedik bir sunucu hatası oluştu.")


@app.get("/")
async def root():
    return {"message": "AI Service is running"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post(
    "/summarize",
    response_model=SummaryResponse,
    summary="Bir Konuşmayı Özetler",
    description="JSON formatında bir konuşma verisi alır ve Mistral 7B modeli kullanarak bir özet oluşturur."
)
async def summarize_conversation_endpoint(payload: ConversationPayload):
    conversation_id = str(uuid.uuid4())
    logger.info(f"Yeni özet isteği alındı. conversationId: {conversation_id}")

    try:
        transcript = format_transcript(payload)
        dynamic_summary = await generate_summary_with_ollama(transcript, conversation_id)
        
        return SummaryResponse(summary=dynamic_summary)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Endpoint içinde beklenmedik hata oluştu. conversationId: {conversation_id}", exc_info=True)
        raise HTTPException(status_code=500, detail="Genel sunucu hatası.")
