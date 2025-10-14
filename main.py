import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import httpx
import json
from Models.Participant import Participant
from Models.Message import Message
from Models.ConversationPayload import ConversationPayload
from Models.SummaryResponse import SummaryResponse

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

async def generate_summary_with_ollama(transcript: str) -> str:
    system_prompt = (
        "Sen uzman bir toplantı asistanısın. Lütfen aşağıdaki konuşmanın kısa ve profesyonel bir özetini çıkar. "
        "Konuşulan ana konulara, alınan kararlara ve belirlenen eylem maddelerine odaklan. "
        "Özet en fazla 4 cümle olmalıdır. İşte konuşma dökümü:"
    )

    full_prompt = f"{system_prompt}\n\n---\n\n{transcript}"

    ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434") + "/api/generate"
    model_name = "mistral:7b"

    print(f"Ollama'ya istek gönderiliyor: {ollama_url}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                ollama_url,
                json={
                    "model": model_name,
                    "prompt": full_prompt,
                    "stream": False
                },
                timeout=90.0
            )
            response.raise_for_status()

            response_data = response.json()
            summary_text = response_data.get("response", "").strip()

            if not summary_text:
                raise ValueError("Ollama'dan boş bir özet yanıtı alındı.")
            
            return summary_text

        except httpx.RequestError as e:
            print(f"Ollama servisine bağlanırken ağ hatası oluştu: {e}")
            raise HTTPException(status_code=503, detail="Özetleme servisine ulaşılamıyor. Lütfen Ollama konteynerinin çalıştığından emin olun.")
        except httpx.HTTPStatusError as e:
            print(f"Ollama API'den hata yanıtı alındı: {e.response.text}")
            raise HTTPException(status_code=502, detail=f"Özetleme servisinde bir hata oluştu: {e.response.text}")
        except Exception as e:
            print(f"Özetleme sırasında beklenmedik bir hata oluştu: {e}")
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
    try:
        transcript = format_transcript(payload)
        dynamic_summary = await generate_summary_with_ollama(transcript)
        print(f"Başarılı özet oluşturuldu: {dynamic_summary}")
        return SummaryResponse(summary=dynamic_summary)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Endpoint içinde beklenmedik hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))
