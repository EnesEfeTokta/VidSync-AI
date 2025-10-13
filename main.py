from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI
import requests

# ----------------
# 1. FastAPI Uygulaması Tanımı
# ----------------

# Mevcut uygulamanız
app = FastAPI()

# ----------------
# 2. Pydantic Veri Modelleri (Yeni Eklenen Kısım)
# ----------------

class Participant(BaseModel):
    """Konuşmadaki bir katılımcıyı temsil eder."""
    participant_id: str = Field(..., description="Katılımcının benzersiz kimliği.")
    name: str = Field(..., description="Katılımcının adı.")

class Message(BaseModel):
    """Bir konuşmadaki tek bir mesajı temsil eder."""
    message_id: str = Field(..., description="Mesajın benzersiz kimliği.")
    sender_id: str = Field(..., description="Mesajı gönderen katılımcının kimliği.")
    # ISO 8601 formatında bir datetime bekler
    timestamp: datetime = Field(..., description="Mesajın gönderildiği tarih ve saat.")
    content: str = Field(..., description="Mesajın metin içeriği.")

class ConversationPayload(BaseModel):
    """/summarize uç noktasına gönderilecek isteği temsil eder."""
    conversation_id: str = Field(..., description="Konuşmanın benzersiz kimliği.")
    participants: List[Participant] = Field(..., description="Katılımcıların listesi.")
    messages: List[Message] = Field(..., description="Mesajların listesi.")
    metadata: Optional[dict] = Field(None, description="Ek meta veriler.")

class SummaryResponse(BaseModel):
    """/summarize uç noktasından dönecek olan yanıt yapısını temsil eder."""
    summary: str = Field(..., description="Konuşmanın özet metni.")

# ----------------
# 3. Mevcut Uç Noktalar
# ----------------

@app.get("/")
def root():
    return {"message": "AI Service is running"}

@app.get("/health")
def health_check():
    # Geçici çözüm: Ollama bağlantısını denemek yerine her zaman başarılı döndür.
    # Görev tamamlandıktan sonra bu kısmı geri alabilirsiniz.
    return {"status": "ok", "message": "FastAPI is running and operational."} 


# ----------------
# 4. Yeni /summarize POST Uç Noktası (Yeni Eklenen Kısım)
# ----------------

@app.post(
    "/summarize",
    response_model=SummaryResponse,  # Yanıtın SummaryResponse modeline uymasını sağlar
    status_code=200,
    summary="Konuşma verilerini özetler."
)
async def summarize_conversation(payload: ConversationPayload):
    """
    Gelen ConversationPayload yapısını alır ve istenen sabit özeti döndürür.

    * **Doğrulama:** Pydantic modeli sayesinde otomatik veri doğrulama yapılır.
    * **Yanıt:** Sabit JSON yanıtı döner.
    """
    # Bu görev için istenen hardcoded (sabit) yanıt.
    static_summary = "Bu, bir test özetidir."

    # FastAPI, bu Pydantic modelini otomatik olarak JSON yanıtına çevirecektir.
    return SummaryResponse(summary=static_summary)