from pydantic import BaseModel
from typing import List, Dict, Any
from Models.Participant import Participant
from Models.Message import Message

class ConversationPayload(BaseModel):
    """/summarize uç noktasına gönderilecek isteği temsil eder. Sağlanan JSON ile eşleşir."""
    metadata: Dict[str, Any]
    participants: List[Participant]
    chat_history: List[Message]
    processing_results: Dict[str, Any]