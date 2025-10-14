from pydantic import BaseModel

class Message(BaseModel):
    """Bir konuşmadaki tek bir mesajı temsil eder."""
    message_id: int
    timestamp: str
    sender_id: str
    message: str