from pydantic import BaseModel

class Participant(BaseModel):
    """Konuşmadaki bir katılımcıyı temsil eder."""
    participant_id: str
    full_name: str
    role: str