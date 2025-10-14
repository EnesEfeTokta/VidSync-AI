from pydantic import BaseModel, Field

class SummaryResponse(BaseModel):
    """/summarize uç noktasından dönecek olan yanıt yapısını temsil eder."""
    summary: str = Field(..., description="Konuşmanın özet metni.")