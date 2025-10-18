from pydantic_settings import BaseSettings, SettingsConfigDict

class Ayarlar(BaseSettings):
    """
    AI Servisi için konfigürasyon ayarları. 
    Ortam değişkenlerinden 'AI_' önekiyle okur (örneğin AI_OLLAMA_BASE_URL).
    """
    model_config = SettingsConfigDict(env_prefix='AI_')

    LOG_SEVIYESI: str = "INFO"

    OLLAMA_BASE_URL: str = "http://ollama:11434/api" 
    
    OLLAMA_MODEL_ADI: str = "mistral:7b"


ayarlar = Ayarlar()