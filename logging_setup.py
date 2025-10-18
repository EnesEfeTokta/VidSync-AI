import logging
from config import ayarlar 

def loglamayi_kur():
    """Ayarlara göre root logger'ı yapılandırır."""
    logging.basicConfig(
        
        level=ayarlar.LOG_SEVIYESI.upper(),
        
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


logger = logging.getLogger("ai-service")