from fastapi import FastAPI
import requests

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Service is running"}

@app.get("/health")
def health_check():
    try:
        r = requests.get("http://ollama:11434")
        return {"ollama_status": "reachable", "ollama_response": r.status_code}
    except Exception as e:
        return {"ollama_status": "unreachable", "error": str(e)}
