# 🎥 VidSync-AI: AI-Powered Meeting Intelligence Service

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive AI service for **real-time meeting transcription**, **intelligent summarization**, and **interactive chatbot assistance**. Built with FastAPI, OpenAI Whisper, and Ollama LLM, VidSync-AI transforms your video meetings into actionable insights.

---

## 🌟 Features

### 🎙️ Real-Time Speech-to-Text Transcription
- **WebSocket-based streaming** for low-latency audio processing
- Powered by **OpenAI Whisper** for accurate multilingual transcription
- Configurable buffer sizes for optimal performance
- Support for continuous audio stream processing

### 🤖 AI-Powered Meeting Summarization
- Automatic generation of **concise meeting summaries**
- **Smart extraction** of action items, decisions, and key points
- **Intelligent reminder detection** with dates and times
- Participant-aware summarization with role-based context

### 💬 Interactive VidSync Chatbot
- Real-time Q&A about VidSync platform features
- Context-aware responses powered by **Ollama LLM**
- Friendly and professional conversational tone
- Instant answers to user queries

### 🔒 Enterprise-Ready Architecture
- **Docker containerization** for easy deployment
- **NVIDIA GPU support** for accelerated AI processing
- **Cloudflare Tunnel** integration for secure remote access
- Comprehensive logging and error handling
- RESTful API design with OpenAPI documentation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VidSync-AI Service                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   FastAPI    │  │   Whisper    │  │    Ollama    │      │
│  │   Server     │  │     STT      │  │     LLM      │      │
│  │  (Port 8000) │  │   Service    │  │ (Port 11434) │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            │
                   ┌────────┴────────┐
                   │  Cloudflare     │
                   │    Tunnel       │
                   └─────────────────┘
```

### Component Breakdown

- **FastAPI Application** (`main.py`): Core REST API and WebSocket server
- **STT Service** (`stt_service.py`): Real-time speech-to-text processing with Whisper
- **Ollama LLM**: Local language model for summarization and chatbot responses
- **Cloudflare Tunnel**: Secure external access without port forwarding
- **Docker Compose**: Orchestrates all services with GPU support

---

## 📋 Prerequisites

- **Docker** & **Docker Compose** (v3.9+)
- **NVIDIA GPU** with Docker runtime (optional, but recommended for performance)
- **Cloudflare Account** (for tunnel configuration)
- **Python 3.11+** (for local development)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/EnesEfeTokta/VidSync-AI.git
cd VidSync-AI
```

### 2. Configure Environment Variables

Edit the `.env` file with your Cloudflare tunnel token:

```env
CLOUDFLARE_TUNNEL_TOKEN=your_actual_tunnel_token_here
```

**Getting your Cloudflare Tunnel Token:**
1. Visit [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/)
2. Navigate to **Access** > **Tunnels**
3. Create a new tunnel or select an existing one
4. Copy the tunnel token

### 3. Start the Services

```bash
# Pull and start all containers
docker-compose up -d

# View logs
docker-compose logs -f
```

### 4. Download the AI Model

First-time setup requires downloading the Ollama model:

```bash
docker exec -it ollama ollama pull mistral:7b
```

### 5. Verify Installation

Open your browser and navigate to:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 📡 API Endpoints

### Core Endpoints

#### `GET /` - Service Status
```json
{
  "message": "AI Service is running"
}
```

#### `GET /health` - Health Check
```json
{
  "status": "ok"
}
```

#### `POST /summarize` - Generate Meeting Summary

**Request Body:**
```json
{
  "participants": [
    {
      "participant_id": "user-123",
      "full_name": "John Doe",
      "email": "john@example.com",
      "role": "Product Manager",
      "is_moderator": true
    }
  ],
  "transcript": [
    {
      "start_time": 0.0,
      "end_time": 5.2,
      "speaker_id": "user-123",
      "text": "Let's discuss the Q4 roadmap"
    }
  ],
  "metadata": {
    "meeting_id": "meeting-456",
    "duration": 3600
  }
}
```

**Response:**
```json
{
  "summary": "The team discussed Q4 roadmap priorities...",
  "reminders": [
    {
      "event": "Q4 Planning Meeting",
      "date": "2025-11-15",
      "time": "14:00"
    }
  ],
  "participants": [...]
}
```

#### `POST /chatbot` - Ask VidSync Assistant

**Request Body:**
```json
{
  "query": "What features does VidSync offer?"
}
```

**Response:**
```json
{
  "response": "VidSync is a comprehensive video conferencing platform..."
}
```

#### `WebSocket /transcribe` - Real-Time Transcription

**Connection:** `ws://localhost:8000/transcribe`

**Send:** Binary audio data (16kHz, 16-bit PCM)

**Receive:**
```json
{
  "text": "transcribed text from audio"
}
```

---

## ⚙️ Configuration

### Environment Variables

All configuration is managed through environment variables with the `AI_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_OLLAMA_BASE_URL` | `http://ollama:11434/api` | Ollama API endpoint |
| `AI_OLLAMA_MODEL_ADI` | `mistral:7b` | LLM model name |
| `AI_LOG_SEVIYESI` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `CLOUDFLARE_TUNNEL_TOKEN` | *(required)* | Cloudflare tunnel authentication token |

### Customizing the AI Model

To use a different Ollama model:

1. Update `docker-compose.yml`:
```yaml
environment:
  - AI_OLLAMA_MODEL_ADI=llama2:13b
```

2. Download the model:
```bash
docker exec -it ollama ollama pull llama2:13b
```

3. Restart the service:
```bash
docker-compose restart ai
```

### Whisper Model Selection

Edit `main.py` to change the Whisper model size:

```python
stt_service = StreamingSTTService(model_name="base")  # tiny, base, small, medium, large
```

**Model Trade-offs:**
- `tiny`: Fastest, least accurate (~1GB RAM)
- `base`: Balanced (~1GB RAM)
- `small`: Good accuracy (~2GB RAM)
- `medium`: High accuracy (~5GB RAM)
- `large`: Best accuracy (~10GB RAM)

---

## 🛠️ Development

### Local Setup (Without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AI_OLLAMA_BASE_URL=http://localhost:11434/api
export AI_OLLAMA_MODEL_ADI=mistral:7b

# Run the development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Project Structure

```
VidSync-AI/
├── main.py                 # FastAPI application & endpoints
├── config.py              # Configuration management
├── stt_service.py         # Whisper STT service
├── logging_setup.py       # Logging configuration
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Multi-container orchestration
├── .env                  # Environment variables
└── Models/               # Pydantic data models
    ├── ConversationPayload.py
    ├── Message.py
    ├── Participant.py
    └── SummaryResponse.py
```

### Testing the API

Use the interactive API documentation at http://localhost:8000/docs to test endpoints:

```bash
# Or use curl
curl -X POST "http://localhost:8000/chatbot" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is VidSync?"}'
```

---

## 🐳 Docker Deployment

### Building Custom Image

```bash
docker build -t vidsync-ai:custom .
```

### Production Deployment

For production, consider:

1. **Use production WSGI server** (already configured with uvicorn)
2. **Enable HTTPS** via Cloudflare or reverse proxy
3. **Add authentication** for API endpoints
4. **Configure resource limits** in docker-compose.yml:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

### Monitoring Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ai

# Ollama model logs
docker-compose logs -f ollama
```

---

## 🔧 Troubleshooting

### Common Issues

#### **Ollama Connection Timeout**
```bash
# Check if Ollama is running
docker ps | grep ollama

# Test Ollama directly
curl http://localhost:11434/api/generate -d '{"model":"mistral:7b","prompt":"test"}'
```

#### **GPU Not Detected**
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# If error, install nvidia-container-toolkit
```

#### **Whisper Model Loading Errors**
- Ensure sufficient disk space (models are 1-10GB)
- Check Docker container has enough memory allocated
- Try a smaller model (`tiny` or `base`)

#### **WebSocket Connection Failed**
- Verify audio format: 16kHz, 16-bit PCM, mono
- Check CORS settings if connecting from web client
- Ensure WebSocket protocol (ws://) not HTTP

### Performance Optimization

1. **Use GPU acceleration** for Whisper (requires CUDA-compatible GPU)
2. **Increase buffer size** for better transcription quality
3. **Use smaller LLM models** for faster responses
4. **Enable caching** for Ollama responses

---

## 📊 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Transcription Latency** | < 500ms | With GPU acceleration |
| **Summary Generation** | 2-5s | Depends on meeting length |
| **Chatbot Response Time** | 1-3s | With mistral:7b model |
| **Concurrent WebSocket** | 50+ | Limited by system resources |

---

## 🔐 Security Considerations

- **API Authentication**: Implement JWT or API keys for production
- **Rate Limiting**: Add rate limiting to prevent abuse
- **Input Validation**: All inputs are validated using Pydantic
- **Cloudflare Tunnel**: Provides encrypted connection without exposing ports
- **Environment Variables**: Keep `.env` file out of version control

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add type hints to all functions
- Write descriptive commit messages
- Update documentation for new features
- Add tests for new functionality

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Enes Efe Tokta**
- GitHub: [@EnesEfeTokta](https://github.com/EnesEfeTokta)

---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework for building APIs
- [OpenAI Whisper](https://github.com/openai/whisper) - Robust speech recognition
- [Ollama](https://ollama.ai/) - Local LLM deployment
- [Cloudflare](https://www.cloudflare.com/) - Secure tunnel infrastructure

---

## 📞 Support

For issues, questions, or contributions:
- **Issues**: [GitHub Issues](https://github.com/EnesEfeTokta/VidSync-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/EnesEfeTokta/VidSync-AI/discussions)

---

<div align="center">

Made with ❤️ for the VidSync Platform

**[Documentation](https://github.com/EnesEfeTokta/VidSync-AI)** • **[Report Bug](https://github.com/EnesEfeTokta/VidSync-AI/issues)** • **[Request Feature](https://github.com/EnesEfeTokta/VidSync-AI/issues)**

</div>
