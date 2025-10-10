# 🎥 AI Conversation Summarizer (Python Backend)

This repository provides the **AI backend service** for a WebRTC-based video chat application built with ASP.NET and React (TypeScript).  
It handles the **speech-to-text transcription** and **conversation summarization** processes using Python and NLP techniques.

---

## 🧠 Features

- 🎙️ Real-time **speech-to-text** transcription of video call audio  
- ✍️ **Automatic summarization** of transcribed text  
- 🤖 Powered by **Python**, **OpenAI / Hugging Face** models, and NLP libraries  
- ⚡ Designed to integrate seamlessly with the main ASP.NET + React application  

---

## 🧩 System Architecture

```plaintext
[React (Client)] → [ASP.NET (Server)] → [Python AI Service]
                               ↳ Transcription
                               ↳ Summarization
