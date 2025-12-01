# MIA - Mood-Intelligence Assistant

**MIA** is an emotion-aware desktop AI assistant designed to provide personalized support by detecting and responding to user emotions in real-time.

## Project Overview

MIA uses a multimodal approach, analyzing:
- **Speech Tone** (Voice Emotion Recognition)
- **Text Sentiment** (NLP)
- **Facial Expressions** (Computer Vision)
- **Gaze & Posture**

Based on the user's emotional state, MIA adapts its:
- **Dialogue:** Empathetic responses, coping strategies.
- **Interface:** Dynamic themes, calming animations, ambient sounds.

## Tech Stack

- **Frontend:** React, TailwindCSS, Vite
- **Backend:** Python (FastAPI)
- **AI/ML:** OpenAI Whisper, MediaPipe, Transformers (Hugging Face)

## Setup Instructions

### Prerequisites
- Node.js & npm
- Python 3.8+

### Backend Setup
1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```

## Repository
[https://github.com/abiruth29/MIA-Mood-Intelligence-Assistant](https://github.com/abiruth29/MIA-Mood-Intelligence-Assistant)
