# Development Handover Guide: MIA (Mood Intelligence Assistant)

## Current Status (As of Jan 2026)

**Phase**: Sprint 3-4 (Audio Pipeline & Multimodal Foundation)

We have successfully implemented the core structure of the backend and frontend, specifically focusing on the Audio and Textual Analysis pipeline.

### What is Working:
1.  **Backend (FastAPI)**:
    *   **WebSocket Streaming**: Handles real-time communication for video frames, audio chunks, and analysis results.
    *   **Video Capture**: Captures webcam feed (ready for processing).
    *   **Audio Capture**: Captures microphone input in real-time chunks.
    *   **ASR (Speech-to-Text)**: Uses `Whisper` to transcribe audio with high accuracy.
    *   **Hybrid Emotion Detection**:
        *   **Audio**: Uses `Wav2Vec2` to analyze tone/pitch (40% weight).
        *   **Text**: Uses `DistilRoberta` to analyze sentiment/vocabulary (60% weight).
        *   **Fusion**: A weighted algorithm combines these scores to produce a stable "Hybrid Emotion".

2.  **Frontend (React + Vite)**:
    *   **Live Feed**: Displays the webcam video.
    *   **Live Feedback**: Shows real-time transcription and fluid emotion confidence scores.
    *   **WebSocket**: Establishes a robust connection to the Python backend.

---

## Remaining Development Plan

The following modules need to be built to complete the project as per the original Scrum Plan.

### 1. Vision Processing Pipeline (Sprint 5-6)
*Goal: Give MIA "eyes" to read facial expressions.*

*   **Facial Landmark Detection (MediaPipe)**:
    *   **Task**: Integrate MediaPipe FaceMesh in `backend/app/vision_pipeline.py`.
    *   **Why**: To get 468 facial landmarks. This is lightweight and runs on CPU.
    *   **Output**: Draw these landmarks on the frontend video feed for a cool "tech" look.
*   **Facial Expression Recognition (FER)**:
    *   **Task**: Implement a CNN model (or use a pre-trained one like purely `DeepFace` or a custom `EfficientNet`) to classify the face into 7 emotions.
    *   **Integration**: Add this as the 3rd pillar to our Hybrid Fusion logic (Audio + Text + **Video**).
*   **Gaze & Head Pose**:
    *   **Task**: Use the landmarks to calculate if the user is looking down (sad/tired), away (distracted), or at the screen (engaged).

### 2. Advanced Multimodal Fusion (Sprint 7)
*Goal: smarter combination of signals.*

*   **Tri-modal Fusion**:
    *   Currently, we do `0.4 * Audio + 0.6 * Text`.
    *   **New Formula**: `0.3 * Audio + 0.3 * Text + 0.4 * Video`.
    *   *Note*: Video is often the most reliable for "instant" reactions, while text is best for "deep" sentiment.
*   **Temporal Smoothing**:
    *   The current system has a basic history window. Enhance this to avoid rapid flickering between emotions. Use a "Sliding Window Average" of the last 5 seconds of scores.

### 3. Adaptive UI/UX (Sprint 8)
*Goal: The interface should heal the user.*

*   **Dynamic Themes**:
    *   If **Stress** is high -> Change UI to Calming Blue/Green, slow down animations.
    *   If **Sadness** is high -> Warm Orange/Yellow glow, comforting visuals.
    *   **Implementation**: Use React Context (`ThemeContext`) that listens to the WebSocket emotion stream and updates Tailwind classes.
*   **Visualizations**:
    *   Add a real-time "Emotion Bar Chart" or a "Spider Graph" to show the breakdown of emotions (e.g., 20% Happy, 80% Excited).

### 4. Interactive AI Companion (Sprint 8+)
*Goal: MIA talks back.*

*   **LLM Integration**:
    *   Connect the transcribed text + detected emotion to an LLM (local `Llama 3` via `Ollama` or OpenAI API).
    *   **Prompt Engineering**: "You are a helpful assistant. The user is feeling {current_emotion}. Respond empathetically."
*   **TTS (Text-to-Speech)**:
    *   Give MIA a voice to speak the response back to the user.

---

## Important Technical Notes for the New Developer

1.  **Environment**: All Python dependencies are in `backend/venv`. Always run `backend/venv/Scripts/activate` before working.
2.  **Model Caching**: Models (Whisper, Wav2Vec2, Roberta) are large. They are cached locally in `~/.cache/huggingface`. Ensure the deployment machine has disk space.
3.  **Concurrency**: The backend uses `asyncio`. Be careful not to use blocking code (like `time.sleep()`) inside the async loops. Use `await asyncio.sleep()` instead.
4.  **Hardware**: The current setup runs on CPU but is optimized. If you add the Vision pipeline, watch out for FPS drops. MediaPipe is fast, but deep CNNs for FER might need optimization (like ONNX runtime) if not using a GPU.
