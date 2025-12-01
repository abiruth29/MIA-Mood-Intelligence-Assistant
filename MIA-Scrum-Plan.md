# MIA (Mood-Intelligence Assistant)
## Scrum-Based Modularized Implementation Plan

**Project Duration:** 4 Months (16 weeks)  
**Sprint Duration:** 2 weeks  
**Total Sprints:** 8  
**Team Size:** 2-4 developers  
**Tech Stack:** React + TailwindCSS (Frontend) | Python Backend (Flask/FastAPI) | MediaPipe + Transformers (ML)

---

## Project Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MIA DESKTOP APPLICATION                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │   React UI    │  │ WebSocket/API    │  │ Local Cache  │ │
│  │  (TailwindCSS)│─→│    Connection    │←─│   (SQLite)   │ │
│  └────────────────┘  └──────────────────┘  └──────────────┘ │
│         │                      │                     ▲        │
│         └──────────────────────┼─────────────────────┘        │
│                                │                              │
├────────────────────────────────┼──────────────────────────────┤
│                    PYTHON BACKEND (FastAPI)                   │
├────────────────────────────────┼──────────────────────────────┤
│                                ▼                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Audio      │  │   Vision     │  │   NLP/Fusion     │   │
│  │  Pipeline    │  │   Pipeline   │  │   Pipeline       │   │
│  │              │  │              │  │                  │   │
│  │ • Whisper    │  │ • MediaPipe  │  │ • Sentiment      │   │
│  │ • Voice Emot │  │ • FER (CNN)  │  │ • Dialogue Mgmt  │   │
│  │ • MFCC Extr. │  │ • Pose Est.  │  │ • State Fusion   │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│         │                   │                    │             │
│         └───────────────────┴────────────────────┘             │
│                           │                                    │
│  ┌────────────────────────▼──────────────────────┐            │
│  │         Emotion State Manager                 │            │
│  │  (Valence-Arousal Scores, Emotion Labels)    │            │
│  └────────────────────┬─────────────────────────┘            │
│                       │                                        │
│  ┌────────────────────▼──────────────────────┐               │
│  │    Intervention & Response Generator      │               │
│  │  (Breathing Exercises, Journaling,        │               │
│  │   Dialogue Suggestions, UI Themes)        │               │
│  └───────────────────────────────────────────┘               │
│                                                               │
│  ┌─────────────────────────────────────────┐                │
│  │   Logging & Analytics                   │                │
│  │  (Local SQLite DB for User History)     │                │
│  └─────────────────────────────────────────┘                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Modules (Organized by Function)

### Module 1: **Media Capture & Streaming**
- Microphone input capture
- Webcam video capture
- Stream buffering and preprocessing
- Fault tolerance (device disconnection handling)

### Module 2: **Audio Processing Pipeline**
- Real-time audio buffering
- Whisper ASR (speech-to-text)
- Audio feature extraction (MFCC, spectrograms, pitch)
- Voice emotion classifier

### Module 3: **Vision Processing Pipeline**
- Real-time video capture and frame buffering
- MediaPipe FaceMesh (facial landmarks + blendshapes)
- Face bounding box extraction
- CNN-based Facial Expression Recognition (FER)
- Gaze/head pose estimation
- Preprocessing and normalization

### Module 4: **NLP & Sentiment Analysis**
- Text-to-emotion classification (BERT/DistilBERT)
- Sentiment scoring
- Dialogue generation (rule-based or LLM-based)
- Intervention suggestion engine

### Module 5: **Emotion State Fusion**
- Multimodal signal aggregation
- Valence-Arousal score computation
- Emotion label classification
- Temporal smoothing (maintaining emotion state across frames)

### Module 6: **UI Adaptation Engine**
- Dynamic theme generation (color palettes)
- Animation system (Framer Motion)
- Audio cue system (background sounds, breathing audio)
- Layout adaptation based on emotion

### Module 7: **Dialogue & Intervention Manager**
- Response selection based on emotion
- Breathing exercise renderer
- Journaling prompt suggestions
- Mindfulness tips generator

### Module 8: **Data Logging & Analytics**
- SQLite database schema
- Session logging
- Emotion history tracking
- Performance metrics

---

## Sprint Breakdown

### **SPRINT 1-2: Foundation & Setup (Weeks 1-4)**

**Sprint Goal:** Establish development environment, create project scaffolding, and test basic I/O.

#### **Sprint 1 Deliverables:**

1. **Project Setup**
   - [ ] GitHub repository created with `.gitignore`, `README.md`
   - [ ] Python virtual environment (`requirements.txt` with dependencies)
   - [ ] React project initialized with Vite/Create React App
   - [ ] Backend server skeleton (FastAPI with CORS, WebSocket support)
   - [ ] Docker configuration for reproducibility

2. **Task Breakdown:**
   - `SETUP-1`: Initialize Git repo + branching strategy
   - `SETUP-2`: Create Python backend scaffold (FastAPI)
   - `SETUP-3`: Create React frontend scaffold
   - `SETUP-4`: Configure WebSocket communication
   - `SETUP-5`: Write project README with setup instructions

3. **Acceptance Criteria:**
   - Backend runs locally on `http://localhost:8000`
   - Frontend runs locally on `http://localhost:3000`
   - WebSocket connection between frontend and backend establishes
   - No critical dependency conflicts

#### **Sprint 2 Deliverables:**

1. **Media I/O Module (Module 1)**
   - [ ] Microphone input capture (PyAudio/sounddevice)
   - [ ] Webcam video feed capture (OpenCV)
   - [ ] Stream buffering and queue management
   - [ ] Error handling for device disconnection
   - [ ] Frontend visualization of raw video/audio

2. **Task Breakdown:**
   - `IO-1`: Implement microphone capture + buffering
   - `IO-2`: Implement webcam capture + video preprocessing
   - `IO-3`: Add error handling + device fallbacks
   - `IO-4`: Create frontend component for media preview
   - `IO-5`: Write unit tests for I/O modules

3. **Acceptance Criteria:**
   - Microphone and webcam feed displayed in React UI
   - No latency > 100ms
   - Graceful handling of missing camera/mic
   - All unit tests passing

---

### **SPRINT 3-4: Audio Pipeline (Weeks 5-8)**

**Sprint Goal:** Implement ASR, voice emotion recognition, and real-time audio feature extraction.

#### **Sprint 3 Deliverables:**

1. **Whisper ASR Integration (Module 2 - Part A)**
   - [ ] Download and cache Whisper model (base or small variant)
   - [ ] Implement real-time transcription via Transformers pipeline
   - [ ] Audio chunk buffering for ASR (e.g., 2-second chunks)
   - [ ] Confidence scoring
   - [ ] API endpoint for transcription
   - [ ] Frontend UI for displaying transcribed text

2. **Task Breakdown:**
   - `ASR-1`: Download + cache Whisper model
   - `ASR-2`: Implement Whisper inference pipeline
   - `ASR-3`: Create audio chunking strategy
   - `ASR-4`: Build `/api/transcribe` endpoint
   - `ASR-5`: Create React component for transcription display
   - `ASR-6`: Write end-to-end test

3. **Acceptance Criteria:**
   - Whisper transcribes speech within 1-2s of audio capture
   - Accuracy > 80% on clear English speech
   - Confidence scores provided
   - Handles multiple languages (optional, nice-to-have)

#### **Sprint 4 Deliverables:**

1. **Audio Feature Extraction & Voice Emotion (Module 2 - Part B)**
   - [ ] Extract MFCCs, spectrograms, pitch, energy, prosody
   - [ ] Train or use pretrained voice emotion classifier (Wav2Vec2 or custom CNN)
   - [ ] Combine text sentiment with voice emotion
   - [ ] Implement temporal smoothing
   - [ ] API endpoint for emotion scores

2. **Task Breakdown:**
   - `VOICE-1`: Implement MFCC/spectrogram extraction (librosa)
   - `VOICE-2`: Implement pitch + energy extraction
   - `VOICE-3`: Integrate/train voice emotion classifier
   - `VOICE-4`: Build `/api/voice_emotion` endpoint
   - `VOICE-5`: Add temporal smoothing logic
   - `VOICE-6`: Write tests + benchmark on RAVDESS dataset

3. **Acceptance Criteria:**
   - Voice emotion classifier accuracy ≥ 70% on RAVDESS
   - Latency < 500ms per chunk
   - Smooth emotion transitions (no jittering)
   - JSON response with confidence scores

---

### **SPRINT 5-6: Vision Pipeline (Weeks 9-12)**

**Sprint Goal:** Implement facial landmark detection, facial expression recognition, and gaze estimation.

#### **Sprint 5 Deliverables:**

1. **MediaPipe Integration & Facial Detection (Module 3 - Part A)**
   - [ ] Integrate MediaPipe FaceMesh for landmark detection
   - [ ] Extract 3D facial landmarks + blendshapes
   - [ ] Normalize + validate landmark quality
   - [ ] Gaze vector computation from eye landmarks
   - [ ] Head pose estimation (yaw, pitch, roll)
   - [ ] Visualize landmarks on frontend

2. **Task Breakdown:**
   - `FACE-1`: Integrate MediaPipe FaceMesh
   - `FACE-2`: Extract and cache 468 landmarks
   - `FACE-3`: Compute blendshape coefficients
   - `FACE-4`: Implement head pose estimation
   - `FACE-5`: Build `/api/facial_landmarks` endpoint
   - `FACE-6`: Create React visualization component
   - `FACE-7`: Write tests for landmark accuracy

3. **Acceptance Criteria:**
   - Landmarks detected at 20+ FPS
   - Gaze direction vector computed reliably
   - Head pose angles within ±15° accuracy
   - Real-time visualization on React UI

#### **Sprint 6 Deliverables:**

1. **Facial Expression Recognition (Module 3 - Part B)**
   - [ ] Implement or integrate CNN-based FER model
   - [ ] Classify 6-7 basic emotions (happy, sad, angry, neutral, surprised, disgusted, fearful)
   - [ ] Extract face ROI and preprocess
   - [ ] Confidence scoring per emotion
   - [ ] Combine FER with blendshape data
   - [ ] API endpoint for expression scores

2. **Task Breakdown:**
   - `FER-1`: Prepare FER2013 or AffectNet dataset splits
   - `FER-2`: Train/fine-tune CNN (ResNet18 or EfficientNet)
   - `FER-3`: Implement inference pipeline
   - `FER-4`: Build `/api/facial_emotion` endpoint
   - `FER-5`: Add blendshape-based emotion refinement
   - `FER-6`: Write validation tests on benchmark datasets
   - `FER-7`: Optimize for latency

3. **Acceptance Criteria:**
   - FER accuracy ≥ 65% on FER2013 test set
   - Inference latency < 100ms per frame
   - Confidence scores normalized (0-1 range)
   - Smooth emotion predictions across frames

---

### **SPRINT 7: NLP & Fusion (Weeks 13-14)**

**Sprint Goal:** Implement sentiment analysis, dialogue generation, and multimodal fusion.

#### **Sprint 7 Deliverables:**

1. **Text Sentiment & Emotion Analysis (Module 4 - Part A)**
   - [ ] Use pretrained BERT-based sentiment model (distilbert-base-uncased-emotion)
   - [ ] Fine-tune on emotional dataset if needed
   - [ ] Classify emotions: joy, sadness, anger, fear, surprise, neutral
   - [ ] Extract emotion scores + dominance
   - [ ] API endpoint for text emotion

2. **Multimodal Fusion Engine (Module 5)**
   - [ ] Aggregate voice, facial, and text emotion signals
   - [ ] Implement late fusion (weighted ensemble)
   - [ ] Compute final valence-arousal score
   - [ ] Generate emotion label (stressed, calm, happy, etc.)
   - [ ] Maintain emotion history (sliding window)
   - [ ] API endpoint for fused emotion state

3. **Task Breakdown:**
   - `NLP-1`: Load + test distilbert-base-uncased-emotion
   - `NLP-2`: Build `/api/text_sentiment` endpoint
   - `NLP-3`: Design fusion weights (audio: 0.3, video: 0.4, text: 0.3)
   - `NLP-4`: Implement valence-arousal computation
   - `NLP-5`: Build `/api/emotion_state` endpoint
   - `NLP-6`: Create emotion history buffer
   - `NLP-7`: Write fusion validation tests

4. **Acceptance Criteria:**
   - Sentiment analysis latency < 200ms
   - Fusion produces reasonable emotion labels
   - Emotion state consistent across multiple signals
   - Ablation tests show multimodal > unimodal accuracy
   - Tests passing on sample scenarios

---

### **SPRINT 8: UI/UX Adaptation & Interventions (Weeks 15-16)**

**Sprint Goal:** Complete UI adaptation, intervention system, and data logging. Integration & system testing.

#### **Sprint 8 Deliverables:**

1. **Dialogue & Intervention Manager (Module 7)**
   - [ ] Create scripted dialogue templates based on emotion
   - [ ] Implement breathing exercise renderer (with visual/audio cues)
   - [ ] Journaling prompt suggestion engine
   - [ ] Mindfulness tip generator
   - [ ] API endpoints for suggestions

2. **UI Adaptation Engine (Module 6)**
   - [ ] Dynamic color theme switching (calm, energetic, neutral palettes)
   - [ ] Animated background elements (particles, gradients, etc.)
   - [ ] Sound system integration (background ambient sounds)
   - [ ] Responsive layout adjustments
   - [ ] React components for each theme

3. **Data Logging & Analytics (Module 8)**
   - [ ] SQLite schema design (sessions, emotions, interactions)
   - [ ] Session tracking logic
   - [ ] Export emotion history as CSV
   - [ ] Simple dashboard for user stats

4. **Task Breakdown:**
   - `UI-1`: Design emotion → theme color mappings
   - `UI-2`: Create Framer Motion animation library
   - `UI-3`: Implement background adaptive UI
   - `UI-4`: Integrate sound/audio cues
   - `UI-5`: Build breathing exercise component
   - `UI-6`: Create journaling prompt system
   - `UI-7`: Build `/api/suggestions` endpoint
   - `DB-1`: Design SQLite schema
   - `DB-2`: Implement session logging
   - `DB-3`: Create analytics dashboard
   - `DB-4`: Export functionality
   - `INT-1`: Full system integration test
   - `INT-2`: End-to-end latency profiling
   - `INT-3`: Stress testing with continuous I/O
   - `INT-4`: User feedback incorporation

5. **Acceptance Criteria:**
   - UI theme changes within 300ms
   - Breathing exercise guide renders smoothly
   - Journaling prompts contextually relevant
   - All data persisted to local SQLite
   - End-to-end latency < 1 second
   - No crashes on 30-minute continuous session

---

## Detailed Module Specifications

### **Module 1: Media Capture & Streaming**

**Responsibilities:**
- Capture audio and video from system devices
- Stream data in real-time to processing pipelines
- Handle device lifecycle (connect/disconnect)

**Key Files:**
```
backend/
├── modules/
│   └── media_capture.py
│       ├── class AudioCapture(device_id=None, sample_rate=16000, chunk_size=4096)
│       ├── class VideoCapture(device_id=0, fps=30, resolution=(640, 480))
│       └── class MediaStreamBuffer(max_size=100)
└── tests/
    └── test_media_capture.py
```

**API Endpoints:**
```
POST /api/stream/start        # Start audio/video capture
POST /api/stream/stop         # Stop capture
GET /api/stream/status        # Get current stream status
GET /api/stream/devices       # List available cameras/mics
```

**Data Flow:**
```
Device → Capture Buffer → Processing Queue → Pipelines (Audio/Vision)
```

**Error Handling:**
- Camera/mic not found → fallback to default (id=0)
- Permission denied → raise UserWarning, pause stream
- Device disconnected → auto-reconnect with 3-second retry logic

---

### **Module 2: Audio Processing Pipeline**

**Sub-modules:**
1. **ASR (Automatic Speech Recognition)** → Whisper
2. **Voice Emotion** → Acoustic feature extraction + classifier
3. **Audio Buffering** → Chunking strategy

**Key Files:**
```
backend/
├── modules/
│   └── audio_pipeline.py
│       ├── class WhisperTranscriber(model_size="base")
│       ├── class VoiceEmotionClassifier(model_path)
│       ├── class AudioFeatureExtractor()
│       └── class AudioBuffer(chunk_duration=2.0, sr=16000)
└── tests/
    └── test_audio_pipeline.py
```

**API Endpoints:**
```
POST /api/audio/transcribe         # Transcribe audio chunk
POST /api/audio/voice_emotion      # Extract voice emotion
POST /api/audio/features           # Extract MFCC, pitch, energy
```

**Implementation Details:**

**Whisper ASR:**
```python
def transcribe_audio(audio_chunk):
    """
    Args:
        audio_chunk: np.array (sr=16000, mono, float32)
    Returns:
        {
            "text": str,
            "confidence": float,
            "language": str,
            "timing": float (seconds)
        }
    """
```

**Voice Emotion:**
```python
def classify_voice_emotion(audio_chunk):
    """
    Extract MFCC, spectrogram, pitch, energy
    Feed to pretrained classifier
    Args:
        audio_chunk: np.array
    Returns:
        {
            "emotion": str,
            "scores": {emotion: confidence},
            "arousal": float (0-1),
            "valence": float (0-1)
        }
    """
```

---

### **Module 3: Vision Processing Pipeline**

**Sub-modules:**
1. **Facial Landmark Detection** → MediaPipe FaceMesh
2. **Facial Expression Recognition** → CNN classifier
3. **Gaze & Pose Estimation** → Geometric computation

**Key Files:**
```
backend/
├── modules/
│   └── vision_pipeline.py
│       ├── class FaceMeshDetector()
│       ├── class FacialExpressionClassifier(model_path)
│       ├── class GazeEstimator()
│       ├── class HeadPoseEstimator()
│       └── class FaceROIExtractor()
└── tests/
    └── test_vision_pipeline.py
```

**API Endpoints:**
```
POST /api/vision/landmarks         # Get facial landmarks + blendshapes
POST /api/vision/expression        # Classify facial expression
POST /api/vision/gaze              # Compute gaze direction
POST /api/vision/pose              # Estimate head pose (yaw/pitch/roll)
```

**Implementation Details:**

**MediaPipe Integration:**
```python
def detect_facial_landmarks(frame):
    """
    Args:
        frame: np.array (H, W, 3) RGB
    Returns:
        {
            "landmarks": np.array (468, 3),  # 3D coordinates
            "blendshapes": dict {shape_name: weight},
            "confidence": float,
            "detection_time": float (ms)
        }
    """
```

**FER Classification:**
```python
def classify_expression(face_roi):
    """
    Args:
        face_roi: np.array (224, 224, 3) preprocessed face
    Returns:
        {
            "emotion": str,
            "scores": {emotion: confidence},
            "intensity": float (0-1)
        }
    """
```

---

### **Module 4: NLP & Sentiment Analysis**

**Sub-modules:**
1. **Text Sentiment** → BERT-based classifier
2. **Dialogue Response Selection** → Rule-based or LLM
3. **Intervention Suggestion** → Template engine

**Key Files:**
```
backend/
├── modules/
│   └── nlp_pipeline.py
│       ├── class SentimentAnalyzer(model_name)
│       ├── class DialogueManager()
│       ├── class InterventionEngine()
│       └── class PromptTemplateEngine()
└── tests/
    └── test_nlp_pipeline.py
```

**API Endpoints:**
```
POST /api/nlp/sentiment            # Analyze text sentiment
POST /api/nlp/dialogue             # Generate response
POST /api/nlp/intervention         # Get intervention suggestion
```

**Implementation Details:**

**Sentiment Analysis:**
```python
def analyze_sentiment(text):
    """
    Args:
        text: str
    Returns:
        {
            "emotion": str,
            "scores": {emotion: confidence},
            "valence": float (negative → positive, -1 to 1),
            "arousal": float (calm → excited, -1 to 1)
        }
    """
```

**Dialogue Response:**
```python
def generate_response(emotion_state, user_text):
    """
    Args:
        emotion_state: {label, valence, arousal}
        user_text: str (recent user input)
    Returns:
        {
            "response": str,
            "tone": str (calm, empathetic, uplifting),
            "confidence": float
        }
    """
```

---

### **Module 5: Emotion State Fusion**

**Responsibilities:**
- Aggregate signals from audio, video, and text
- Compute unified emotion representation
- Maintain temporal emotion history

**Key Files:**
```
backend/
├── modules/
│   └── emotion_fusion.py
│       ├── class MultimodalFusion(weights={...})
│       ├── class EmotionStateBuffer(window_size=30)
│       └── class ValenceArousalCompute()
└── tests/
    └── test_emotion_fusion.py
```

**API Endpoints:**
```
POST /api/fusion/combine           # Fuse multimodal signals
GET /api/fusion/state              # Get current emotion state
GET /api/fusion/history            # Get emotion history
```

**Implementation Details:**

**Fusion Logic:**
```python
def fuse_emotions(audio_emotion, visual_emotion, text_emotion):
    """
    Weighted ensemble:
        fused_score = 0.3 * audio + 0.4 * visual + 0.3 * text
    
    Args:
        audio_emotion: {emotion, valence, arousal, confidence}
        visual_emotion: {emotion, intensity, confidence}
        text_emotion: {emotion, valence, arousal, confidence}
    
    Returns:
        {
            "primary_emotion": str,
            "valence": float (0-1),
            "arousal": float (0-1),
            "confidence": float,
            "modality_contributions": {audio: %, visual: %, text: %}
        }
    """
```

**Emotion State Enum:**
```
Calm: valence > 0.5, arousal < 0.4
Happy: valence > 0.6, arousal > 0.4
Stressed: valence < 0.4, arousal > 0.6
Sad: valence < 0.4, arousal < 0.4
Neutral: 0.4 < valence < 0.6, 0.3 < arousal < 0.7
```

---

### **Module 6: UI Adaptation Engine**

**Responsibilities:**
- Dynamically adjust theme colors, animations, sounds
- Render adaptive UI components
- Manage animation lifecycle

**Key Files:**
```
frontend/
├── components/
│   ├── AdaptiveTheme.jsx
│   ├── BackgroundAnimation.jsx
│   ├── SoundController.jsx
│   └── ResponsiveLayout.jsx
├── styles/
│   ├── themes.css  (calm, energetic, neutral palettes)
│   └── animations.css  (Framer Motion definitions)
└── hooks/
    └── useEmotionTheme.js

backend/
├── modules/
│   └── theme_engine.py
│       ├── class ThemeGenerator()
│       ├── class ThemeColorPalette()
│       └── class AnimationConfig()
```

**API Endpoints:**
```
POST /api/ui/theme                 # Get theme config for emotion
POST /api/ui/animations            # Get animation set for emotion
```

**Theme Mapping:**

| Emotion | Background | Primary Color | Animation | Sound |
|---------|------------|---------------|-----------|-------|
| Calm | Soft blue/green gradient | Teal | Slow float | Gentle rain, birds |
| Stressed | Warm colors (orange/red) | Red accent | Breathing pulse | Calm breathing guide |
| Happy | Bright yellows/oranges | Gold | Playful bounce | Uplifting music |
| Sad | Muted blues/grays | Silver | Slow drift | Soft piano |

---

### **Module 7: Dialogue & Intervention Manager**

**Responsibilities:**
- Provide context-aware conversational responses
- Suggest and render interventions (breathing, journaling, etc.)
- Maintain dialogue history

**Key Files:**
```
frontend/
├── components/
│   ├── ChatInterface.jsx
│   ├── BreathingExercise.jsx
│   ├── JournalingPrompt.jsx
│   └── MindfulnessCard.jsx
├── data/
│   ├── dialogueTemplates.json
│   └── interventionLibrary.json
└── hooks/
    └── useIntervention.js

backend/
├── modules/
│   └── intervention_manager.py
│       ├── class DialogueTemplateEngine()
│       ├── class BreathingExerciseGenerator()
│       ├── class JournalingPromptSelector()
│       └── class InterventionScheduler()
```

**API Endpoints:**
```
POST /api/interventions/dialogue   # Get dialogue response
POST /api/interventions/breathing  # Get breathing exercise
POST /api/interventions/journal    # Get journaling prompt
GET /api/interventions/suggestions # Get all available interventions
```

**Intervention Types:**

**1. Breathing Exercises:**
```json
{
    "type": "breathing",
    "name": "4-7-8 Breathing",
    "steps": [
        {"action": "inhale", "duration": 4, "visual": "expand circle"},
        {"action": "hold", "duration": 7, "visual": "hold circle"},
        {"action": "exhale", "duration": 8, "visual": "contract circle"}
    ],
    "cycles": 3,
    "duration_seconds": 147
}
```

**2. Journaling Prompts:**
```json
{
    "emotion": "stressed",
    "prompt": "What's one small thing you can let go of right now?",
    "guidance": "Take 2 minutes to write freely..."
}
```

---

### **Module 8: Data Logging & Analytics**

**Responsibilities:**
- Persist user sessions and emotion data
- Generate analytics and trends
- Export data for evaluation

**Key Files:**
```
backend/
├── modules/
│   └── data_logger.py
│       ├── class SessionLogger()
│       ├── class EmotionDataStore()
│       └── class AnalyticsCompute()
├── db/
│   └── schema.sql
└── tests/
    └── test_data_logger.py

frontend/
├── components/
│   └── AnalyticsDashboard.jsx
└── pages/
    └── History.jsx
```

**Database Schema:**

```sql
-- Sessions
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Emotion Logs (sampled every 500ms)
CREATE TABLE emotion_logs (
    log_id TEXT PRIMARY KEY,
    session_id TEXT,
    timestamp TIMESTAMP,
    primary_emotion TEXT,
    valence REAL,
    arousal REAL,
    confidence REAL,
    audio_emotion TEXT,
    visual_emotion TEXT,
    text_emotion TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Interactions
CREATE TABLE interactions (
    interaction_id TEXT PRIMARY KEY,
    session_id TEXT,
    timestamp TIMESTAMP,
    user_input TEXT,
    assistant_response TEXT,
    intervention_type TEXT,
    user_feedback INT (-1 to 1),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Analytics Summary
CREATE TABLE session_analytics (
    analytics_id TEXT PRIMARY KEY,
    session_id TEXT,
    avg_valence REAL,
    avg_arousal REAL,
    dominant_emotion TEXT,
    intervention_count INT,
    user_engagement_score REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
```

**API Endpoints:**
```
GET /api/analytics/dashboard       # Get user dashboard stats
GET /api/analytics/export?format=csv  # Export session data
POST /api/analytics/session-summary # Get session analytics
```

---

## Development Best Practices

### **Git Workflow**
```
main (production)
  ↑
  └── develop (integration)
       ↑
       ├── feature/audio-pipeline (create from develop)
       ├── feature/vision-pipeline
       ├── feature/ui-adaptation
       └── bugfix/[issue-number]

Commit Message Format:
[MODULE] Description (Ticket #XXX)
e.g., [AUDIO] Implement Whisper ASR integration (#15)
```

### **Testing Strategy**

**Unit Tests:** 
- Test each module in isolation
- Mock external dependencies (ML models, hardware)
- Target: >80% code coverage

**Integration Tests:**
- Test module interactions
- Test API endpoints
- Test frontend ↔ backend communication

**System Tests:**
- End-to-end workflow testing
- Latency profiling
- Stress testing (continuous 30-minute session)

**Acceptance Tests:**
- Verify against sprint acceptance criteria
- User feedback validation

### **Code Quality**
- Linting: `pylint`, `flake8` (Python); `ESLint` (JavaScript)
- Type Hints: Use `typing` module in Python
- Code Review: Peer review before merging to `develop`
- Documentation: Docstrings for all functions

### **Performance Profiling**
```python
import time
import cProfile

# Profile audio pipeline
profiler = cProfile.Profile()
profiler.enable()
# ... run code
profiler.disable()
profiler.print_stats(sort='cumulative')

# Latency tracking
start_time = time.perf_counter()
result = process_audio(chunk)
latency = (time.perf_counter() - start_time) * 1000  # ms
print(f"Audio processing latency: {latency:.2f} ms")
```

---

## Dependency Management

### **Python Dependencies**
```
# Core ML/AI
torch==2.0.0 or tensorflow==2.13.0
transformers==4.30.0
mediapipe==0.10.0

# Audio
librosa==0.10.0
sounddevice==0.4.5
scipy==1.11.0

# Computer Vision
opencv-python==4.8.0
numpy==1.24.0
Pillow==10.0.0

# Web Framework
fastapi==0.103.0
uvicorn==0.23.0
python-multipart==0.0.6
pydantic==2.0.0

# Database
sqlalchemy==2.0.0
sqlite3 (built-in)

# Testing
pytest==7.4.0
pytest-cov==4.1.0
pytest-asyncio==0.21.0

# Dev Tools
black==23.9.0
pylint==2.17.0
flake8==6.0.0
```

### **Frontend Dependencies**
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.5.0",
    "tailwindcss": "^3.3.0"
  },
  "devDependencies": {
    "vite": "^4.5.0",
    "@vitejs/plugin-react": "^4.0.0",
    "eslint": "^8.50.0",
    "prettier": "^3.0.0"
  }
}
```

---

## Risk Management

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| GPU/Hardware unavailable | Medium | High | Implement CPU fallback, optimize models for inference |
| Model accuracy lower than expected | Medium | High | Use ensemble methods, fine-tune on domain data |
| Real-time latency > 1s | Medium | High | Profile early, optimize bottlenecks, use quantization |
| Privacy concerns (camera/audio) | Low | High | Local processing only, clear user consent, no data upload |
| Scope creep | High | Medium | Strict sprint planning, defer non-essential features to v2 |
| Team member unavailability | Low | Medium | Code documentation, knowledge sharing, pair programming |

---

## Quality Gates (DoD - Definition of Done)

Each task must satisfy ALL criteria before marking complete:

- [ ] Code written + locally tested
- [ ] Unit tests written (target: >80% coverage)
- [ ] Code review approved by 1 peer
- [ ] Documentation updated (docstrings, README, architecture doc)
- [ ] Linting checks passing (no errors)
- [ ] Integration tests passing
- [ ] Performance benchmarks met (latency, memory)
- [ ] No blocking issues created
- [ ] Committed to feature branch + PR created

---

## Delivery Artifacts per Sprint

### **Post-Sprint 2:**
- ✅ Project scaffolding complete
- ✅ Backend + Frontend running locally
- ✅ Media capture proof-of-concept

### **Post-Sprint 4:**
- ✅ Audio pipeline complete + tested
- ✅ Whisper ASR working
- ✅ Voice emotion baseline

### **Post-Sprint 6:**
- ✅ Vision pipeline complete + tested
- ✅ MediaPipe FaceMesh + FER working
- ✅ Video+Audio integration

### **Post-Sprint 8 (Final):**
- ✅ **Working Desktop Application** (packaged .exe/.dmg/.deb)
- ✅ **Source Code on GitHub** (public repo)
- ✅ **Technical Report** (architecture, results, evaluation)
- ✅ **Demo Video** (5-10 minutes showing system in action)
- ✅ **User Study Results** (survey responses, emotion detection accuracy)
- ✅ **API Documentation** (Swagger/OpenAPI)
- ✅ **Deployment Guide** (setup, running instructions)

---

## Success Metrics (KPIs)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Audio Latency** | < 500ms | Per-chunk processing time |
| **Vision FPS** | ≥ 20 FPS | Real-time frame processing |
| **ASR Accuracy** | ≥ 80% | Word error rate on test set |
| **Emotion Accuracy** | ≥ 65% (FER), ≥ 70% (Voice) | Benchmark dataset F1 score |
| **UI Responsiveness** | < 300ms | Theme change latency |
| **Session Duration** | ≥ 30 min stable | Without crashes |
| **User Satisfaction** | ≥ 4/5 (SUS score) | Likert scale survey |
| **Code Coverage** | ≥ 80% | pytest coverage report |

---

## Communication & Standup Format

**Daily Standup (15 min):**
```
Each person answers:
1. What did I complete yesterday?
2. What am I working on today?
3. Any blockers or help needed?

Format: Text in #standup Slack channel or verbal sync
```

**Sprint Planning (2 hours):**
- Review sprint goals
- Break down user stories into tasks
- Estimate story points (1, 2, 3, 5, 8)
- Assign tasks + identify dependencies

**Sprint Review (1 hour):**
- Demo completed features to stakeholders
- Gather feedback
- Update product backlog

**Sprint Retrospective (1 hour):**
- What went well?
- What could improve?
- Action items for next sprint

---

## Timeline Summary

```
Week  1-4   → Sprint 1-2   (Foundation & Media I/O)
Week  5-8   → Sprint 3-4   (Audio Pipeline)
Week  9-12  → Sprint 5-6   (Vision Pipeline)
Week  13-14 → Sprint 7     (NLP & Fusion)
Week  15-16 → Sprint 8     (UI/UX & Integration)

Milestones:
- End of Week 4: Media capture MVP ✓
- End of Week 8: Complete audio pipeline ✓
- End of Week 12: Complete vision pipeline ✓
- End of Week 14: Multimodal fusion working ✓
- End of Week 16: Full system ready for evaluation & demo ✓
```

---

## Next Steps (Immediate Actions)

### **Week 1 Tasks:**
1. [ ] Create GitHub repository
2. [ ] Set up development environment (Python venv, Node.js)
3. [ ] Create Jira/GitHub Projects board for sprint tracking
4. [ ] Install & verify dependencies
5. [ ] Finalize sprint 1 task breakdown
6. [ ] Create communication channels (Slack, daily standup schedule)
7. [ ] Draft API specification (OpenAPI/Swagger)
8. [ ] Schedule weekly reviews

---

## References & Resources

- **Transformers/NLP:** https://huggingface.co/docs/transformers
- **MediaPipe:** https://developers.google.com/mediapipe
- **OpenAI Whisper:** https://github.com/openai/whisper
- **FastAPI:** https://fastapi.tiangolo.com/
- **React:** https://react.dev/
- **TailwindCSS:** https://tailwindcss.com/
- **Framer Motion:** https://www.framer.com/motion/

---

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Project Lead:** [Abiruth S]  
**Status:** Ready for Sprint 1 Kickoff ✅
