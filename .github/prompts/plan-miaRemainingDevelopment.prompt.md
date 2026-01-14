## Plan: MIA Remaining Development Roadmap

A 4-phase, ~4-week plan to complete MIA's vision pipeline, enhance multimodal fusion, build adaptive UI, and add AI companion features. The foundation (audio pipeline, WebSocket streaming, hybrid emotion fusion) is solid—now we layer on facial analysis and intelligent UI.

---

### Steps

1. **Phase 1 (Days 1-5): Vision Pipeline** — Create `backend/app/vision_pipeline.py` with MediaPipe FaceMesh for 468 landmarks, integrate a FER model (DeepFace or EfficientNet), add gaze/head pose estimation, and update `backend/main.py` to include video emotion in the WebSocket processing loop.

2. **Phase 2 (Days 6-8): Tri-Modal Fusion** — Modify the fusion logic in `backend/main.py` from `0.4*audio + 0.6*text` to `0.3*audio + 0.3*text + 0.4*video`, implement a 5-second sliding window average for temporal smoothing (replacing the current 3-frame mode), and add valence-arousal score computation.

3. **Phase 3 (Days 9-12): Adaptive UI System** — Create `ThemeContext` in React (`frontend/src/`) that subscribes to the emotion WebSocket stream, implement dynamic Tailwind theme switching (calming blues for stress, warm oranges for sadness), add Framer Motion animations, and build an emotion bar chart/spider graph visualization component.

4. **Phase 4 (Days 13-16): AI Companion & Persistence** — Integrate LLM (Ollama with Llama 3 or OpenAI API) for empathetic responses driven by `{current_emotion}` context, add TTS output for MIA's voice, create SQLite database schema for emotion history logging, and build a simple analytics dashboard.

5. **Phase 5 (Days 17-18): Polish & Testing** — Add error handling for device disconnection/reconnection, implement user settings/preferences storage, stress test the full pipeline for FPS drops, and ensure ONNX runtime optimization if GPU isn't available.

---

### Further Considerations

1. **FER Model Choice?** DeepFace (simpler, pre-packaged) vs. custom EfficientNet (more control, better accuracy potential) — *Recommend DeepFace for speed.*

2. **LLM Hosting?** Local Ollama (privacy, no API costs, needs 8GB+ RAM) vs. OpenAI API (faster setup, ongoing cost) — *Depends on deployment constraints.*

3. **Should we add landmark overlay visualization on the video feed?** This gives a "tech look" but adds rendering overhead — *Optional, can be toggled.*
