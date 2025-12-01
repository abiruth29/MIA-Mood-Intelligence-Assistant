# Project Proposal: Emotion-Aware Desktop AI Assistant 

# **MIA \-** Mood-Intelligence Assistant

**Motivation.** Modern digital tools lack the emotional sensitivity needed for truly personalized support. Affective computing research shows that systems able to detect and respond to user emotions can **“personalize care, enhance engagement, and improve outcomes”** in mental health contexts[\[1\]](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1657031/full#:~:text=Affective%20computing%2C%20which%20is%20considered,enhance%20engagement%2C%20and%20improve%20outcomes)[\[2\]](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1657031/full#:~:text=Digital%20mental%20health%20interventions%20,The%20findings%20suggest). For example, adaptive systems that sense stress or mood can proactively offer coping strategies, leading to better user satisfaction and well-being[\[2\]](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1657031/full#:~:text=Digital%20mental%20health%20interventions%20,The%20findings%20suggest)[\[3\]](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1657031/full#:~:text=match%20at%20L264%20disorders%20,this%20fusion%20reshapes%20human%E2%80%93technology%20interaction). Our project addresses the gap by building a desktop AI assistant that continuously gauges a user’s affective state (via speech tone, text sentiment, facial expression, and gaze) and adapts its dialogue and interface to help the user feel calm and supported.

**Real-world Problem.** Students and professionals often face stress, burnout, or emotional lows without easy access to support. Traditional chatbots or reminder apps can be effective, but they usually ignore *how* the user is feeling. An emotion-aware assistant on the desktop can detect signs of anxiety, frustration or fatigue in real time and intervene with personalized prompts (like relaxation exercises or journaling questions) and an environment tailored to soothe the user. This proactive, empathetic support could improve concentration, mood regulation, and overall well-being during study or work, addressing the unmet need for sensitive digital companionship[\[2\]](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1657031/full#:~:text=Digital%20mental%20health%20interventions%20,The%20findings%20suggest)[\[3\]](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1657031/full#:~:text=match%20at%20L264%20disorders%20,this%20fusion%20reshapes%20human%E2%80%93technology%20interaction).

## Proposed Solution

We will develop a desktop application (using **React+TailwindCSS**) that runs an AI assistant in real time. Key components are shown conceptually in Figure 1:

* **User Input:** The user interacts naturally: speaking to the assistant via microphone and facing the computer camera. (No VR headset needed.)

* **Multimodal Emotion Analysis:** The system processes audio and video from the webcam/microphone to infer emotions. Specifically:

* **Speech-to-Text (ASR):** Convert spoken language to text using the Whisper model[\[4\]](https://huggingface.co/openai/whisper-large-v3#:~:text=Whisper%20is%20a%20state,shot%20setting). Whisper is a robust open-source ASR transformer from OpenAI, trained on millions of hours of audio[\[4\]](https://huggingface.co/openai/whisper-large-v3#:~:text=Whisper%20is%20a%20state,shot%20setting), which provides highly accurate transcripts even in noisy settings.

* **Text Sentiment Analysis:** Apply a transformer-based NLP model (e.g. BERT or DistilBERT fine-tuned for sentiment/emotion) on the transcribed text to gauge the user’s emotional tone. (Pretrained models like distilbert-base-uncased-emotion can detect sadness, joy, anger, etc.[\[5\]](https://huggingface.co/blog/sentiment-analysis-python#:~:text=sentiment%20analysis%20on%20product%20reviews,love%2C%20anger%2C%20fear%20and%20surprise).)

* **Voice Emotion Recognition:** Extract acoustic features (pitch, energy, MFCC, prosody) from the speech waveform and feed them into a neural classifier to detect emotion from tone. Combining these with text sentiment is known to improve accuracy[\[6\]](https://arxiv.org/pdf/1906.05681#:~:text=Abstract,The%20proposed).

* **Facial Expression Recognition (FER):** Use MediaPipe FaceMesh (or similar) to detect the user’s face and 468 3D landmarks[\[7\]](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#:~:text=The%20MediaPipe%20Face%20Landmarker%20task,transformations%20required%20for%20effects%20rendering). A CNN-based model classifies basic expressions (happy, sad, angry, etc.) from the cropped face. The Face Landmarker also provides *blendshape* coefficients (e.g. eyebrow raise, smile) that quantify expression intensity[\[7\]](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#:~:text=The%20MediaPipe%20Face%20Landmarker%20task,transformations%20required%20for%20effects%20rendering).

* **Gaze and Head Pose Estimation:** Compute head orientation and eye direction from the facial landmarks. Gaze (e.g. looking down, away) and slumped posture can indicate fatigue or disinterest. We will estimate gaze by projecting eye landmarks into 3D or by using simple geometry on the facial mesh.

All these signals are fused into a unified **user state model**. For example, we might assign emotional categories or a valence-arousal score by combining modalities (e.g. if voice is anxious and face shows sadness, classify “stressed”). Late fusion via a small neural network or weighted rules can integrate the modalities. The assistant continuously updates its estimate of the user’s emotional state as the conversation proceeds.

**Adaptive Dialogue and Suggestions.** Based on the inferred emotion, the assistant adjusts its language and suggestions. For instance, if stress or anger is detected, the assistant might shift to a calm tone, say something comforting, and offer a brief breathing exercise or journal prompt. If the user sounds cheerful, it might proceed with regular tasks or offer an uplifting quote. The NLP component (possibly a fine-tuned dialogue model or scripted responses) uses the emotional context to choose empathetic responses. Suggested interventions include guided breathing exercises, self-reflection prompts, mindfulness tips, or simply pausing for a moment of silence. Research shows that even simple prompts and mindful activities can improve emotional regulation[\[8\]](https://positivepsychology.com/journaling-prompts/#:~:text=The%20journaling%20process%20involves%20the,2003%3B%20Ullrich%20%26%20Lutgendorf%2C%202002), so we will encode several such strategies in the assistant.

![][image1]  
*Figure 1: Adaptive assistant concept. The UI (a) monitors webcam and microphone input, (b) analyzes facial expression, speech tone, text sentiment, and gaze in real time, and (c) responds with adaptive dialogue and soothing visual/audio feedback. Such “emotion-powered” interfaces have been shown to engage users and improve mood[\[9\]](https://block.github.io/goose/blog/2025/06/17/goose-emotion-detection-app/#:~:text=In%20the%20end%2C%20Goose%20and,For%20example)[\[10\]](https://arxiv.org/html/2510.00489v1#:~:text=Adaptive%20UIs%20are%20not%20limited,transform%20the%20current%20technological%20landscape).*

**Dynamic UI/UX Adaptation.** In addition to dialogue, the application environment itself will adapt in real time to support the user’s mood. For example, if the user appears anxious or angry, the background theme could shift to gentle colors or nature scenes, a soft melody or white noise could play, and subtle animations (like drifting shapes or guided breathing visuals) might be displayed. Conversely, if the user is bored or down, the UI might introduce bright elements or playful animations to uplift them. Such emotion-aware UI adaptations (color changes, soundscapes, animations) have been explored in prior work and were found to make interfaces more engaging and mood-improving[\[10\]](https://arxiv.org/html/2510.00489v1#:~:text=Adaptive%20UIs%20are%20not%20limited,transform%20the%20current%20technological%20landscape)[\[9\]](https://block.github.io/goose/blog/2025/06/17/goose-emotion-detection-app/#:~:text=In%20the%20end%2C%20Goose%20and,For%20example). Our system will let the user customize or disable these features if desired, but the default behavior will be to subtly “tune” the environment for comfort.

## Key Features

* **Multimodal Emotion Sensing:** Real-time analysis of voice (tone & text) and face (expression, gaze). Combines Automatic Speech Recognition (Whisper) with deep audio emotion models and MediaPipe-based facial analysis.

* **Adaptive Dialogue Engine:** A response generator (rule-based or simple transformer chatbot) that produces empathetic conversation and coping suggestions according to the user’s detected affect.

* **Calming Interventions:** Provision of guided exercises or prompts (e.g. breathing techniques, journaling questions) when stress or sadness is detected. This draws on cognitive-behavioral strategies to help regulate emotion[\[8\]](https://positivepsychology.com/journaling-prompts/#:~:text=The%20journaling%20process%20involves%20the,2003%3B%20Ullrich%20%26%20Lutgendorf%2C%202002).

* **Environment Adaptation:** Dynamic UI adjustments (colors, backgrounds, animations, sound) tied to user mood. For example, a soothing image or pattern might expand when calm, or bright cues when cheerful. This “emotion-powered UI” approach can change background color or add effects (as illustrated in Figure 1) to help users feel better[\[10\]](https://arxiv.org/html/2510.00489v1#:~:text=Adaptive%20UIs%20are%20not%20limited,transform%20the%20current%20technological%20landscape)[\[9\]](https://block.github.io/goose/blog/2025/06/17/goose-emotion-detection-app/#:~:text=In%20the%20end%2C%20Goose%20and,For%20example).

* **Continuous Learning and Logging:** The system logs interactions and emotional states over time (locally) so that we can evaluate patterns. This history can help the assistant recognize persistent issues (e.g. chronic stress) and maybe adapt future behavior.

## Technologies and Tools

We will leverage open-source Python frameworks and libraries:

* **Speech-to-Text (ASR):** [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3) via Hugging Face Transformers. Whisper is “state-of-the-art” and robust to accents, available through the transformers pipeline[\[4\]](https://huggingface.co/openai/whisper-large-v3#:~:text=Whisper%20is%20a%20state,shot%20setting).

* **NLP Models:** Hugging Face Transformers library for sentiment/emotion classification and dialogue generation. For text emotion, we can use a model like distilbert-base-uncased-emotion or fine-tune BERT on an emotion dataset[\[5\]](https://huggingface.co/blog/sentiment-analysis-python#:~:text=sentiment%20analysis%20on%20product%20reviews,love%2C%20anger%2C%20fear%20and%20surprise). The transformers library provides easy pipelines for sentiment analysis[\[5\]](https://huggingface.co/blog/sentiment-analysis-python#:~:text=sentiment%20analysis%20on%20product%20reviews,love%2C%20anger%2C%20fear%20and%20surprise), and also chatbot models (GPT-2/Neo) if needed.

* **Computer Vision:** Google’s MediaPipe FaceMesh (Python) to detect 3D facial landmarks and blendshapes in real time[\[7\]](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#:~:text=The%20MediaPipe%20Face%20Landmarker%20task,transformations%20required%20for%20effects%20rendering). OpenCV for video capture and preprocessing (face ROI extraction, resizing). A lightweight CNN for facial expression classification (using frameworks like TensorFlow/PyTorch). For gaze/head pose, we can use MediaPipe landmarks with a solvePnP algorithm or a prepackaged pose detector.

* **GUI Framework:** The front-end will be developed using **React with Tailwind CSS**, allowing a modern, highly responsive, and visually dynamic user interface. React enables modular component-based development and smooth real-time updates through WebSockets API calls to the Python backend. Tailwind CSS provides utility-first styling, making it easy to design adaptive UI themes (calm, energetic, dark, minimal) that change based on the user’s emotional state. Additional animation libraries such as **Framer Motion / particles.js / Canvas effects** will support visual transitions and ambient elements (e.g., rain, glow, gradient movement) that reinforce emotional feedback. This stack offers significantly better UX flexibility and visual polish than typical Python GUI frameworks and aligns with the project’s goal of delivering an **emotion-aware, immersive interface without requiring VR hardware**.

* **Libraries:** transformers, datasets (HuggingFace), mediapipe, opencv-python, numpy, scipy, torch or tensorflow as needed. Possibly audio libraries like librosa or pyAudio for feature extraction.

## Applied NLP and CV Concepts

Our project integrates concepts from both NLP and computer vision coursework:

* **Speech Emotion Recognition (SER):** Extract audio features (MFCCs, spectrograms, pitch) to classify emotion from the user’s tone. As noted in literature, combining acoustic cues with textual content boosts performance[\[6\]](https://arxiv.org/pdf/1906.05681#:~:text=Abstract,The%20proposed). We will use Whisper for transcription and feed raw audio to a model or service (e.g. a fine-tuned Wav2Vec2 or SpeechBrain model) for emotion tagging.

* **Sentiment/Emotion Analysis:** Apply text classification (e.g. BERT) on transcripts. This captures latent emotion behind words. Transformer-based sentiment models achieve state-of-the-art results on large datasets[\[5\]](https://huggingface.co/blog/sentiment-analysis-python#:~:text=sentiment%20analysis%20on%20product%20reviews,love%2C%20anger%2C%20fear%20and%20surprise), making them ideal to gauge positivity or stress in language.

* **Facial Expression Recognition (FER):** Use convolutional neural networks or ensemble (e.g. variants of VGG or ResNet) to map facial images to emotion classes. This relies on the basics of image processing (face detection, normalization) and supervised learning on labeled face emotion datasets. The MediaPipe FaceMesh task simplifies landmark detection[\[7\]](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#:~:text=The%20MediaPipe%20Face%20Landmarker%20task,transformations%20required%20for%20effects%20rendering), after which a simple classifier (even a small CNN) can detect expressions.

* **Gaze/Head Pose Estimation:** Using 3D landmarks from FaceMesh, we compute head orientation (yaw/pitch/roll) and eye direction. These are classic CV tasks (solvePnP or regression networks). Gaze features give context (e.g. “user looking away” might signal disengagement).

* **Multimodal Fusion:** We will combine audio, text, and visual features. This could be a late-fusion approach (ensemble of separate classifiers) or an integrated model (concatenate embeddings). For instance, we might weight the confidences: if facial expression and tone both indicate stress, we react immediately. Fusion of multimodal signals is well-known to improve robustness in affective systems[\[6\]](https://arxiv.org/pdf/1906.05681#:~:text=Abstract,The%20proposed).

## Innovation and Feasibility

This project is innovative because it brings together recent advances in ASR, transformers, and CV into a cohesive assistive tool. Unlike standard chatbots, our assistant is *emotionally adaptive*: it senses non-verbal cues and changes not just what it says but *how* the environment looks and sounds. Similar emotion-aware UIs have been explored in research (e.g. adaptive recommendation interfaces[\[10\]](https://arxiv.org/html/2510.00489v1#:~:text=Adaptive%20UIs%20are%20not%20limited,transform%20the%20current%20technological%20landscape)), but rarely packaged as a simple desktop assistant. Leveraging powerful pre-trained models (Whisper, BERT, etc.) means we can implement sophisticated features without training huge networks from scratch.

Feasibility for a B.Tech semester project is high because: \- **Reusing Pre-trained Models:** All core components (ASR, sentiment, face landmarks) are available as libraries/models. Integration rather than invention is the focus.  
\- **Scope Control:** We can deliver a proof-of-concept UI with a subset of features (e.g. only basic emotions, a few dialogues, one calming exercise) and iterate. Incremental development (audio pipeline → vision pipeline → fusion → UI) fits a semester timeline.  
\- **Learning Opportunity:** Students will apply class concepts (RNNs/CNNs, transformer models, feature extraction) in practice. Many documentation and tutorials exist (e.g. HuggingFace forums) to smooth development.

## Evaluation Plan

We will evaluate both technical performance and user impact:

* **Technical Metrics:**

* *Emotion Recognition Accuracy:* On benchmark datasets (e.g. RAVDESS for voice, FER2013 for faces), we will measure accuracy/F1 of our models (speech emotion classifier, face emotion classifier, sentiment analyzer).

* *Latency/FPS:* Measure real-time performance – the system should analyze and respond within a fraction of a second. We will profile frame rates (vision pipeline) and transcription speed to ensure responsiveness.

* *Ablation Tests:* Check each modality’s contribution. For example, compare emotion detection accuracy with audio-only vs. audio+vision fusion to demonstrate multimodal gains[\[6\]](https://arxiv.org/pdf/1906.05681#:~:text=Abstract,The%20proposed).

* **User Study:** We will conduct a small user study with peers or volunteers. Participants will interact with the assistant for a set task (e.g. answering a stressful quiz while the assistant monitors them). We will collect:

* *Self-reported Feedback:* Surveys (Likert scale) on user satisfaction, perceived empathy of the assistant, and usefulness of suggestions. Questions like “The assistant’s response made me feel understood” or “I found the calming exercises helpful.”

* *Affective Outcome:* If possible, simple before/after stress measures (e.g. short anxiety questionnaire or heart rate via webcam PPG).

* *System Usability:* Standard SUS questionnaire to evaluate the UI’s usability.

* *Objective Logs:* Does the assistant correctly identify obvious emotions? (We can include controlled prompts to simulate emotions and check detection accuracy in practice.)

We will refine the system based on this feedback and include a discussion of limitations and improvements.

## Deliverables and Outcomes

By the end of the semester, we expect to deliver:

1. **Working Desktop Application:** A packaged app (using React+TailwindCSS) that runs on Windows/Linux/Mac. It will capture webcam/microphone input, display the adaptive interface, and chat with the user.

2. **Source Code and Documentation:** Well-commented code on a public GitHub repository. Documentation will include setup instructions, model sources, and a developer guide.

3. **Technical Report:** A detailed project report describing the architecture, models, data flow, and experimental results (emotion detection accuracies, user study outcomes).

4. **Demonstration Video:** A short video demo showing the assistant in action (e.g., how it changes UI when the user pretends to be stressed vs. calm).

5. **Presentation/Poster:** For showcasing to professors, summarizing our vision, features, and findings.

These outcomes will demonstrate both the *concept* and the *implementation* of an emotion-aware assistant. The code and methods will be reusable for future projects (e.g., extending to multimodal learning or mental health apps), meeting the educational and technical goals of a B.Tech capstone.

**Sources:** Foundational ideas on affective computing and adaptive UIs from recent literature[\[2\]](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1657031/full#:~:text=Digital%20mental%20health%20interventions%20,The%20findings%20suggest)[\[3\]](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1657031/full#:~:text=match%20at%20L264%20disorders%20,this%20fusion%20reshapes%20human%E2%80%93technology%20interaction)[\[10\]](https://arxiv.org/html/2510.00489v1#:~:text=Adaptive%20UIs%20are%20not%20limited,transform%20the%20current%20technological%20landscape), and technical guidance from Hugging Face, Google MediaPipe, and related documentation[\[4\]](https://huggingface.co/openai/whisper-large-v3#:~:text=Whisper%20is%20a%20state,shot%20setting)[\[7\]](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#:~:text=The%20MediaPipe%20Face%20Landmarker%20task,transformations%20required%20for%20effects%20rendering) were used in this proposal. All tools mentioned are open-source and well-supported for Python development (Transformers[\[4\]](https://huggingface.co/openai/whisper-large-v3#:~:text=Whisper%20is%20a%20state,shot%20setting), etc.). The integration of speech and vision for emotional intelligence follows established research on multimodal emotion recognition[\[6\]](https://arxiv.org/pdf/1906.05681#:~:text=Abstract,The%20proposed), ensuring a sound methodological basis for our design.

---

