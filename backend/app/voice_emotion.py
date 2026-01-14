from transformers import pipeline
import numpy as np
import torch

class VoiceEmotionClassifier:
    def __init__(self, model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        print(f"Loading Voice Emotion model: {model_name}...")
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "audio-classification",
            model=model_name,
            device=device,
        )
        self.emotion_history = []
        self.smoothing_window = 3
        print("Voice Emotion model loaded.")

    def classify(self, audio_data, sample_rate=16000):
        """
        Classify emotion from audio data.
        Args:
            audio_data: np.array of int16 or float32
            sample_rate: int
        Returns:
            dict: {"emotion": str, "confidence": float, "all_scores": list}
        """
        if audio_data is None or (hasattr(audio_data, 'size') and audio_data.size == 0):
            return {"emotion": "neutral", "confidence": 0.0, "all_scores": []}

        # Convert int16 to float32 if needed
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        try:
            # Run classification
            results = self.pipe(audio_data, sampling_rate=sample_rate)
            
            if results:
                top_emotion = results[0]["label"]
                top_confidence = float(results[0]["score"])
                
                # Apply temporal smoothing
                self.emotion_history.append(top_emotion)
                if len(self.emotion_history) > self.smoothing_window:
                    self.emotion_history.pop(0)
                
                # Get most common emotion in window
                smoothed_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
                
                return {
                    "emotion": smoothed_emotion,
                    "confidence": round(top_confidence, 3),
                    "all_scores": [{"label": r["label"], "score": round(float(r["score"]), 3)} for r in results[:5]]
                }
            
            return {"emotion": "neutral", "confidence": 0.0, "all_scores": []}
            
        except Exception as e:
            print(f"Emotion classification error: {e}")
            return {"emotion": "neutral", "confidence": 0.0, "all_scores": []}
