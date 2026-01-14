from transformers import pipeline
import numpy as np
import torch
import re

class WhisperTranscriber:
    def __init__(self, model_name="openai/whisper-base.en"):
        print(f"Loading Whisper model: {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device=device,
        )
        print(f"Whisper model loaded on {device}")

    def transcribe(self, audio_data, sample_rate=16000):
        """
        Transcribes audio data.
        Args:
            audio_data: np.array of float32
            sample_rate: int
        Returns:
            dict: {"text": str}
        """
        if audio_data is None or (hasattr(audio_data, 'size') and audio_data.size == 0):
            return {"text": ""}

        # Transformers pipeline expects raw audio
        # Ensure it's float32 and normalized if coming from int16
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        try:
            result = self.pipe(audio_data)
            text = result.get("text", "")
            
            # Filter out noise/silence artifacts (dots, repeated chars, etc.)
            # Remove strings that are just dots, spaces, or very short
            cleaned = re.sub(r'[.\s]+', ' ', text).strip()
            if len(cleaned) < 2 or cleaned.count('.') > len(cleaned) * 0.5:
                return {"text": ""}
            
            return {"text": text.strip()}
        except Exception as e:
            print(f"Transcription error: {e}")
            return {"text": ""}
