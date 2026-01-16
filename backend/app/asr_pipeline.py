from transformers import pipeline
import numpy as np
import torch
import re

class WhisperTranscriber:
    # Known Whisper hallucinations when audio is silent/quiet
    HALLUCINATION_PATTERNS = [
        r"thanks for watching",
        r"see you in the next",
        r"subscribe",
        r"like and share",
        r"don't forget to",
        r"please subscribe",
        r"bye bye",
        r"thank you for listening",
        r"music",
        r"applause",
        r"silence",
        r"\.{3,}",  # Multiple dots
    ]
    
    def __init__(self, model_name="openai/whisper-base.en"):
        print(f"Loading Whisper model: {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device=device,
        )
        # Compile hallucination patterns
        self.hallucination_regex = re.compile(
            '|'.join(self.HALLUCINATION_PATTERNS), 
            re.IGNORECASE
        )
        print(f"Whisper model loaded on {device}")
    
    def _is_audio_silent(self, audio_data, threshold=0.01):
        """Check if audio is mostly silent based on RMS energy."""
        if audio_data is None or len(audio_data) == 0:
            return True
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms < threshold
    
    def _is_hallucination(self, text):
        """Check if text matches known Whisper hallucination patterns."""
        if not text:
            return True
        return bool(self.hallucination_regex.search(text))

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
        
        # Check if audio is silent - skip transcription
        if self._is_audio_silent(audio_data, threshold=0.01):
            return {"text": ""}

        try:
            result = self.pipe(audio_data)
            text = result.get("text", "").strip()
            
            # Filter out noise/silence artifacts (dots, repeated chars, etc.)
            cleaned = re.sub(r'[.\s]+', ' ', text).strip()
            if len(cleaned) < 2 or cleaned.count('.') > len(cleaned) * 0.5:
                return {"text": ""}
            
            # Check for known hallucination patterns
            if self._is_hallucination(text):
                print(f"Filtered hallucination: {text}")
                return {"text": ""}
            
            return {"text": text}
        except Exception as e:
            print(f"Transcription error: {e}")
            return {"text": ""}
