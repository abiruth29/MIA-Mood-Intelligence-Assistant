from transformers import pipeline
import torch

class TextEmotionClassifier:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        print(f"Loading Text Emotion model: {model_name}...")
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            top_k=None # Return all scores
        )
        print("Text Emotion model loaded.")

    def classify(self, text):
        """
        Classify emotion from text.
        Args:
            text: str
        Returns:
            dict: {"emotion": str, "confidence": float, "all_scores": list}
        """
        if not text or not text.strip():
            return {"emotion": "neutral", "confidence": 0.0, "all_scores": []}

        try:
            results = self.pipe(text)
            # results is a list of lists (batch size 1)
            # [[{'label': 'joy', 'score': 0.9}, ...]]
            
            if results and len(results) > 0:
                scores = results[0]
                # Sort by score desc
                scores.sort(key=lambda x: float(x['score']), reverse=True)
                
                top_emotion = scores[0]['label']
                top_confidence = float(scores[0]['score'])
                
                # Map 'joy' to 'happiness' to match common audio models if needed
                # But let's keep it raw for now and map in fusion if needed
                
                return {
                    "emotion": top_emotion,
                    "confidence": round(top_confidence, 3),
                    "all_scores": [{"label": s["label"], "score": round(float(s["score"]), 3)} for s in scores]
                }
            
            return {"emotion": "neutral", "confidence": 0.0, "all_scores": []}
            
        except Exception as e:
            print(f"Text Emotion classification error: {e}")
            return {"emotion": "neutral", "confidence": 0.0, "all_scores": []}
