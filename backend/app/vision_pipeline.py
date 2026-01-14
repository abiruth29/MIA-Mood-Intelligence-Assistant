"""
Vision Pipeline for MIA - Facial Expression Recognition
Uses FER (Facial Expression Recognition) library for emotion classification.
"""

import cv2
import numpy as np
from collections import deque
import threading

# For facial expression recognition (includes face detection)
from fer.fer import FER

class VisionEmotionClassifier:
    """
    Analyzes video frames for:
    1. Facial expression/emotion (7 classes via FER)
    2. Basic head pose estimation using face bounding box
    3. Engagement estimation
    """
    
    # Emotion label mapping to match audio/text pipeline
    EMOTION_MAP = {
        'angry': 'anger',
        'disgust': 'disgust', 
        'fear': 'fear',
        'happy': 'happiness',
        'sad': 'sadness',
        'surprise': 'surprise',
        'neutral': 'neutral'
    }
    
    def __init__(self):
        print("Loading FER emotion detector...")
        # FER uses MTCNN by default, but we can use opencv for speed
        self.emotion_detector = FER(mtcnn=False)  # Use OpenCV cascade (faster)
        
        # Temporal smoothing - store last N emotion predictions
        self.emotion_history = deque(maxlen=10)  # ~5 seconds at 2 FPS processing
        
        # Track face position for engagement
        self.last_face_box = None
        self.face_center_history = deque(maxlen=5)
        
        print("Vision pipeline ready.")
    
    def process_frame(self, frame: np.ndarray, draw_landmarks: bool = True) -> dict:
        """
        Process a single video frame.
        
        Args:
            frame: BGR image from OpenCV
            draw_landmarks: Whether to draw face box on the frame
            
        Returns:
            dict with keys: emotion, confidence, all_scores, head_pose, gaze, engagement, annotated_frame
        """
        result = {
            "emotion": "neutral",
            "confidence": 0.0,
            "all_scores": [],
            "landmarks": None,
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
            "gaze": "unknown",
            "engagement": 0.5,
            "annotated_frame": frame
        }
        
        if frame is None:
            return result
        
        if not hasattr(frame, 'size') or frame.size == 0:
            return result
        
        h, w = frame.shape[:2]
        frame_center = (w // 2, h // 2)
        
        # Detect emotions using FER
        emotion_result = self._detect_emotion(frame)
        
        if emotion_result:
            result["emotion"] = emotion_result["emotion"]
            result["confidence"] = emotion_result["confidence"]
            result["all_scores"] = emotion_result["all_scores"]
            face_box = emotion_result.get("box")
            
            if face_box is not None:
                self.last_face_box = face_box
                # Calculate face center
                fx, fy, fw, fh = face_box
                face_center = (fx + fw // 2, fy + fh // 2)
                self.face_center_history.append(face_center)
                
                # Estimate head pose from face position relative to frame center
                result["head_pose"] = self._estimate_head_pose(face_center, (w // 2, h // 2), w, h)
                
                # Estimate gaze/engagement
                result["gaze"], result["engagement"] = self._estimate_engagement(
                    face_center, (w // 2, h // 2), face_box, w, h
                )
                
                # Draw face box if requested
                if draw_landmarks:
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                    # Draw emotion label
                    label = f"{result['emotion']}: {result['confidence']*100:.0f}%"
                    cv2.putText(frame, label, (fx, fy - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Apply temporal smoothing
            self.emotion_history.append(emotion_result)
            smoothed = self._get_smoothed_emotion()
            result["emotion"] = smoothed["emotion"]
            result["confidence"] = smoothed["confidence"]
            result["all_scores"] = smoothed["all_scores"]
        
        result["annotated_frame"] = frame
        return result
    
    def _detect_emotion(self, frame: np.ndarray) -> dict:
        """Detect facial emotion using FER library."""
        try:
            # FER can work with BGR (OpenCV format)
            emotions = self.emotion_detector.detect_emotions(frame)
            
            if emotions and len(emotions) > 0:
                # Take the first (largest) face
                face_data = emotions[0]
                face_emotions = face_data.get("emotions", {})
                face_box = face_data.get("box")  # (x, y, w, h)
                
                if not face_emotions:
                    return None
                
                # Find dominant emotion
                dominant = max(face_emotions, key=face_emotions.get)
                confidence = face_emotions[dominant]
                
                # Normalize label to match our pipeline
                normalized_emotion = self.EMOTION_MAP.get(dominant, dominant)
                
                # Build all_scores list
                all_scores = []
                for emo, score in face_emotions.items():
                    normalized = self.EMOTION_MAP.get(emo, emo)
                    all_scores.append({"label": normalized, "score": round(float(score), 3)})
                
                return {
                    "emotion": normalized_emotion,
                    "confidence": round(float(confidence), 3),
                    "all_scores": all_scores,
                    "box": face_box
                }
        except Exception as e:
            print(f"FER error: {e}")
        
        return None
    
    def _estimate_head_pose(self, face_center: tuple, frame_center: tuple, w: int, h: int) -> dict:
        """
        Estimate approximate head pose based on face position in frame.
        This is a simplified estimation without 3D landmarks.
        """
        # Calculate offset from center (normalized -1 to 1)
        x_offset = (face_center[0] - frame_center[0]) / (w / 2)
        y_offset = (face_center[1] - frame_center[1]) / (h / 2)
        
        # Approximate yaw and pitch (in degrees)
        # Face to the right = positive yaw, face down = positive pitch
        yaw = x_offset * 30  # Max ~30 degrees
        pitch = y_offset * 25  # Max ~25 degrees
        
        return {
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
            "roll": 0  # Can't estimate roll without landmarks
        }
    
    def _estimate_engagement(self, face_center: tuple, frame_center: tuple, 
                            face_box: tuple, w: int, h: int) -> tuple:
        """
        Estimate user engagement based on face position and size.
        Returns (gaze_direction, engagement_score)
        """
        fx, fy, fw, fh = face_box
        
        # Calculate how centered the face is (0 = edge, 1 = center)
        x_offset = abs(face_center[0] - frame_center[0]) / (w / 2)
        y_offset = abs(face_center[1] - frame_center[1]) / (h / 2)
        
        # Face size relative to frame (larger = closer/more engaged)
        face_size_ratio = (fw * fh) / (w * h)
        
        # Determine gaze
        if x_offset < 0.2 and y_offset < 0.2:
            gaze = "engaged"
            base_engagement = 0.9
        elif x_offset > 0.5:
            gaze = "looking_away"
            base_engagement = 0.3
        elif y_offset > 0.4 and face_center[1] > frame_center[1]:
            gaze = "looking_down"
            base_engagement = 0.4
        elif y_offset > 0.4 and face_center[1] < frame_center[1]:
            gaze = "looking_up"
            base_engagement = 0.5
        else:
            gaze = "partially_engaged"
            base_engagement = 0.6
        
        # Adjust engagement by face size (closer = more engaged)
        size_bonus = min(face_size_ratio * 5, 0.2)  # Up to 0.2 bonus
        engagement = min(base_engagement + size_bonus, 1.0)
        
        return gaze, round(engagement, 2)
    
    def _get_smoothed_emotion(self) -> dict:
        """
        Apply temporal smoothing using weighted moving average.
        Recent predictions have higher weight.
        """
        if not self.emotion_history:
            return {"emotion": "neutral", "confidence": 0.0, "all_scores": []}
        
        # Aggregate scores with time-decay weighting
        aggregated = {}
        total_weight = 0
        
        for i, entry in enumerate(self.emotion_history):
            weight = (i + 1) / len(self.emotion_history)  # Later entries have higher weight
            total_weight += weight
            
            for score_item in entry.get("all_scores", []):
                label = score_item["label"]
                score = score_item["score"]
                aggregated[label] = aggregated.get(label, 0) + score * weight
        
        # Normalize
        if total_weight > 0:
            for label in aggregated:
                aggregated[label] /= total_weight
        
        if aggregated:
            final_emotion = max(aggregated, key=aggregated.get)
            final_confidence = aggregated[final_emotion]
            all_scores = [{"label": k, "score": round(v, 3)} for k, v in aggregated.items()]
            
            return {
                "emotion": final_emotion,
                "confidence": round(final_confidence, 3),
                "all_scores": all_scores
            }
        
        return {"emotion": "neutral", "confidence": 0.0, "all_scores": []}
    
    def classify_frame_simple(self, frame: np.ndarray) -> dict:
        """
        Simplified interface matching other classifiers.
        Returns just emotion classification without landmarks/pose.
        """
        result = self.process_frame(frame, draw_landmarks=False)
        return {
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"]
        }


class VisionProcessor:
    """
    Wrapper class to process frames from the video capture stream.
    Handles threading and frame buffering for async processing.
    """
    
    def __init__(self):
        self.classifier = VisionEmotionClassifier()
        self.latest_result = None
        self._lock = threading.Lock()
    
    def process(self, frame_bgr: np.ndarray) -> dict:
        """Process a frame and return emotion result."""
        result = self.classifier.process_frame(frame_bgr, draw_landmarks=True)
        with self._lock:
            self.latest_result = result
        return result
    
    def get_latest(self) -> dict:
        """Get the most recent processing result."""
        with self._lock:
            return self.latest_result
