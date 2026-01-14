import cv2
import pyaudio
import numpy as np
import threading
import time
import base64

class VideoCapture:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.cap = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.device_id)
        self.running = True
        print(f"Video capture started on device {self.device_id}")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.cap = None
        print("Video capture stopped")

    def get_frame(self):
        if not self.running or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def get_raw_frame(self):
        """Get raw BGR frame for vision processing (without encoding)."""
        return self.get_frame()

    def get_jpeg_frame(self):
        frame = self.get_frame()
        if frame is None:
            return None
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        
        # Convert to base64 string
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text

class AudioCapture:
    def __init__(self, rate=16000, chunk=1024):
        self.rate = rate
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.buffer = []
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.buffer = []
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk,
                                  stream_callback=self._callback)
        self.running = True
        print("Audio capture started")

    def _callback(self, in_data, frame_count, time_info, status):
        with self.lock:
            self.buffer.append(np.frombuffer(in_data, dtype=np.int16))
        return (in_data, pyaudio.paContinue)

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.stream = None
        print("Audio capture stopped")

    def get_buffer(self):
        with self.lock:
            if not self.buffer:
                return np.array([], dtype=np.int16)
            data = np.concatenate(self.buffer)
            self.buffer = [] # Clear buffer after reading
            return data

    def __del__(self):
        self.p.terminate()
