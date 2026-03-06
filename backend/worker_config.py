"""
MIA Distributed Worker Config
==============================
Edit the IP addresses below to match your laptops' local network IPs.
Run `ipconfig` (Windows) or `ifconfig` (Mac/Linux) on each laptop to find its IP.

Laptop A (yours) = runs main_distributed.py + React frontend
Laptop B         = runs workers/worker_asr_voice.py
Laptop C         = runs workers/worker_vision.py
Laptop D         = runs workers/worker_llm_text.py
"""

# ─── SET THESE TO YOUR ACTUAL LOCAL IP ADDRESSES ───────────────────────────
LAPTOP_B_HOST = "192.168.1.2"   # Whisper + Voice Emotion
LAPTOP_C_HOST = "192.168.1.4"   # Vision / FER
LAPTOP_D_HOST = "192.168.1.1"   # Text Emotion + LLM
# ───────────────────────────────────────────────────────────────────────────

WORKER_B_URL = f"http://{LAPTOP_B_HOST}:8001"   # ASR + Voice Emotion
WORKER_C_URL = f"http://{LAPTOP_C_HOST}:8002"   # Vision Emotion
WORKER_D_URL = f"http://{LAPTOP_D_HOST}:8003"   # Text Emotion + LLM

# Timeout (seconds) when calling a remote worker
WORKER_TIMEOUT = 15.0
