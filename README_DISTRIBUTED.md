# MIA — Distributed Setup Guide (4 Laptops)

## What Changed & Why

Your original `main.py` ran **everything on one machine**, which caused slowness
because Whisper, wav2vec2, FER, and Ollama all load into RAM/VRAM at the same time.

The distributed version **splits the work** across 4 laptops on the same WiFi:

| Laptop | Role | Runs |
|--------|------|------|
| **A (yours)** | Coordinator + UI | `main_distributed.py` + React frontend |
| **B** | Audio AI | `workers/worker_asr_voice.py` |
| **C** | Vision AI | `workers/worker_vision.py` |
| **D** | Language AI | `workers/worker_llm_text.py` |

All 3 workers are called **in parallel** every 2 seconds — so you get the speed
of all 4 CPUs/GPUs working simultaneously.

---

## Step 1 — Find all laptop IPs

On each laptop, run:
- **Windows:** `ipconfig` → look for "IPv4 Address"
- **Mac/Linux:** `ifconfig` → look for `inet` on `en0` or `wlan0`

Example: `192.168.1.100`, `192.168.1.101`, etc.

---

## Step 2 — Edit `worker_config.py` (on Laptop A)

```python
LAPTOP_B_HOST = "192.168.1.2"   # <-- Laptop B's actual IP
LAPTOP_C_HOST = "192.168.1.4"   # <-- Laptop C's actual IP
LAPTOP_D_HOST = "192.168.1.1"   # <-- Laptop D's actual IP
```

---

## Step 3 — Install dependencies

### Laptop B
```bash
pip install fastapi uvicorn httpx pydantic torch transformers numpy
```

### Laptop C
```bash
pip install fastapi uvicorn httpx pydantic opencv-python-headless numpy fer
```

### Laptop D
```bash
pip install fastapi uvicorn httpx pydantic torch transformers
# Also install Ollama: https://ollama.com/download
ollama pull llama3.2
```

### Laptop A (yours)
```bash
pip install fastapi uvicorn httpx pydantic opencv-python pyaudio numpy edge-tts
```

---

## Step 4 — Copy files

- **Copy entire `backend/` folder** to all laptops (they share the `app/` modules)
- **Copy `worker_config.py`** only to Laptop A

---

## Step 5 — Start workers (Laptops B, C, D first)

**Laptop B:**
```bash
cd backend
python workers/worker_asr_voice.py
# Should print: ✅ Worker B ready.
# Listening on: http://0.0.0.0:8001
```

**Laptop C:**
```bash
cd backend
python workers/worker_vision.py
# Should print: ✅ Worker C ready.
# Listening on: http://0.0.0.0:8002
```

**Laptop D:**
```bash
ollama serve   # (in a separate terminal)
cd backend
python workers/worker_llm_text.py
# Should print: ✅ Worker D ready.
# Listening on: http://0.0.0.0:8003
```

---

## Step 6 — Start coordinator (Laptop A)

```bash
cd backend
uvicorn main_distributed:app --host 0.0.0.0 --port 8000 --reload
```

---

## Step 7 — Start frontend (Laptop A)

```bash
cd frontend
# Point frontend to YOUR laptop's IP:
VITE_BACKEND_HOST=192.168.1.100 npm run dev
# Or on Windows PowerShell:
$env:VITE_BACKEND_HOST="192.168.1.100"; npm run dev
```

---

## Step 8 — Check worker health

Open in browser:
```
http://localhost:8000/api/workers/health
```

You should see all 3 workers responding with `"status": "ok"`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Worker unreachable | Check firewall — allow port 8001/8002/8003 inbound |
| Worker B down | Emotion + transcription falls back to neutral gracefully |
| Slow on Worker D | Ollama llama3.2 is large — try `ollama pull llama3.2:1b` for speed |
| Frontend can't connect | Set `VITE_BACKEND_HOST` to Laptop A's IP, not localhost |

---

## Windows Firewall (allow worker ports)

Run on each worker laptop as Administrator:
```
netsh advfirewall firewall add rule name="MIA Worker" dir=in action=allow protocol=TCP localport=8001
netsh advfirewall firewall add rule name="MIA Worker" dir=in action=allow protocol=TCP localport=8002
netsh advfirewall firewall add rule name="MIA Worker" dir=in action=allow protocol=TCP localport=8003
```
