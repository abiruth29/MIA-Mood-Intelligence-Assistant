# Neuro-Emotion TinyBERT — GPU Training Package

Self-contained folder to train the TinyBERT emotion model on a GPU laptop.

## Structure
```
neuro_emotion_trainer/
├── model.py           ← TinyBERT architecture (~4.4M params)
├── train.py           ← Training script
├── requirements.txt   ← Dependencies
├── data/
│   └── unified-dataset.jsonl   ← Copy dataset here
└── output/            ← Created during training
    ├── neuro_emotion_best.pt
    └── neuro_emotion_tokenizer.json
```

## Steps

### 1. Copy dataset into `data/`
```
unified-dataset.jsonl → neuro_emotion_trainer/data/
```

### 2. Install dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tokenizers datasets
```

### 3. Train
```bash
python train.py
python train.py --epochs 20 --batch-size 128   # custom options
python train.py --no-goemotions                 # skip HuggingFace download
```

| Hardware | ~Time/Epoch | Total     |
| -------- | ----------- | --------- |
| RTX 3060 | 2-4 min     | 30-60 min |
| RTX 4070 | 1-2 min     | 15-30 min |
| CPU only | 20-30 min   | 5-7 hours |

### 4. Copy output back
```
output/neuro_emotion_best.pt       → MIA/backend/models/
output/neuro_emotion_tokenizer.json → MIA/backend/models/
```
