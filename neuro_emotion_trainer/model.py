"""
Neuro-Emotion Model Architecture
TinyBERT ~4.4M params, 8-class emotion classifier
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

NUM_CLASSES  = 8
VOCAB_SIZE   = 30000
MAX_SEQ_LEN  = 64
HIDDEN_SIZE  = 256
NUM_HEADS    = 4
NUM_LAYERS   = 4
FFN_DIM      = HIDDEN_SIZE * 4
DROPOUT      = 0.1
PAD_TOKEN_ID = 0

EMOTION_LABELS = ["love", "neutral", "anger", "joy", "sadness", "curiosity", "surprise", "fear"]

EMOTION_VA_MAP = {
    "love":      {"valence":  0.75, "arousal":  0.30},
    "neutral":   {"valence":  0.00, "arousal":  0.00},
    "anger":     {"valence": -0.60, "arousal":  0.70},
    "joy":       {"valence":  0.70, "arousal":  0.55},
    "sadness":   {"valence": -0.65, "arousal": -0.40},
    "curiosity": {"valence":  0.20, "arousal":  0.35},
    "surprise":  {"valence":  0.15, "arousal":  0.75},
    "fear":      {"valence": -0.55, "arousal":  0.65},
}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TinyBERTForClassification(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 num_heads=NUM_HEADS, ffn_dim=FFN_DIM, max_seq_len=MAX_SEQ_LEN,
                 num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_embedding    = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_TOKEN_ID)
        self.positional_encoding = PositionalEncoding(hidden_size, max_seq_len, dropout)
        self.embedding_norm     = nn.LayerNorm(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.padding_idx is not None:
                    with torch.no_grad(): m.weight[m.padding_idx].fill_(0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.embedding_norm(x)
        mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        return self.classifier(x[:, 0, :])


class NeuroEmotionTokenizer:
    """Wrapper around a HuggingFace tokenizers BPE tokenizer."""
    def __init__(self, tokenizer_path):
        from tokenizers import Tokenizer
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = MAX_SEQ_LEN

    def encode(self, text):
        enc = self.tokenizer.encode(text)
        ids, mask = enc.ids, enc.attention_mask
        if len(ids) > self.max_len:
            ids, mask = ids[:self.max_len], mask[:self.max_len]
        pad = self.max_len - len(ids)
        return {"input_ids": ids + [PAD_TOKEN_ID]*pad, "attention_mask": mask + [0]*pad}


def load_model(model_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyBERTForClassification()
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return model.to(device).eval()


def predict(model, tokenizer, text, device="cpu"):
    enc = tokenizer.encode(text)
    ids  = torch.tensor([enc["input_ids"]], dtype=torch.long).to(device)
    mask = torch.tensor([enc["attention_mask"]], dtype=torch.long).to(device)
    with torch.no_grad():
        probs = F.softmax(model(ids, mask), dim=-1).squeeze(0)
    conf, idx = torch.max(probs, 0)
    emotion = EMOTION_LABELS[idx.item()]
    all_scores = sorted(
        [{"label": EMOTION_LABELS[i], "score": round(probs[i].item(), 4)} for i in range(NUM_CLASSES)],
        key=lambda x: x["score"], reverse=True
    )
    valence = sum(probs[i].item() * EMOTION_VA_MAP[EMOTION_LABELS[i]]["valence"] for i in range(NUM_CLASSES))
    arousal = sum(probs[i].item() * EMOTION_VA_MAP[EMOTION_LABELS[i]]["arousal"] for i in range(NUM_CLASSES))
    return {"emotion": emotion, "confidence": round(conf.item(), 4),
            "valence": round(valence, 4), "arousal": round(arousal, 4), "all_scores": all_scores}
