"""
train.py — Neuro-Emotion TinyBERT Training Script
===================================================

Trains on the unified-dataset.jsonl from unify-emotion-datasets.
Automatically uses GPU if available.

Usage:
    python train.py
    python train.py --epochs 20 --batch-size 128

Output:
    output/neuro_emotion_best.pt
    output/neuro_emotion_tokenizer.json
"""

import os
import sys
import json
import random
import math
import argparse
import time
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import (
    TinyBERTForClassification, EMOTION_LABELS, NUM_CLASSES, MAX_SEQ_LEN, predict, load_model
)

# ──────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Neuro-Emotion TinyBERT")
parser.add_argument("--epochs",     type=int,   default=15)
parser.add_argument("--batch-size", type=int,   default=64)
parser.add_argument("--lr",         type=float, default=2e-4)
parser.add_argument("--patience",   type=int,   default=4)
parser.add_argument("--data",       type=str,   default="data/unified-dataset.jsonl",
                    help="Path to unified-dataset.jsonl")
parser.add_argument("--no-goemotions", action="store_true",
                    help="Skip downloading GoEmotions (use only unified JSONL)")
args = parser.parse_args()

SEED         = 42
WARMUP_STEPS = 500
VOCAB_SIZE   = 30000
OUTPUT_DIR   = "output"
MODEL_PATH   = os.path.join(OUTPUT_DIR, "neuro_emotion_best.pt")
TOKENIZER_PATH = os.path.join(OUTPUT_DIR, "neuro_emotion_tokenizer.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Label mappings
# ──────────────────────────────────────────────────────────────
UNIFIED_LABEL_MAP = {
    "joy": "joy", "fear": "fear", "sadness": "sadness", "noemo": "neutral",
    "anger": "anger", "surprise": "surprise", "love": "love",
    "disgust": "anger", "shame": "sadness", "guilt": "sadness",
    "trust": "love", "anticipation": "curiosity", "confusion": "curiosity",
}

GOEMOTIONS_LABEL_MAP = {
    "admiration": "love", "amusement": "joy", "anger": "anger",
    "annoyance": "anger", "approval": "love", "caring": "love",
    "confusion": "curiosity", "curiosity": "curiosity", "desire": "love",
    "disappointment": "sadness", "disapproval": "anger", "disgust": "anger",
    "embarrassment": "fear", "excitement": "joy", "fear": "fear",
    "gratitude": "love", "grief": "sadness", "joy": "joy",
    "love": "love", "nervousness": "fear", "optimism": "joy",
    "pride": "joy", "realization": "surprise", "relief": "joy",
    "remorse": "sadness", "sadness": "sadness", "surprise": "surprise",
    "neutral": "neutral",
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────
def load_unified_jsonl(path) -> List[Tuple[str, int]]:
    label_to_idx = {name: idx for idx, name in enumerate(EMOTION_LABELS)}
    samples, skipped = [], 0

    print(f"  Loading unified JSONL from: {path}")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found!")
        print("  Make sure 'data/unified-dataset.jsonl' exists in this folder.")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text or len(text) < 5:
                skipped += 1; continue
            emotions = obj.get("emotions", {})
            best_emo, best_val = None, 0
            for emo_key, emo_val in emotions.items():
                if emo_val is not None and emo_val > best_val:
                    best_emo, best_val = emo_key, emo_val
            if best_emo is None:
                skipped += 1; continue
            mapped = UNIFIED_LABEL_MAP.get(best_emo)
            if mapped is None:
                skipped += 1; continue
            idx = label_to_idx.get(mapped)
            if idx is not None:
                samples.append((text, idx))
            else:
                skipped += 1

    print(f"  -> {len(samples):,} loaded, {skipped:,} skipped")
    return samples


def load_goemotions() -> List[Tuple[str, int]]:
    print("  Loading GoEmotions from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  WARNING: 'datasets' not installed. Skipping GoEmotions.")
        print("           Install with: pip install datasets")
        return []

    dataset = load_dataset(
        "google-research-datasets/go_emotions", "simplified",
        trust_remote_code=True
    )
    go_label_names = dataset["train"].features["labels"].feature.names
    label_to_idx = {name: idx for idx, name in enumerate(EMOTION_LABELS)}
    samples = []

    for split in ["train", "validation", "test"]:
        for example in dataset[split]:
            text = example["text"].strip()
            if not text or not example["labels"]:
                continue
            go_label_name = go_label_names[example["labels"][0]]
            mapped = GOEMOTIONS_LABEL_MAP.get(go_label_name)
            if mapped is None: continue
            idx = label_to_idx.get(mapped)
            if idx is not None:
                samples.append((text, idx))

    print(f"  -> {len(samples):,} samples")
    return samples


# ──────────────────────────────────────────────────────────────
# Tokenizer training
# ──────────────────────────────────────────────────────────────
def train_tokenizer(texts, save_path):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    print("\nTraining BPE tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]"],
        min_frequency=2, show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=MAX_SEQ_LEN
    )
    tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
    tokenizer.save(save_path)
    print(f"  Tokenizer saved -> {save_path}")


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels, self.tokenizer = texts, labels, tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode(self.texts[idx])
        ids  = enc.ids[:MAX_SEQ_LEN]
        mask = enc.attention_mask[:MAX_SEQ_LEN]
        pad  = MAX_SEQ_LEN - len(ids)
        ids  += [0] * pad
        mask += [0] * pad
        return {
            "input_ids":      torch.tensor(ids,  dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────
def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  Neuro-Emotion TinyBERT Trainer")
    print(f"{'='*70}")
    print(f"  Device : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Epochs : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'='*70}\n")

    # Load data
    print("Loading datasets...")
    all_samples = load_unified_jsonl(args.data)
    if not args.no_goemotions:
        all_samples.extend(load_goemotions())

    random.shuffle(all_samples)
    texts  = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]
    print(f"\nTotal: {len(texts):,} samples")

    # Class distribution
    dist = Counter(labels)
    print("\nClass distribution:")
    for i, name in enumerate(EMOTION_LABELS):
        count = dist.get(i, 0)
        pct   = count / len(labels) * 100
        print(f"  {name:>12s}: {count:>6,} ({pct:5.1f}%) {'#' * int(pct)}")

    # Train tokenizer
    train_tokenizer(texts, TOKENIZER_PATH)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # Split 90/10
    split = int(0.9 * len(texts))
    train_texts, val_texts = texts[:split], texts[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    train_ds = EmotionDataset(train_texts, train_labels, tokenizer)
    val_ds   = EmotionDataset(val_texts,   val_labels,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
    print(f"\nTrain: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # Class weights
    class_counts = Counter(train_labels)
    weights = [min(len(train_labels) / (len(EMOTION_LABELS) * class_counts.get(i, 1)), 5.0)
               for i in range(len(EMOTION_LABELS))]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"\nClass weights: {[f'{w:.2f}' for w in weights]}")

    # Model
    model = TinyBERTForClassification().to(device)
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    total_steps = len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc, best_epoch, patience_counter = 0.0, -1, 0
    batches_per_epoch = len(train_loader)

    print(f"\nStarting training: {args.epochs} epochs x {batches_per_epoch} batches...")
    print("=" * 70)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, batch in enumerate(train_loader, 1):
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            target = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss   = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss    += loss.item() * target.size(0)
            preds          = logits.argmax(-1)
            train_correct += (preds == target).sum().item()
            train_total   += target.size(0)

            # Progress every 100 batches
            if batch_idx % 100 == 0:
                running_acc = 100.0 * train_correct / train_total
                elapsed = time.time() - epoch_start
                eta = elapsed / batch_idx * (batches_per_epoch - batch_idx)
                print(f"  Epoch {epoch:>2d} | Batch {batch_idx:>4d}/{batches_per_epoch} | "
                      f"Loss: {loss.item():.4f} | Acc: {running_acc:.1f}% | ETA: {eta:.0f}s",
                      flush=True)

        avg_train_loss = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids    = batch["input_ids"].to(device)
                mask   = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                logits = model(ids, mask)
                loss   = criterion(logits, target)
                val_loss    += loss.item() * target.size(0)
                preds        = logits.argmax(-1)
                val_correct += (preds == target).sum().item()
                val_total   += target.size(0)

        avg_val_loss = val_loss / val_total
        val_acc      = 100.0 * val_correct / val_total
        epoch_time   = time.time() - epoch_start

        # ── Checkpoint ──
        improved = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, patience_counter = val_acc, epoch, 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc, "val_loss": avg_val_loss,
                "emotion_labels": EMOTION_LABELS,
            }, MODEL_PATH)
            improved = " << BEST"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:>2d}/{args.epochs} | "
            f"Train: loss={avg_train_loss:.4f} acc={train_acc:.1f}% | "
            f"Val: loss={avg_val_loss:.4f} acc={val_acc:.1f}% | "
            f"Time: {epoch_time:.0f}s{improved}"
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stop at epoch {epoch}")
            break

    print(f"\n{'='*70}")
    print(f"Training done!")
    print(f"  Best val accuracy : {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Model saved       : {MODEL_PATH}")
    print(f"  Tokenizer saved   : {TOKENIZER_PATH}")

    # ── Inference test ──
    print(f"\n{'='*70}")
    print("Quick inference test:\n")
    from model import NeuroEmotionTokenizer
    inf_model = load_model(MODEL_PATH, device=str(device))
    inf_tok   = NeuroEmotionTokenizer(TOKENIZER_PATH)

    test_phrases = [
        "I love you so much, you mean the world to me!",
        "I am so angry right now, this is completely unacceptable!",
        "What a wonderful surprise, I didn't expect this at all!",
        "I feel really sad and depressed today.",
        "I'm terrified of what might happen next.",
        "The weather is about average for this time of year.",
        "This is the happiest day of my entire life!",
        "My breath hitched and my pulse hammered as shadows stretched toward me.",
        "The sun settled behind my ribs and warmth radiated to my very fingertips.",
    ]
    for phrase in test_phrases:
        r = predict(inf_model, inf_tok, phrase, device=str(device))
        disp = f'"{phrase[:60]}..."' if len(phrase) > 60 else f'"{phrase}"'
        print(f"  {disp}")
        print(f"     -> {r['emotion'].upper():<10} conf={r['confidence']:.3f}  V={r['valence']:+.2f}  A={r['arousal']:+.2f}\n")

    print("Copy output/neuro_emotion_best.pt and output/neuro_emotion_tokenizer.json")
    print("back to your laptop's MIA/backend/models/ folder. Done!")


if __name__ == "__main__":
    train()
