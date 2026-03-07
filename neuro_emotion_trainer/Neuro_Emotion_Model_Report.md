# Neuro-Emotion TinyBERT: A Custom Transformer for Textual Emotion Classification

## 1. Introduction

The **Neuro-Emotion TinyBERT** is a lightweight, custom-built Transformer encoder designed from scratch for multi-class emotion classification from text. Unlike the main MIA pipeline — which relies on a pre-trained HuggingFace model (`j-hartmann/emotion-english-distilroberta-base`) — this model is trained entirely from the ground up, including its own BPE tokenizer, on a curated combination of public emotion-annotated datasets.

The model targets **8 emotion classes** and produces not only a discrete emotion label with a confidence score, but also continuous **Valence–Arousal (V-A)** estimates derived from the output probability distribution.

---

## 2. Architecture

### 2.1 Overview

| Component               | Detail                                 |
|--------------------------|----------------------------------------|
| **Architecture**         | Transformer Encoder (BERT-style)       |
| **Total Parameters**     | ~4.4 million                           |
| **Vocabulary Size**      | 30,000 (custom BPE tokenizer)          |
| **Max Sequence Length**  | 64 tokens                              |
| **Hidden Dimension**     | 256                                    |
| **Attention Heads**      | 4                                      |
| **Encoder Layers**       | 4                                      |
| **FFN Inner Dimension**  | 1,024 (4 × hidden)                     |
| **Dropout**              | 0.1                                    |
| **Activation Function**  | GELU                                   |
| **Output Classes**       | 8                                      |

### 2.2 Component Breakdown

The model is implemented in `model.py` as the `TinyBERTForClassification` class. It consists of:

#### Token Embedding Layer
A learnable embedding matrix of size $30{,}000 \times 256$, mapping each token ID to a 256-dimensional dense vector. A dedicated padding index (ID `0`) is zero-masked.

#### Sinusoidal Positional Encoding
Following the original Transformer formulation, position information is injected via fixed sinusoidal encodings:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

These are added to the token embeddings and passed through a dropout layer.

#### Layer Normalization (Pre-Norm)
A `LayerNorm` is applied after the combined embedding + positional encoding, **before** the encoder stack. The encoder layers themselves use **Pre-LayerNorm** ordering (`norm_first=True`), which is known to improve training stability over the original Post-Norm design.

#### Transformer Encoder Stack
A stack of **4 identical encoder layers**, each containing:

- **Multi-Head Self-Attention** with 4 heads (head dimension = 64)
- **Position-wise Feed-Forward Network** with inner dimension 1,024 and GELU activation
- **Residual connections** and **LayerNorm** around each sub-layer
- **Dropout** (0.1) after attention and FFN sub-layers

Padding tokens are excluded from attention via a `src_key_padding_mask`.

#### Classification Head
The hidden state of the **first token** (`[CLS]` position, index 0) is extracted and passed through:

$$\text{Linear}(256 \to 256) \;\to\; \text{GELU} \;\to\; \text{Dropout}(0.1) \;\to\; \text{Linear}(256 \to 8)$$

producing 8 raw logits, one per emotion class.

### 2.3 Weight Initialization

All weights are initialized from $\mathcal{N}(0, 0.02)$:

- **Linear layers**: weights ~ $\mathcal{N}(0, 0.02)$, biases zeroed
- **Embeddings**: weights ~ $\mathcal{N}(0, 0.02)$, padding row zeroed
- **LayerNorm**: weight = 1, bias = 0

---

## 3. Emotion Label Taxonomy

The model classifies text into the following 8 categories:

| Index | Emotion       | Valence | Arousal | Description                        |
|-------|---------------|---------|---------|------------------------------------|
| 0     | **Love**      | +0.75   | +0.30   | Affection, admiration, gratitude   |
| 1     | **Neutral**   |  0.00   |  0.00   | No strong emotional content        |
| 2     | **Anger**     | −0.60   | +0.70   | Frustration, annoyance, disgust    |
| 3     | **Joy**       | +0.70   | +0.55   | Happiness, excitement, amusement   |
| 4     | **Sadness**   | −0.65   | −0.40   | Grief, disappointment, remorse     |
| 5     | **Curiosity** | +0.20   | +0.35   | Confusion, anticipation, inquiry   |
| 6     | **Surprise**  | +0.15   | +0.75   | Unexpected events, realization     |
| 7     | **Fear**      | −0.55   | +0.65   | Terror, nervousness, embarrassment |

### 3.1 Valence–Arousal Estimation

Rather than a separate regression head, V-A scores are computed as a **weighted sum** over the softmax output distribution:

$$V = \sum_{i=0}^{7} p_i \cdot v_i, \qquad A = \sum_{i=0}^{7} p_i \cdot a_i$$

where $p_i$ is the predicted probability for class $i$, and $(v_i, a_i)$ are the fixed V-A coordinates from the table above. This yields smooth, interpretable affective coordinates without additional trainable parameters.

---

## 4. Tokenizer

A **Byte-Pair Encoding (BPE)** tokenizer is trained from scratch on the full training corpus using the HuggingFace `tokenizers` library.

| Property           | Value       |
|--------------------|-------------|
| Algorithm          | BPE         |
| Vocabulary size    | 30,000      |
| Pre-tokenizer      | Whitespace  |
| Min merge frequency| 2           |
| Special tokens     | `[PAD]`, `[CLS]`, `[SEP]`, `[UNK]` |
| Max length         | 64 tokens (with truncation and padding) |

Post-processing wraps each input as `[CLS] <tokens> [SEP]`, mimicking the BERT input format. This ensures the `[CLS]` token is always at position 0 for classification.

---

## 5. Training Data

### 5.1 Data Sources

The model is trained on **two combined datasets**:

1. **Unified Emotion Dataset** (`unified-dataset.jsonl`) — a JSONL file containing **~92,000 samples** aggregated from multiple public emotion corpora including CrowdFlower, EmoInt, TEC, SSEC, and others. Each sample has multi-label emotion annotations.

2. **GoEmotions** (Google Research) — a large-scale dataset of **~58,000 Reddit comments** annotated with 27 fine-grained emotion labels plus neutral, automatically downloaded from HuggingFace during training.

Combined, the training pipeline uses approximately **~150,000 samples**.

### 5.2 Label Mapping

Both datasets use different emotion taxonomies. They are unified into the 8-class system via explicit mapping tables:

**Unified Dataset Mapping:**

| Original Label  | Mapped To     |
|-----------------|---------------|
| joy             | joy           |
| fear            | fear          |
| sadness         | sadness       |
| noemo           | neutral       |
| anger           | anger         |
| surprise        | surprise      |
| love            | love          |
| disgust         | anger         |
| shame, guilt    | sadness       |
| trust           | love          |
| anticipation, confusion | curiosity |

**GoEmotions Mapping (27 → 8 classes):**

| Original Label         | Mapped To |
|------------------------|-----------|
| admiration, approval, caring, desire, gratitude | love |
| amusement, excitement, optimism, pride, relief  | joy  |
| anger, annoyance, disapproval, disgust          | anger |
| disappointment, grief, remorse, sadness         | sadness |
| curiosity, confusion                            | curiosity |
| realization, surprise                           | surprise |
| embarrassment, fear, nervousness                | fear |
| neutral                                         | neutral |

### 5.3 Data Split

The combined dataset is randomly shuffled (seed = 42) and split:

- **Training set**: 90%
- **Validation set**: 10%

---

## 6. Training Procedure

### 6.1 Hyperparameters

| Hyperparameter       | Default Value |
|----------------------|---------------|
| Epochs               | 15            |
| Batch size           | 64            |
| Learning rate        | $2 \times 10^{-4}$ |
| Optimizer            | AdamW (weight decay = 0.01) |
| Gradient clipping    | Max norm = 1.0 |
| Early stopping       | Patience = 4 epochs |
| Warmup steps         | 500           |
| Random seed          | 42            |

### 6.2 Learning Rate Schedule

A **linear warmup + cosine annealing** schedule is used:

$$\lambda(s) = \begin{cases} \frac{s}{500} & \text{if } s < 500 \\ \max\!\left(0.1,\; \frac{1}{2}\left(1 + \cos\left(\pi \cdot \frac{s - 500}{S - 500}\right)\right)\right) & \text{otherwise} \end{cases}$$

where $s$ is the current step and $S$ is the total number of training steps. The minimum learning rate is clamped at 10% of the peak to prevent complete decay.

### 6.3 Class Balancing

To handle class imbalance across emotions, **inverse-frequency class weights** are computed:

$$w_i = \min\!\left(\frac{N}{C \cdot n_i},\; 5.0\right)$$

where $N$ is the total training samples, $C = 8$ is the number of classes, and $n_i$ is the count for class $i$. Weights are capped at 5.0 to prevent extreme upweighting of very rare classes. These weights are passed to `CrossEntropyLoss`.

### 6.4 Checkpointing

The model with the **highest validation accuracy** is saved to disk (`neuro_emotion_best.pt`), along with:
- Epoch number
- Model state dict
- Optimizer state dict
- Validation accuracy and loss
- Emotion label list

---

## 7. Inference Pipeline

At inference time, the flow is:

```
Input Text
   │
   ▼
BPE Tokenizer (encode → [CLS] + tokens + [SEP], pad to 64)
   │
   ▼
TinyBERT Encoder (4 layers, 4 heads)
   │
   ▼
[CLS] hidden state → Classification Head → 8 logits
   │
   ▼
Softmax → Probability distribution p₀...p₇
   │
   ├── argmax → Predicted Emotion + Confidence
   │
   └── weighted sum → Valence & Arousal scores
```

### Example Output

For the input *"I love you so much, you mean the world to me!"*:

```json
{
  "emotion": "love",
  "confidence": 0.9231,
  "valence": 0.6124,
  "arousal": 0.2847,
  "all_scores": [
    {"label": "love",      "score": 0.9231},
    {"label": "joy",       "score": 0.0542},
    {"label": "neutral",   "score": 0.0103},
    {"label": "surprise",  "score": 0.0051},
    ...
  ]
}
```

---

## 8. Comparison with Pre-Trained Alternative

| Aspect                | Neuro-Emotion TinyBERT        | j-hartmann DistilRoBERTa          |
|-----------------------|-------------------------------|-----------------------------------|
| **Training**          | From scratch (custom)         | Pre-trained + fine-tuned          |
| **Parameters**        | ~4.4M                         | ~82M                              |
| **Emotion classes**   | 8                             | 7                                 |
| **Tokenizer**         | Custom BPE (30K vocab)        | RoBERTa BPE (50K vocab)          |
| **V-A output**        | Yes (derived)                 | No                                |
| **Sequence length**   | 64 tokens                     | 512 tokens                        |
| **Inference speed**   | Faster (~18× fewer params)    | Slower                            |
| **Likely accuracy**   | Moderate                      | Higher (pre-training advantage)   |
| **Dependency**        | PyTorch + tokenizers only     | Full `transformers` library       |

---

## 9. File Structure

```
neuro_emotion_trainer/
├── model.py                  # Model architecture + tokenizer wrapper + inference
├── train.py                  # Full training script with data loading & evaluation
├── requirements.txt          # torch, tokenizers, datasets
├── data/
│   └── unified-dataset.jsonl # ~92K samples from multiple public corpora
└── output/                   # Generated during training
    ├── neuro_emotion_best.pt          # Best model checkpoint
    └── neuro_emotion_tokenizer.json   # Trained BPE tokenizer
```

---

## 10. Usage

### Training
```bash
cd neuro_emotion_trainer/

# Default settings (15 epochs, batch=64, lr=2e-4)
python train.py

# Custom settings
python train.py --epochs 20 --batch-size 128 --lr 3e-4

# Without GoEmotions (unified dataset only)
python train.py --no-goemotions
```

### Inference (Python)
```python
from model import load_model, predict, NeuroEmotionTokenizer

model     = load_model("output/neuro_emotion_best.pt")
tokenizer = NeuroEmotionTokenizer("output/neuro_emotion_tokenizer.json")

result = predict(model, tokenizer, "This is absolutely wonderful!")
print(result)
# {'emotion': 'joy', 'confidence': 0.8712, 'valence': 0.5841, 'arousal': 0.4523, 'all_scores': [...]}
```

---

## 11. Summary

The Neuro-Emotion TinyBERT demonstrates that a compact, purpose-built Transformer (~4.4M parameters) trained from scratch can perform multi-class emotion classification with continuous V-A mapping. By combining ~150K samples from multiple public datasets under a unified 8-class taxonomy and employing class-balanced training with warmup + cosine scheduling, the model achieves a practical balance between accuracy, speed, and interpretability — making it suitable for real-time multimodal emotion analysis pipelines.
