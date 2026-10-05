
<div align="center">

# Investigating the Possibility of Improving Persian Automatic Speech Recognition by Combining Outputs of Existing Models

**A Multi-View Fusion Study of Whisper and MMS for Persian Post-ASR Correction**


*Bachelor's Thesis — Isfahan University of Technology*

*Author: Zahra Tavakoli · Supervisor: Dr. Zeinab Maleki · Summer 1405 (2026)*

</div>

---

## Overview

This repository contains the full implementation, experiments, and analysis for a Bachelor's thesis project titled **"Investigating the Possibility of Improving Persian Automatic Speech Recognition by Combining Outputs of Existing Models."**

The project investigates whether the **gated multi-view fusion architecture** — originally proposed for the low-resource **Rajasthani** language (Devanagari script) by Bhandari & Harit (2026) — can effectively improve **Persian ASR** by combining the outputs of two heterogeneous models: **Whisper-large-v3** (Transformer-based) and **MMS-1B-All** (wav2vec 2.0-based).

> ### Key Result — A Negative Finding with Theoretical Analysis
>
> The gated fusion architecture **fails for Persian**. The gate collapses to a **positional heuristic** (r = +0.77 with character position, partial r ≈ 0 with content correctness), because Persian ASR errors are **lexical and memory-based** (homophones like ذ/ز/ض/ظ, س/ص/ث, ت/ط, ه/ح) rather than **positional and structural** (like vowel matras in Devanagari).
>
> This is a **systematic study** showing that a state-of-the-art multi-view fusion method does not transfer across script families — a **fundamental lesson** for multilingual ASR system design.

---

## Key Findings

### 1. Multi-View Fusion Cannot Outperform the Best Single View on Persian

| Model | CER ↓ | WER ↓ | Exact Match ↑ |
|---|---|---|---|
| Whisper-large-v3 (raw) | 0.0980 | 0.3283 | 25.87% |
| **MMS-1B-All (raw)** | **0.0702** | **0.2841** | **28.91%** |
| Subword model (SentencePiece, 1280 tokens) | 0.1897 | 0.3998 | — |
| **Character model + gap alignment** (best fusion) | 0.0726 | 0.2855 | **29.65%** |
| Character model + longer-string alignment | 0.0830 | 0.3130 | — |
| Paper formula + Gate Variance | 0.0774 | 0.3026 | — |
| Convex + Gate Variance | 0.0786 | 0.3062 | — |

**Conclusions:**
- The best fusion model (CER = 0.0726) is only 3.4% worse than the raw MMS baseline.
- The encoder-decoder part provides most of the improvement over Whisper, **not** the gate.

### 2. Gate Collapse into a Positional Heuristic

We performed a statistical hypothesis test on **57,958 data points** to determine whether the gate is content-aware or position-based:

| Correlation Test | r | p-value | Interpretation |
|---|---|---|---|
| gate ↔ **position** | **+0.7744** | < 0.001 | Gate is strongly position-dependent |
| gate ↔ MMS correctness | −0.2970 | < 0.001 | Apparent content awareness |
| gate ↔ Whisper correctness | −0.2930 | < 0.001 | Apparent content awareness |
| **Partial** gate ↔ MMS (control position) | **−0.0504** | — | **Content signal nearly disappears** |
| **Partial** gate ↔ Whisper (control position) | **−0.0573** | — | **Content signal nearly disappears** |

**Interpretation:** Over **80%** of the apparent gate-content correlation is a **spurious artifact** caused by the shared confound of character position. Once position is controlled, the gate's content-awareness vanishes into statistical noise.

### 3. Root Cause: Lexical vs. Positional Errors

| Feature | Rajasthani (Devanagari) ✓ | Persian ✗ |
|---|---|---|
| Error location | Local, positional | Lexical, memory-based |
| Distribution pattern | Systematic, learnable | Random at character level |
| Examples | matras, aspiration, word boundaries | ذ/ز/ض/ظ, س/ص/ث, ت/ط, ه/ح |
| Predictable from position | Yes | No |


---


### Data Preparation

We use the [Common Voice 22.0 Persian subset](https://huggingface.co/datasets/aliyzd95/common_voice_22_0_fa), which contains **51,141 audio samples** with corresponding transcriptions, split speaker-independently into train/validation/test.


The pipeline:
1. **Whisper-large-v3** (`language="fa"`, `task="transcribe"`) → first view
2. **MMS-1B-All** (`target_lang="fas"`) → second view
3. **Shekar normalization:** alphabet, digits, punctuation, diacritics, spacing
4. **Regex-based correction** for Common Voice typos (e.g., «خانه‌ای»‌ → «خانهای»)
5. **Invalid sample filtering** (CER/WER too high, empty or invalid ASR outputs, low Persian character ratio)

**Final dataset:**

| Split | Initial Samples | Removed | Final Samples|
|---|---|---|---|
| Train | 29,789 | 657 | **29,132** |
| Validation | 10,676 | 349 | **10,327** |
| Test | 10,676 | 536 | **10,140** |


---

## Methodology

### Problem Formulation

Given two aligned character sequences from Whisper (`x⁽¹⁾`) and MMS (`x⁽²⁾`):

```
x⁽¹⁾ = [x₁⁽¹⁾, x₂⁽¹⁾, ..., xₙ⁽¹⁾]      (Whisper hypothesis)
x⁽²⁾ = [x₁⁽²⁾, x₂⁽²⁾, ..., xₘ⁽²⁾]      (MMS hypothesis)
y    = [y₁, y₂, ..., y_K]              (ground truth)
```

The model learns:

$$\hat{y} = \arg\max_{y'} p(y' \mid x^{(1)}, x^{(2)})$$

### Architecture

```
┌──────────────┐         ┌──────────────┐
│   Whisper    │         │     MMS      │
│  character   │         │  character   │
│   sequence   │         │   sequence   │
└──────┬───────┘         └──────┬───────┘
       │                        │
       └───────┬────────────────┘
               │
        ┌──────▼───────┐
        │  Levenshtein │
        │  Alignment   │
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │ Concatenate  │
        │   + Embed    │
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │  BiLSTM      │
        │  Encoder     │
        │ (shared)     │
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │    Gated     │
        │    Fusion    │
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │  Attention   │
        │  Decoder     │
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │  Corrected   │
        │    text      │
        └──────────────┘
```

### Gated Fusion (Reference Paper Formula)

The original paper uses a **Residual Modulator**:

$$g_{12,t} = \sigma(W_{12}[v_t^{(1)}; v_t^{(2)}])$$
$$\hat{h}_t^{(1)} = v_t^{(1)} + g_{12,t} \odot \tanh(v_t^{(2)})$$
$$\hat{h}_t^{(2)} = v_t^{(2)} + g_{21,t} \odot \tanh(v_t^{(1)})$$

We also experimented with a **Convex Combination** variant with Layer Normalization:

$$v_t^{(i)} = \mathrm{LN}(W_i h_t + b_i)$$
$$\hat{h}_t^{(1)} = (1 - g_{12,t}) \odot v_t^{(1)} + g_{12,t} \odot v_t^{(2)}$$

### Our Extensions

**1. Gate Variance Regularization**

Encourages the gate to be dynamic over time:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} - \lambda \cdot (\mathrm{Var}(g_{12}) + \mathrm{Var}(g_{21}))$$

**2. Scaled Sigmoid**

Keeps the gate in a tighter range `[0.1, 0.9]` to prevent saturation:

$$g_{12} = 0.8 \cdot \sigma(W_{12}[v_1; v_2]) + 0.1$$

**3. Sensory Dropout**

Random masking of one view to prevent over-reliance on a single view.

### Training Configuration

| Parameter | Value |
|---|---|
| Embedding dimension | 128 |
| Encoder hidden size | 128 |
| Decoder hidden size | 256 |
| Number of layers | 2 (encoder), 2 (decoder) |
| Dropout | 0.5 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 64 |
| Gradient clipping | 1.0 |
| Teacher forcing | Exponential or sigmoid or linear decay |
| Early stopping patience | 20 epochs |
| Vocabulary size | 52 characters or 1280 subwords |

---


### Comparison with Reference Paper (Rajasthani)

| Metric | Rajasthani (paper) | Persian (ours) |
|---|---|---|
| Initial Whisper CER | 16.60% | 9.80% |
| Initial MMS CER | 15.73% | **7.02%** |
| Best fusion CER | **7.86%** | 7.26% |
| CER reduction vs Whisper | 52.65% | 25.93% |
| CER reduction vs MMS | 50.02% | **−3.42%** (worse) |
| Gate-position correlation | Not reported | **+0.77** |
| Partial gate-content correlation | Not reported | **−0.05** |

The Rajasthani study reports a **53% CER reduction**, while our Persian study shows **no improvement over the best single view**.

---

## Deep Analysis

### Why Persian Differs from Rajasthani

**Devanagari errors are positional:**
- Vowel matras, aspiration markers, word-boundary merges
- Follow learnable rules from character position
- A small model can learn these patterns

**Persian errors are lexical:**
- Homophone confusion: ذ/ز/ض/ظ, س/ص/ث, ت/ط, ه/ح
- Distinguishing these requires **memorizing word spellings**, not learning positional rules
- For instance: «حضرت» vs. «حذرت» — a purely orthographic, not phonetic, distinction


### Lessons Learned

> **"Success of an architecture in one language is not a guarantee of success in another."**

The multi-view fusion method that achieved **53% CER reduction** on Rajasthani shows **no improvement** on Persian. The difference is **structural, not algorithmic**.

This has direct implications for:
- **Multilingual ASR research** — don't assume method transfer across script families.
- **System designers** — language-specific error analysis must precede method selection.


---

## Contact

For questions, collaboration, or discussion:

- **Zahra Tavakoli** — zahratavakoli763@gmail.com

