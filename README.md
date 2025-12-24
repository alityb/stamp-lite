# STAMP Prototype (Notes)

This repo is a small research prototype based on  
**“STAMP: Spatial-Temporal Adapter with Multi-Head Pooling” (ML4H 2025)**.

---

## Context

Time series foundation models (TSFMs) pretrained on large, diverse datasets show strong transfer performance, but prior to STAMP there wasn’t a clear comparison between **general TSFMs** and **EEG-specific foundation models** on EEG tasks.

STAMP shows that a frozen, general TSFM paired with the right adapter can achieve performance comparable to EEGFMs, suggesting the bottleneck is not the backbone but how structure is injected downstream.

---

## What STAMP does

Instead of training a large EEG-specific model, STAMP:
- freezes a pretrained TSFM
- extracts univariate embeddings
- explicitly models **spatial and temporal structure** on top

Naively mean-pooling TSFM embeddings performs near random. The adapter is what makes the representations usable.

---

## Tokens & structure

A token corresponds to a short univariate time-series segment processed by the TSFM.

EEG data is decomposed into a **spatiotemporal grid of tokens** (channels × time windows). Using this grid requires:
- **mixing**: modeling relationships between tokens
- **aggregation**: summarizing the grid for prediction

Most of STAMP is about doing these two steps efficiently.

---

## Core components

### Positional encodings
STAMP uses learnable:
- token-wise
- spatial (channel)
- temporal (time)

The paper shows that more positional encoding generally improves performance.

**Working hypothesis:** TSFM embeddings are content-rich but location-agnostic, while EEG tasks are location-sensitive. Positional encodings give the adapter an explicit coordinate system, so tokens are no longer treated as exchangeable.

---

### CC-GMLP (token mixing)
STAMP uses a criss-cross gated MLP instead of a standard transformer.

Transformers mix everything with everything, which ignores that EEG structure is axis-aligned (channels ≠ time). CC-GMLP hard-codes this distinction by mixing spatial and temporal dimensions separately using gating, which appears more parameter-efficient and data-efficient.

---

### Aggregation
Tokens are aggregated using mean pooling or attention-based pooling.

Empirically, aggregation choice matters less once token mixing is present, likely because cross-token relationships are already modeled upstream.

---

## Implementation notes

- TSFM backbone is frozen (stand-in encoder here)
- Adapter implements positional encodings + CC-GMLP blocks
- Standard cross-entropy training (focus is architectural, not optimization)
- This is a prototype for understanding and extension

---

## Open questions -- will revisit later

- Why does positional encoding help so much even when it increases parameters?
- Is CC-GMLP’s advantage mostly inductive bias or parameter efficiency?
- How much mixing is needed before aggregation choice stops mattering?
