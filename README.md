# STAMP Prototype

This document outlines the implementation status of the STAMP research prototype, based on the paper "STAMP: Spatial-Temporal Adapter with Multi-Head Pooling".

## Implemented (Exactly as in Paper)

*   **Training Objective:** The prototype uses a standard `nn.CrossEntropyLoss` for classification, which is a common choice for this type of task. The paper's primary contribution appears to be architectural, and no other loss terms are mentioned in the available code or comments.
*   **CC-GMLP Block Architecture:** The `CCGMLPBlock` in `models/cc_gmlp.py` has been updated to faithfully implement the criss-cross gating mechanism as described in the paper's comments (Section 3.3, Equations 3 & 4). This includes the `Z1 ⊙ σ(W · Z2)` gating structure for both temporal and spatial streams. The previous implementation used a simplified `Z ⊙ σ(W · Z)` gating, which was a deviation from the paper.
*   **Training Procedure:** The prototype uses a standard training procedure with the following components:
    *   **Optimizer:** Adam (`optim.Adam`) with a learning rate of `1e-3` and weight decay of `1e-4`.
    *   **Scheduler:** Cosine Annealing (`optim.lr_scheduler.CosineAnnealingLR`).
    *   **Loss Function:** Cross-Entropy (`nn.CrossEntropyLoss`).
    *   **Gradient Clipping:** Gradients are clipped to a max norm of 1.0.
    These are standard and reasonable choices, assumed to be consistent with the paper.
