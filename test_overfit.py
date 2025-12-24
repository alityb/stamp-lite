"""
Quick test to verify model can overfit on a single batch
This is a sanity check to ensure the model is capable of learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from models.adapter import STAMPAdapter
from bci_moabb import get_bci_loaders_moabb

def test_overfit():
    """Test if model can overfit on a single batch"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load one batch from the dataset
    print("Loading BCI data...")
    train_loader, _ = get_bci_loaders_moabb(
        train_subjects=[1],  # Just one subject
        val_subjects=[2],
        batch_size=32,
        window_size=1000
    )

    # Get a single batch
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    print(f"\nBatch shape: {data.shape}")
    print(f"Labels: {target[:10]}")
    print(f"Class distribution: {torch.bincount(target)}\n")

    # Create model
    S, T_raw = data.shape[1], data.shape[2]
    T, K = 10, 100

    model = STAMPAdapter(
        S=S, K=K, T=T, num_classes=4,
        pe_config="all",
        mixer_type="gmlp",
        pool_type="mean"
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Testing if model can overfit on single batch...")
    print("Target: 100% accuracy (should happen in ~50-100 steps)\n")

    for step in range(200):
        model.train()
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Calculate accuracy
        pred = output.argmax(1)
        correct = pred.eq(target).sum().item()
        acc = 100. * correct / target.size(0)

        if step % 10 == 0:
            print(f"Step {step:3d}: Loss={loss.item():.4f}, Acc={acc:5.1f}%, GradNorm={grad_norm:.3f}")

            # Check if we've achieved perfect accuracy
            if acc == 100.0:
                print(f"\n✓ SUCCESS! Achieved 100% accuracy at step {step}")
                print(f"  Final loss: {loss.item():.6f}")
                return True

    print(f"\n✗ WARNING: Did not achieve 100% accuracy after 200 steps")
    print(f"  Final accuracy: {acc:.1f}%")
    print(f"  Final loss: {loss.item():.4f}")
    print("  This suggests there may still be issues with the model")
    return False

if __name__ == "__main__":
    test_overfit()
