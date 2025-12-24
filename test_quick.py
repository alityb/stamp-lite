"""
Quick sanity check with synthetic data - tests if model can learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from models.adapter import STAMPAdapter

def test_quick():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Synthetic data - 32 samples, 22 channels, 1000 timepoints, 4 classes
    torch.manual_seed(42)
    data = torch.randn(32, 22, 1000).to(device)
    # Create clear patterns: class 0 = high mean, class 1 = low mean, etc.
    target = torch.tensor([0]*8 + [1]*8 + [2]*8 + [3]*8).to(device)

    # Make patterns distinguishable
    for i in range(32):
        if target[i] == 0:
            data[i] += 1.0
        elif target[i] == 1:
            data[i] -= 1.0
        elif target[i] == 2:
            data[i, :, :500] += 1.0  # First half high
        else:
            data[i, :, 500:] += 1.0  # Second half high

    print(f"Data shape: {data.shape}")
    print(f"Target: {target}\n")

    # Model
    model = STAMPAdapter(
        S=22, K=100, T=10, num_classes=4,
        pe_config="all",
        mixer_type="gmlp",
        pool_type="mean"
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Check TSFM is trainable
    tsfm_params = sum(p.numel() for p in model.tsfm.parameters() if p.requires_grad)
    print(f"TSFM trainable params: {tsfm_params:,}")

    if tsfm_params == 0:
        print("WARNING: TSFM is frozen!")
        return False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\nTraining on single batch (should reach 100% in ~50-100 steps):\n")

    for step in range(150):
        model.train()
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        pred = output.argmax(1)
        correct = pred.eq(target).sum().item()
        acc = 100. * correct / target.size(0)

        if step % 10 == 0:
            print(f"Step {step:3d}: Loss={loss.item():.4f}, Acc={acc:5.1f}%, GradNorm={grad_norm:.3f}")

        if acc == 100.0:
            print(f"\n✓ SUCCESS! Perfect accuracy at step {step}")
            print(f"  Model can learn - fixes are working!")
            return True

    print(f"\n✗ Did not reach 100% accuracy after 150 steps")
    print(f"  Final accuracy: {acc:.1f}%")
    if acc > 50:
        print("  Model is learning (>50%), but may need tuning")
        return True
    else:
        print("  Model is not learning effectively")
        return False

if __name__ == "__main__":
    success = test_quick()
    exit(0 if success else 1)
