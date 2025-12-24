"""
Quick test to verify gate statistics are healthy (not saturated)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from models.adapter import STAMPAdapter

def test_gate_health():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Synthetic data
    torch.manual_seed(42)
    data = torch.randn(16, 22, 1000).to(device)
    target = torch.tensor([0]*4 + [1]*4 + [2]*4 + [3]*4).to(device)

    # Make patterns distinguishable
    for i in range(16):
        if target[i] == 0:
            data[i] += 1.0
        elif target[i] == 1:
            data[i] -= 1.0
        elif target[i] == 2:
            data[i, :, :500] += 1.0
        else:
            data[i, :, 500:] += 1.0

    # Model with CC-GMLP
    model = STAMPAdapter(
        S=22, K=100, T=10, num_classes=4,
        pe_config="all",
        mixer_type="gmlp",
        pool_type="mean"
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("="*70)
    print("GATE HEALTH MONITORING")
    print("="*70)
    print("\nHealthy gates should have:")
    print("  - Mean: 0.3-0.7 (not saturated at 0 or 1)")
    print("  - Std: > 0.05 (gates are learning, not all the same)")
    print("  - Min/Max: Not all 0s or all 1s")
    print("\n" + "="*70 + "\n")

    for step in range(30):
        model.train()
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(1)
        correct = pred.eq(target).sum().item()
        acc = 100. * correct / target.size(0)

        # Print gate statistics every 10 steps
        if step % 10 == 0:
            print(f"Step {step:3d}: Loss={loss.item():.4f}, Acc={acc:5.1f}%")

            # Get gate stats from first CC-GMLP block
            gate_stats = model.mixer.blocks[0].get_gate_stats()

            print("\n  Gate Statistics (Block 0):")
            print(f"    Temporal: mean={gate_stats['temporal']['mean']:.3f}, "
                  f"std={gate_stats['temporal']['std']:.3f}, "
                  f"min={gate_stats['temporal']['min']:.3f}, "
                  f"max={gate_stats['temporal']['max']:.3f}")
            print(f"    Spatial:  mean={gate_stats['spatial']['mean']:.3f}, "
                  f"std={gate_stats['spatial']['std']:.3f}, "
                  f"min={gate_stats['spatial']['min']:.3f}, "
                  f"max={gate_stats['spatial']['max']:.3f}")

            # Check for saturation
            t_mean = gate_stats['temporal']['mean']
            s_mean = gate_stats['spatial']['mean']

            if t_mean < 0.1 or t_mean > 0.9 or s_mean < 0.1 or s_mean > 0.9:
                print(f"\n  ⚠ WARNING: Gates may be saturating!")
            else:
                print(f"\n  ✓ Gates look healthy!")
            print()

    print("="*70)
    print("Gate monitoring complete!")
    print("="*70)

if __name__ == "__main__":
    test_gate_health()
