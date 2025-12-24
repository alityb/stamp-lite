"""
Fast Hypothesis Test: PE is critical for EEG + CC-GMLP

Hypothesis:
- TSFM embeddings are content-rich but location-agnostic
- EEG tasks are location-sensitive (channel positions matter)
- Without PE: CC-GMLP sees tokens as exchangeable → poor performance
- With PE: CC-GMLP gets coordinate system → big performance jump
- Transformer: needs more data to learn spatial ≠ temporal structure

Test with MINIMAL data to confirm edge cases without waiting hours.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from models.adapter import STAMPAdapter
from tqdm import tqdm
import argparse

try:
    from bci_moabb import get_bci_loaders_moabb as get_bci_loaders
    BCI_AVAILABLE = True
except ImportError:
    BCI_AVAILABLE = False
    print("Warning: MOABB not available")


def train_epoch(model, loader, criterion, optimizer, device):
    """Single training epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data, target in tqdm(loader, desc="Train", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += output.argmax(1).eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validation"""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for data, target in tqdm(loader, desc="Val", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            correct += output.argmax(1).eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(loader), 100. * correct / total


def quick_test(pe_config, mixer_type, train_loader, val_loader,
               S, K, T, num_classes, epochs, device):
    """Run one quick experiment"""

    print(f"\n{'='*60}")
    print(f"Config: PE={pe_config}, Mixer={mixer_type}")
    print(f"{'='*60}")

    model = STAMPAdapter(
        S=S, K=K, T=T, num_classes=num_classes,
        pe_config=pe_config,
        mixer_type=mixer_type,
        pool_type="mean"
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {trainable_params:,} trainable / {total_params:,} total")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0
    final_train_acc = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        best_val_acc = max(best_val_acc, val_acc)
        final_train_acc = train_acc

        print(f"Epoch {epoch:2d}/{epochs}: "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.1f}% | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.1f}% | "
              f"Best Val={best_val_acc:.1f}%")

    return {
        'pe_config': pe_config,
        'mixer_type': mixer_type,
        'best_val_acc': best_val_acc,
        'final_train_acc': final_train_acc,
        'final_val_acc': val_acc
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load MINIMAL dataset - just 1-2 subjects
    if BCI_AVAILABLE:
        print("\nLoading BCI data (MINIMAL: 1-2 subjects for fast testing)...")
        train_loader, val_loader = get_bci_loaders(
            train_subjects=[1, 2],  # Just 2 subjects instead of 6
            val_subjects=[3],       # Just 1 subject instead of 3
            batch_size=32,
            window_size=1000
        )
        x_sample, _ = next(iter(train_loader))
        S, T_raw = x_sample.shape[1], x_sample.shape[2]
        T, K = 10, 100
        num_classes = 4
        dataset_name = "BCI-IV-2a (2 subjects)"
    else:
        print("MOABB not available, install with: pip install moabb")
        return

    print(f"\nDataset: {dataset_name}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Channels: {S}, Tokens: {T}, Token length: {K}")
    print(f"Classes: {num_classes}")

    results = []

    # ======================================================================
    # HYPOTHESIS TEST: PE is critical for CC-GMLP on EEG
    # ======================================================================

    print("\n" + "="*60)
    print("HYPOTHESIS: PE enables spatial awareness for EEG")
    print("="*60)
    print("\nExpected results:")
    print("  1. CC-GMLP without PE: POOR (tokens are exchangeable)")
    print("  2. CC-GMLP with PE: GOOD (tokens get spatial coordinates)")
    print("  3. Transformer with PE: OK but needs more data")

    # Test 1: CC-GMLP WITHOUT PE (should struggle)
    result = quick_test(
        pe_config="none",
        mixer_type="gmlp",
        train_loader=train_loader,
        val_loader=val_loader,
        S=S, K=K, T=T, num_classes=num_classes,
        epochs=args.epochs,
        device=device
    )
    results.append(result)

    # Test 2: CC-GMLP WITH PE (should be much better)
    result = quick_test(
        pe_config="all",
        mixer_type="gmlp",
        train_loader=train_loader,
        val_loader=val_loader,
        S=S, K=K, T=T, num_classes=num_classes,
        epochs=args.epochs,
        device=device
    )
    results.append(result)

    # Test 3: Transformer WITH PE (for comparison)
    if args.compare_transformer:
        result = quick_test(
            pe_config="all",
            mixer_type="transformer",
            train_loader=train_loader,
            val_loader=val_loader,
            S=S, K=K, T=T, num_classes=num_classes,
            epochs=args.epochs,
            device=device
        )
        results.append(result)

    # ======================================================================
    # SUMMARY
    # ======================================================================

    print("\n" + "="*60)
    print("HYPOTHESIS TEST RESULTS")
    print("="*60)

    print("\nConfiguration                    | Best Val Acc | Final Train Acc")
    print("-" * 60)
    for r in results:
        config_str = f"{r['mixer_type'].upper():12s} + PE={r['pe_config']:20s}"
        print(f"{config_str} | {r['best_val_acc']:6.1f}%      | {r['final_train_acc']:6.1f}%")

    # Check hypothesis
    print("\n" + "="*60)
    print("HYPOTHESIS VALIDATION:")
    print("="*60)

    gmlp_no_pe = [r for r in results if r['mixer_type'] == 'gmlp' and r['pe_config'] == 'none'][0]
    gmlp_with_pe = [r for r in results if r['mixer_type'] == 'gmlp' and r['pe_config'] == 'all'][0]

    pe_gain = gmlp_with_pe['best_val_acc'] - gmlp_no_pe['best_val_acc']

    print(f"\n1. CC-GMLP without PE: {gmlp_no_pe['best_val_acc']:.1f}%")
    print(f"2. CC-GMLP with PE:    {gmlp_with_pe['best_val_acc']:.1f}%")
    print(f"\n   → PE gain: {pe_gain:+.1f}%")

    if pe_gain > 5.0:
        print(f"\n✓ HYPOTHESIS CONFIRMED!")
        print(f"  PE provides substantial gain ({pe_gain:.1f}%) for CC-GMLP on EEG.")
        print(f"  This validates that spatial coordinates are critical for")
        print(f"  location-sensitive tasks when using axis-aligned mixers.")
    elif pe_gain > 0:
        print(f"\n⚠ HYPOTHESIS PARTIALLY CONFIRMED")
        print(f"  PE helps ({pe_gain:.1f}%), but gain is smaller than expected.")
        print(f"  May need more epochs or data to see full effect.")
    else:
        print(f"\n✗ HYPOTHESIS NOT CONFIRMED")
        print(f"  PE does not help or hurts performance.")
        print(f"  Suggests other issues (check model, data, training).")

    if args.compare_transformer:
        transformer_pe = [r for r in results if r['mixer_type'] == 'transformer'][0]
        print(f"\n3. Transformer with PE: {transformer_pe['best_val_acc']:.1f}%")

        if gmlp_with_pe['best_val_acc'] > transformer_pe['best_val_acc']:
            gap = gmlp_with_pe['best_val_acc'] - transformer_pe['best_val_acc']
            print(f"\n✓ CC-GMLP beats Transformer by {gap:.1f}%")
            print(f"  Inductive bias (spatial ≠ temporal) helps under low data!")
        else:
            print(f"\n  Transformer competitive or better (may need more data for gap)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fast hypothesis test: PE critical for CC-GMLP on EEG'
    )
    parser.add_argument('--epochs', type=int, default=15,
                        help='Epochs per test (default: 15 for speed)')
    parser.add_argument('--compare-transformer', action='store_true',
                        help='Also test transformer (slower)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("FAST HYPOTHESIS TEST")
    print("="*60)
    print(f"\nSettings:")
    print(f"  - Epochs per test: {args.epochs}")
    print(f"  - Subjects: 2 train, 1 val (minimal dataset)")
    print(f"  - Compare transformer: {args.compare_transformer}")
    print(f"\nThis should run in ~5-15 minutes depending on hardware.")
    print("="*60)

    main(args)
