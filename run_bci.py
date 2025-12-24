"""
Run STAMP experiments on BCI Competition IV-2a dataset

Tests 3 hypotheses on REAL EEG data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from models.adapter import STAMPAdapter
from models.cc_gmlp import CCGMLP
from tqdm import tqdm
import json
import argparse
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from utils import set_seed

try:
    from bci_moabb import get_bci_loaders_moabb as get_bci_loaders
    BCI_AVAILABLE = True
except ImportError:
    BCI_AVAILABLE = False
    print("Warning: MOABB not available, will use synthetic data")

def train_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=1, print_every=10):
    model.train()
    total_loss, correct, total = 0, 0, 0
    batch_losses = []
    batch_accs = []

    use_amp = scaler is not None

    pbar = tqdm(loader, desc=f"Train E{epoch}", leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Mixed precision forward
        if use_amp:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()

            # WARNING: Not in paper. Experimental convenience only.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            # WARNING: Not in paper. Experimental convenience only.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Track metrics
        batch_loss = loss.item()
        batch_correct = output.argmax(1).eq(target).sum().item()
        batch_acc = 100. * batch_correct / target.size(0)

        total_loss += batch_loss
        correct += batch_correct
        total += target.size(0)
        batch_losses.append(batch_loss)
        batch_accs.append(batch_acc)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{batch_loss:.3f}',
            'acc': f'{batch_acc:.1f}%',
            'grad': f'{grad_norm:.2f}'
        })

        # Print detailed diagnostics
        if batch_idx % print_every == 0 and batch_idx > 0:
            avg_loss = np.mean(batch_losses[-print_every:])
            avg_acc = np.mean(batch_accs[-print_every:])
            print(f"  Batch {batch_idx}/{len(loader)}: Loss={avg_loss:.4f}, Acc={avg_acc:.1f}%, GradNorm={grad_norm:.3f}")

    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
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

def run_experiment(pe_config, mixer_type, pool_type,
                   train_loader, val_loader,
                   S, K, T, num_classes,
                   epochs=50, device='cuda', use_amp=True):
    """Run single experiment"""

    model = STAMPAdapter(
        S=S, K=K, T=T, num_classes=num_classes,
        pe_config=pe_config,
        mixer_type=mixer_type,
        pool_type=pool_type
    ).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Model: {model.get_config_str()}")
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    # WARNING: Not in paper. Experimental convenience only.
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # WARNING: Not in paper. Experimental convenience only.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Mixed precision training
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    if scaler:
        print("Using mixed precision training (AMP)")

    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, epoch=epoch, print_every=10
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        best_val_acc = max(best_val_acc, val_acc)

        # Always print epoch summary
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:2d}/{epochs}: Train Loss={train_loss:.4f} Acc={train_acc:.1f}% | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.1f}% | Best={best_val_acc:.1f}% | LR={lr:.6f}")

        # Log gate statistics if using CCGMLP
        if isinstance(model.mixer, CCGMLP):
            gate_stats = model.mixer.get_all_gate_stats()
            # Just show stats for first block for brevity
            t_mean = gate_stats['block_0']['temporal']['mean']
            s_mean = gate_stats['block_0']['spatial']['mean']
            print(f"  Gate Stats (Block 0): Temp Mean={t_mean:.3f}, Spatial Mean={s_mean:.3f}")

    return {
        'config': model.get_config_str(),
        'pe_config': pe_config,
        'mixer_type': mixer_type,
        'pool_type': pool_type,
        'best_val_acc': best_val_acc,
        'final_val_acc': val_acc,
        'history': history
    }

def run_all_experiments(use_bci=True, data_dir='./BCICIV_2a_gdf', epochs=50):
    """
    Test the hypotheses on real or synthetic data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load dataset
    if use_bci:
        if not BCI_AVAILABLE:
            print("MOABB not available. Install with: pip install moabb")
            print("Falling back to synthetic data...")
            use_bci = False
        else:
            print("Loading BCI Competition IV-2a dataset via MOABB...")
            try:
                train_loader, val_loader = get_bci_loaders(
                    train_subjects=[1, 2, 3, 4, 5, 6],
                    val_subjects=[7, 8, 9],
                    batch_size=32,
                    window_size=1000
                )
                # Get data shape from first batch
                x_sample, _ = next(iter(train_loader))
                S, T_raw = x_sample.shape[1], x_sample.shape[2]
                T, K = 10, 100  # 10 tokens of 100 samples each
                num_classes = 4
                dataset_name = "BCI-IV-2a"
            except Exception as e:
                print(f"Error loading BCI dataset: {e}")
                print("Falling back to synthetic data...")
                use_bci = False
    
    if not use_bci:
        print("Using synthetic dataset...")
        train_loader, val_loader = get_synthetic_loaders(S=19, T=10, K=100, batch_size=32)
        S, T, K = 19, 10, 100
        T_raw = T * K
        num_classes = 4
        dataset_name = "Synthetic"
    
    print(f"\nDataset: {dataset_name}")
    print(f"Channels (S): {S}")
    print(f"Time samples (T_raw): {T_raw}")
    print(f"Tokens (T): {T}, Token length (K): {K}")
    print(f"Classes: {num_classes}")
    
    results = []
    
    # Hypothesis 1: PE ablation
    print("\n" + "="*60)
    print("HYPOTHESIS 1: More PE = Better Performance")
    print("="*60)
    
    pe_configs = ["none", "spatial+temporal", "all"]  # Reduced for speed
    for pe in pe_configs:
        result = run_experiment(
            pe, "gmlp", "mean",
            train_loader, val_loader,
            S, K, T, num_classes,
            epochs=epochs, device=device
        )
        results.append(result)
    
    # Hypothesis 2: CC-GMLP vs Transformer
    print("\n" + "="*60)
    print("HYPOTHESIS 2: CC-GMLP > Transformer")
    print("="*60)
    
    for mixer in ["gmlp", "transformer"]:
        result = run_experiment(
            "all", mixer, "mean",
            train_loader, val_loader,
            S, K, T, num_classes,
            epochs=epochs, device=device
        )
        results.append(result)
    
    # Hypothesis 3: Pooling comparison
    print("\n" + "="*60)
    print("HYPOTHESIS 3: Pooling Choice Matters Less")
    print("="*60)
    
    for pool in ["mean", "attention"]:
        result = run_experiment(
            "all", "gmlp", pool,
            train_loader, val_loader,
            S, K, T, num_classes,
            epochs=epochs, device=device
        )
        results.append(result)
    
    # Save results
    output_file = f'experiment_results_{dataset_name.lower()}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print(f"EXPERIMENT SUMMARY ({dataset_name})")
    print("="*60)
    
    print("\n1. PE Ablation:")
    pe_results = [r for r in results if r['mixer_type'] == 'gmlp' and r['pool_type'] == 'mean']
    for r in pe_results:
        print(f"  PE={r['pe_config']:20s}: {r['best_val_acc']:.1f}%")
    
    print("\n2. Mixer Comparison:")
    mixer_results = [r for r in results if r['pe_config'] == 'all' and r['pool_type'] == 'mean']
    for r in mixer_results:
        print(f"  {r['mixer_type']:15s}: {r['best_val_acc']:.1f}%")
    
    print("\n3. Pooling Comparison:")
    pool_results = [r for r in results if r['pe_config'] == 'all' and r['mixer_type'] == 'gmlp']
    for r in pool_results:
        print(f"  {r['pool_type']:15s}: {r['best_val_acc']:.1f}%")
    
    print(f"\n💾 Results saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run STAMP experiments')
    parser.add_argument('--dataset', type=str, default='bci', choices=['bci', 'synthetic'],
                        help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./BCICIV_2a_gdf',
                        help='Path to BCI dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs per experiment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    
    run_all_experiments(
        use_bci=(args.dataset == 'bci'),
        data_dir=args.data_dir,
        epochs=args.epochs
    )
