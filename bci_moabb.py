"""
BCI Competition IV-2a Loader using MOABB

MOABB handles all the GDF complexity for us!
Install: pip install moabb
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    MOABB_AVAILABLE = True
except ImportError:
    MOABB_AVAILABLE = False
    print("MOABB not installed. Install with: pip install moabb")


class BCIDatasetMOABB(Dataset):
    """BCI Competition IV-2a using MOABB"""
    
    def __init__(self, subjects, window_size=1000):
        if not MOABB_AVAILABLE:
            raise ImportError("Please install MOABB: pip install moabb")
        
        print("Loading BCI Competition IV-2a via MOABB...")
        print(f"Subjects: {subjects}")
        
        # --- Paper Assumptions ---
        # Enforce data assumptions from the paper
        dataset = BNCI2014_001()
        assert all(s in dataset.subject_list for s in subjects), f"Invalid subject(s) found in {subjects}. Valid subjects are {dataset.subject_list}"

        paradigm = MotorImagery(n_classes=4, resample=250)
        assert paradigm.n_classes == 4, "This prototype is designed for 4 classes."
        assert paradigm.resample == 250, "This prototype assumes a 250Hz resampling rate."

        # Get data
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
        
        # X: (n_trials, n_channels, n_samples)
        # y: (n_trials,) labels
        
        print(f"Loaded {len(X)} trials")
        print(f"Shape: {X.shape}")
        print(f"Classes: {np.unique(y)}")
        
        # Trim/pad to window_size
        if X.shape[2] > window_size:
            X = X[:, :, :window_size]
        elif X.shape[2] < window_size:
            pad = window_size - X.shape[2]
            X = np.pad(X, ((0,0), (0,0), (0,pad)), mode='edge')
        
        # Normalize per channel per trial
        for i in range(len(X)):
            for ch in range(X.shape[1]):
                X[i, ch] = (X[i, ch] - X[i, ch].mean()) / (X[i, ch].std() + 1e-8)
        
        # Convert labels to 0-indexed
        unique_labels = np.unique(y)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y])
        
        self.data = X.astype(np.float32)
        self.labels = y.astype(np.int64)
        
        print(f"Final shape: {self.data.shape}")
        print(f"Class distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), self.labels[idx]


def get_bci_loaders_moabb(train_subjects=None, val_subjects=None, 
                          batch_size=32, window_size=1000):
    """
    Create loaders using MOABB (easiest method)
    
    Args:
        train_subjects: List [1,2,3,4,5,6] or None for default
        val_subjects: List [7,8,9] or None for default
    """
    if train_subjects is None:
        train_subjects = [1, 2, 3, 4, 5, 6]
    if val_subjects is None:
        val_subjects = [7, 8, 9]
    
    train_dataset = BCIDatasetMOABB(train_subjects, window_size)
    val_dataset = BCIDatasetMOABB(val_subjects, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing MOABB loader...")
    
    if not MOABB_AVAILABLE:
        print("\n❌ MOABB not installed")
        print("Install with: pip install moabb")
        exit(1)
    
    train_loader, val_loader = get_bci_loaders_moabb(
        train_subjects=[1],  # Just one subject for quick test
        val_subjects=[2],
        batch_size=16
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    x, y = next(iter(train_loader))
    print(f"\nBatch shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {torch.bincount(y)}")
    
    print("\n✓ MOABB loader working!")
