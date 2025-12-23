import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SpatioTemporalDataset(Dataset):
    """
    Dataset where classes have spatial AND temporal structure
    (to test if PE helps capture this)
    """
    def __init__(self, n_samples, S, T_raw, num_classes=4):
        self.data = []
        self.labels = []
        
        for _ in range(n_samples):
            cls = np.random.randint(num_classes)
            signal = self._generate_signal(cls, S, T_raw)
            self.data.append(signal)
            self.labels.append(cls)
        
        self.data = torch.FloatTensor(np.array(self.data))
        self.labels = torch.LongTensor(self.labels)
    
    def _generate_signal(self, cls, S, T_raw):
        t = np.linspace(0, 10, T_raw)
        signal = np.zeros((S, T_raw))
        
        for s in range(S):
            if cls == 0:
                # Low freq, varies by spatial position
                freq = 1 + 0.1 * s / S
                signal[s] = np.sin(2 * np.pi * freq * t)
            elif cls == 1:
                # High freq, varies by spatial position
                freq = 5 + 0.5 * s / S
                signal[s] = np.sin(2 * np.pi * freq * t)
            elif cls == 2:
                # Spatial gradient
                signal[s] = (s / S) * np.sin(2 * np.pi * 2 * t)
            else:
                # Temporal pattern that varies by space
                signal[s] = np.sin(2 * np.pi * (s / S + 1) * t)
            
            signal[s] += 0.1 * np.random.randn(T_raw)
            signal[s] = (signal[s] - signal[s].mean()) / (signal[s].std() + 1e-8)
        
        return signal
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_loaders(S=19, T=10, K=100, batch_size=32):
    T_raw = T * K
    train_ds = SpatioTemporalDataset(800, S, T_raw)
    val_ds = SpatioTemporalDataset(200, S, T_raw)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
