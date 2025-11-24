import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

class MelodyDataset(Dataset):
    def __init__(self, npz_path: str):
        print(f"Loading dataset from {npz_path}...")
        try:
            with np.load(npz_path) as data:
                self.X = data['X']
                self.y_chroma = data['y_chroma']
                self.y_octave = data['y_octave']
                self.y_voicing = data['y_voicing']
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {npz_path}")
            exit()
        
        self.num_samples = self.X.shape[0]
        print(f"Loaded {self.num_samples} samples.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.X[idx] 
        y_chroma = self.y_chroma[idx]
        y_octave = self.y_octave[idx]
        y_voicing = self.y_voicing[idx]
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_chroma, dtype=torch.long),
            torch.tensor(y_octave, dtype=torch.long),
            torch.tensor(y_voicing, dtype=torch.long)
        )
