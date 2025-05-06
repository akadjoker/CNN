import torch
from torch.utils.data import Dataset
import numpy as np

class LaneDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Converter para tensor
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        # Ajustar dimensões: (C, H, W) para imagem
        if image.ndim == 3:
            image = image.permute(2, 0, 1)  # HWC → CHW
        else:
            image = image.unsqueeze(0)  # grayscale

        # Ajustar dimensões: (C, H, W) para label
        if label.ndim == 3:
            label = label.permute(2, 0, 1)  # HWC → CHW
        else:
            label = label.unsqueeze(0)

        return image, label


def train_test_split(images, labels, val_ratio=0.2):
    num_samples = len(images)
    split_idx = int(num_samples * (1 - val_ratio))
    X_train = images[:split_idx]
    X_val = images[split_idx:]
    y_train = labels[:split_idx]
    y_val = labels[split_idx:]
    return X_train, X_val, y_train, y_val

