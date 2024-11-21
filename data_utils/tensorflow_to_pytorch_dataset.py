import torch
from torch.utils.data import Dataset
def split_inputs(inputs, labels):
    # Split inputs into features and mask
    features = inputs[..., :-1]  # All channels except the last
    mask = inputs[..., -1:]      # The last channel, keeping dimensions
    return (features, mask), labels

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TFToTorchDataset(Dataset):
    def __init__(self, tf_dataset, resize_size=(224, 224)):
        """
        Args:
            tf_dataset: TensorFlow dataset.
            resize_size: Tuple for resizing inputs and labels (default 224x224).
        """
        self.tf_dataset = list(tf_dataset.unbatch().as_numpy_iterator())  # Convert to NumPy-compatible list
        self.resize_features = transforms.Compose([
            transforms.ToTensor(),  # Converts to [C, H, W] tensor
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR)
        ])
        self.resize_mask = transforms.Compose([
            transforms.ToTensor(),  # Converts to [C, H, W] tensor
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.NEAREST)
        ])

    def __len__(self):
        return len(self.tf_dataset)

    def __getitem__(self, idx):
        # Get the data
        data = self.tf_dataset[idx]

        # Unpack (features, mask), labels
        try:
            (features, mask), labels = data
        except ValueError as e:
            print(f"Error unpacking data at index {idx}: {e}")
            print("Data:", data)
            raise

        # Convert to PyTorch tensors
        features = self.resize_features(features).float()  # Resized to [C, H, W]
        mask = self.resize_mask(mask).long()  # Resized to [H, W]
        labels = torch.tensor(labels, dtype=torch.long)  # Labels don't need resizing if they are scalar

        return (features, mask), labels

