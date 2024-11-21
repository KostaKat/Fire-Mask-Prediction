import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import CrossEntropy2d
import torch

def train_one_epoch(net, train_loader, optimizer, epoch, device):
    net.train()
    total_loss = 0

    for batch_idx, ((features, mask), labels) in enumerate(train_loader):
        # Prepare inputs
        features = features.to(device)  # (batch, channels, height, width)
        mask = mask.to(device)
        labels = labels.to(device)  # Remove channel dimension from labels

        # Debug shapes and ranges
        print("Labels unique values:", torch.unique(labels))
        print("Features shape:", features.shape)
        print("Mask shape:", mask.shape)
        print("Labels shape:", labels.shape)

        optimizer.zero_grad()
        outputs = net(features, mask)  # Outputs: (batch_size, num_classes, height, width)
        print("Outputs shape:", outputs.shape)
     

        loss = CrossEntropy2d(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch}] Training Loss: {avg_loss:.4f}")
