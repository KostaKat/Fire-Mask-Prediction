import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import CrossEntropy2d
from tqdm import tqdm

def train_one_epoch(net, train_loader, optimizer, epoch, device):
    net.train()
    total_loss = 0

    # Wrap the train_loader with tqdm
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for batch_idx, ((features, mask), labels) in progress_bar:
        # Prepare inputs
        features = features.to(device)  # (batch, channels, height, width)
        mask = mask.to(device)
        labels = labels.to(device)  # Remove channel dimension from labels

        optimizer.zero_grad()
        outputs = net(features, mask)  # Outputs: (batch_size, num_classes, height, width)

        loss = CrossEntropy2d(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update tqdm description with current batch loss
        progress_bar.set_postfix({"Loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch}] Training Loss: {avg_loss:.4f}")

