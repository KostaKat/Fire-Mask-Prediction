import torch
import numpy as np
from sklearn.metrics import average_precision_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def validate(net, val_loader, device):
    net.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for ((features, mask), labels) in val_loader:
            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = net(features, mask)
            # Assuming outputs are logits
            probs = torch.softmax(outputs, dim=1)
            preds = probs.cpu().numpy()  # shape (batch_size, num_classes, H, W)
            targets = labels.cpu().numpy()  # shape (batch_size, H, W)

            # Reshape to (N, num_classes)
            N = preds.shape[0] * preds.shape[2] * preds.shape[3]
            preds = preds.transpose(0,2,3,1).reshape(N, -1)
            targets = targets.reshape(-1)

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute average precision score (AUC PR)
    average_precision = average_precision_score(
        np.eye(net.num_classes)[all_targets], all_preds, average='macro')

    print(f'Validation Average Precision (AUC PR): {average_precision:.4f}')

    return average_precision

def test(net, test_loader, device):
    net.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for ((features, mask), labels) in test_loader:
            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = net(features, mask)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.cpu().numpy()
            targets = labels.cpu().numpy()

            N = preds.shape[0] * preds.shape[2] * preds.shape[3]
            preds = preds.transpose(0,2,3,1).reshape(N, -1)
            targets = targets.reshape(-1)

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    average_precision = average_precision_score(
        np.eye(net.num_classes)[all_targets], all_preds, average='macro')

    print(f'Test Average Precision (AUC PR): {average_precision:.4f}')

    return average_precision
