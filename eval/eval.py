import torch
import numpy as np
from sklearn.metrics import average_precision_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def validate(net, val_loader, device, ignore_classes=[0, 2]):
    net.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for ((features, mask), labels) in val_loader:
            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = net(features, mask)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.cpu().numpy()
            targets = labels.cpu().numpy()

            N = preds.shape[0] * preds.shape[2] * preds.shape[3]
            preds = preds.transpose(0, 2, 3, 1).reshape(N, -1)
            targets = targets.reshape(-1)

            # Filter out ignored classes
            valid_idx = ~np.isin(targets, ignore_classes)
            preds = preds[valid_idx]
            targets = targets[valid_idx]

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # One-hot encoding for valid classes
    one_hot_targets = np.eye(net.num_classes)[all_targets]

    # Compute average precision score (AUC PR)
    average_precision = average_precision_score(
        one_hot_targets, all_preds, average='macro')

    print(f'Validation Average Precision (AUC PR): {average_precision:.4f}')
    return average_precision


def test(net, test_loader, device, ignore_classes=[0, 2]):
    """
    Test the model, ignoring specific classes in metrics.

    Args:
        net (torch.nn.Module): The model to test.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run the test on.
        ignore_classes (list): List of class indices to ignore in evaluation.

    Returns:
        float: Average Precision (AUC PR) excluding ignored classes.
    """
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
            preds = preds.transpose(0, 2, 3, 1).reshape(N, -1)
            targets = targets.reshape(-1)

            # Filter out ignored classes
            valid_idx = ~np.isin(targets, ignore_classes)  # Exclude ignored classes
            preds = preds[valid_idx]
            targets = targets[valid_idx]

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # One-hot encoding for valid classes
    one_hot_targets = np.eye(net.num_classes)[all_targets]

    # Compute average precision score (AUC PR)
    average_precision = average_precision_score(
        one_hot_targets, all_preds, average='macro'
    )

    print(f'Test Average Precision (AUC PR, excluding classes {ignore_classes}): {average_precision:.4f}')
    return average_precision
