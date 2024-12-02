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


from sklearn.metrics import precision_score, recall_score, average_precision_score, confusion_matrix
import numpy as np

def test(net, test_loader, device, ignore_classes=[2]):
    """
    Test the model, calculating metrics for all classes and overall, while optionally ignoring specific classes.

    Args:
        net (torch.nn.Module): The model to test.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run the test on.
        ignore_classes (list): List of class indices to ignore in global evaluation.

    Returns:
        dict: Metrics including class-wise and overall Average Precision, Mean IoU, precision, and recall.
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

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # One-hot encoding for valid classes
    one_hot_targets = np.eye(net.num_classes)[all_targets]

    # Compute Average Precision (AUC PR) for all classes
    class_wise_ap = []
    for c in range(net.num_classes):
        class_indices = all_targets == c
        if np.any(class_indices):  # Only compute for classes present in the data
            ap = average_precision_score(one_hot_targets[class_indices, c], all_preds[class_indices, c])
            class_wise_ap.append(ap)
        else:
            class_wise_ap.append(np.nan)

    average_precision = np.nanmean([ap for i, ap in enumerate(class_wise_ap) if i not in (ignore_classes or [])])

    # Get predicted classes
    pred_classes = np.argmax(all_preds, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, pred_classes, labels=np.arange(net.num_classes))

    # Compute IoU per class
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = intersection / np.maximum(union, 1e-6)  # Avoid division by zero

    # Mean IoU excluding ignored classes
    mean_iou = np.nanmean([iou[c] for c in range(len(iou)) if c not in (ignore_classes or [])])

    # Class-wise precision and recall
    class_wise_precision = []
    class_wise_recall = []
    for c in range(net.num_classes):
        if cm[c].sum() > 0:  # Only compute for non-empty classes
            precision = cm[c, c] / np.maximum(cm[:, c].sum(), 1e-6)
            recall = cm[c, c] / np.maximum(cm[c, :].sum(), 1e-6)
            class_wise_precision.append(precision)
            class_wise_recall.append(recall)
        else:
            class_wise_precision.append(np.nan)
            class_wise_recall.append(np.nan)

    # Overall precision and recall
    precision = np.nanmean([p for i, p in enumerate(class_wise_precision) if i not in (ignore_classes or [])])
    recall = np.nanmean([r for i, r in enumerate(class_wise_recall) if i not in (ignore_classes or [])])

    # Print metrics
    print(f'Test Metrics:')
    print(f' - Average Precision (AUC PR, excluding ignored classes): {average_precision:.4f}')
    print(f' - Mean IoU (excluding ignored classes): {mean_iou:.4f}')
    print(f' - Precision (excluding ignored classes): {precision:.4f}')
    print(f' - Recall (excluding ignored classes): {recall:.4f}')

    # Return detailed metrics
    return {
        'average_precision': average_precision,
        'mean_iou': mean_iou,
        'precision': precision,
        'recall': recall,
        'class_wise_ap': class_wise_ap,
        'class_wise_iou': iou.tolist(),
        'class_wise_precision': class_wise_precision,
        'class_wise_recall': class_wise_recall,
    }
