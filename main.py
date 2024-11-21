import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from train.train import train_one_epoch
from eval.eval import validate, test
from data_utils.tensorflow_to_pytorch_dataset import TFToTorchDataset, split_inputs
from data_utils.parse_files import get_dataset
from tqdm import tqdm

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    NUM_CLASSES = args.num_classes
    BASE_LR = args.base_lr

    # Get the TensorFlow dataset
    dataset_train = get_dataset(
        args.data_pattern + 'train*',
        data_size=args.data_size,
        sample_size=args.sample_size,
        batch_size=BATCH_SIZE,
        num_in_channels=args.num_in_channels,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=True,
        center_crop=False
    )
    dataset_val = get_dataset(
        args.data_pattern + 'eval*',
        data_size=args.data_size,
        sample_size=args.sample_size,
        batch_size=BATCH_SIZE,
        num_in_channels=args.num_in_channels,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=True,
        center_crop=False
    )
    dataset_test = get_dataset(
        args.data_pattern + 'test*',
        data_size=args.data_size,
        sample_size=args.sample_size,
        batch_size=BATCH_SIZE,
        num_in_channels=args.num_in_channels,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=True,
        center_crop=False
    )
    dataset_train = dataset_train.map(split_inputs)
    dataset_val = dataset_val.map(split_inputs)
    dataset_test = dataset_test.map(split_inputs)
    dataset_train = TFToTorchDataset(dataset_train)
    dataset_val = TFToTorchDataset(dataset_val)
    dataset_test = TFToTorchDataset(dataset_test)

    # 
    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    config_vit.patches.grid = (4, 4)

    net = ViT_seg(config_vit, img_size=224, num_classes=NUM_CLASSES).to(device)
    net.load_from(weights=np.load(config_vit.pretrained_path))
    net.num_classes = NUM_CLASSES  # Add num_classes attribute for convenience

    # Initialize optimizer and scheduler
    optimizer = optim.SGD(net.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 35, 45], gamma=0.1)

    # Training loop
    best_auc = 0.0

    # Wrap the epoch loop with tqdm
    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Training Progress"):
        train_one_epoch(net, train_loader, optimizer, epoch, device)
        scheduler.step()

        # Validation
        val_auc = validate(net, val_loader, device)
        if val_auc > best_auc:
            best_auc = val_auc
            # Save the best model
            torch.save(net.state_dict(), f'best_model_epoch_{epoch}.pth')
        print(f'Epoch {epoch}, Validation AUC PR: {val_auc:.4f}, Best AUC PR: {best_auc:.4f}')

    # Test after training
    test_auc = test(net, test_loader, device)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--data_pattern', type=str, required=True, help='Data file pattern')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--base_lr', type=float, default=0.01, help='Base learning rate')
    parser.add_argument('--data_size', type=int, default=64, help='Data size')
    parser.add_argument('--sample_size', type=int, default=64, help='Sample size')
    parser.add_argument('--num_in_channels', type=int, default=12, help='Number of input channels')
    parser.add_argument('--img_size', type=int, default=64, help='Image size for the model')
    args = parser.parse_args()
    main(args)
