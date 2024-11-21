import torch
from utils import CrossEntropy2d

def train_one_epoch(net, train_loader, optimizer, epoch, device):
    net.train()
    total_loss = 0
    for batch_idx, ((features, mask), labels) in enumerate(train_loader):
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(features, mask)

        loss = CrossEntropy2d(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch}] Training Loss: {avg_loss:.4f}')
