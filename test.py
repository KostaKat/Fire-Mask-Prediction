from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torch
import numpy as np
# Adjusted configurations
config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = 2
config_vit.n_skip = 3
config_vit.patches.grid = (4, 4)


# Initialize the model with img_size=256
net = ViT_seg(config_vit, img_size=64, num_classes=2).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path))

# Create dummy input data with size 256x256
dummy_input_1 = torch.randn(1, 12, 64, 64).cuda()
dummy_input_2 = torch.randn(1, 1, 64, 64).cuda()

# Perform a forward pass with the dummy data
try:
    output = net(dummy_input_1, dummy_input_2)
    print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"Error during forward pass: {e}")