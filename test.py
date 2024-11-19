from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torch
config_vit = CONFIGS_ViT_seg['R50-ViT-B_16_C12-1']
config_vit.n_classes = 2
config_vit.n_skip = 3
config_vit.patches.grid = (4, 4)
net = ViT_seg(config_vit, img_size=64, num_classes=2).cuda()
print(net)
# After initializing the model
print("Checking input channels for encoders:")

# Access the hybrid model
hybrid_model = net.transformer.embeddings.hybrid_model

# Check the input channels for the first encoder (root)
print("Encoder 1 input channels (root):", hybrid_model.root[0].in_channels)

# Check the input channels for the second encoder (rootd)
print("Encoder 2 input channels (rootd):", hybrid_model.rootd[0].in_channels)
