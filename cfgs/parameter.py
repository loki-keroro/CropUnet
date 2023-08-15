import torch
TRAINED_MODEL  = 'logs/best_epoch_weights.pth'
PT_MODEL       = 'model_deployment/Crop_unet.pt'
DATA_SIZE      = (1, 3, 512, 512)
NUM_CLASSES    = 2
BACKBONE       = 'vgg'