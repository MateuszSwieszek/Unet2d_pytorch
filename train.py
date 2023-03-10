import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
# from utils import (
#     load_checkpoint
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save__predictions_as_imgs
# )

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 1
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train"
TRAIN_MASK_DIR = "data/train_masks"