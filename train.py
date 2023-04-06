import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from dataset import CaravanDataset
import torch._dynamo as td
from torch.utils.tensorboard import SummaryWriter
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_preds_as_imgs
)
# from torch.profiler import profile, record_function, ProfilerActivity

        # with torch.cuda.amp.autocast():        #mixed-precision fp16
        #     # predictions = model(data)
        #     # loss = loss_fn(predictions, targets)
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
# Hyperparameters
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 500
NUM_WORKERS = 1
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train"
TRAIN_MASK_DIR = "data/train_masks"
VAL_IMG_DIR = "data/train"
VAL_MASK_DIR = "data/train_masks"
STEP = 0
COMPILE_MODEL = True
# train_loader, val_loader, model, optimizer, loss_fn, scaler, 10, summary_writer, step
def train_fn(loader, val_loader, model, optimizer, scheduler, loss_fn, scaler, log_window_size, summary_writer):
    loop = tqdm(loader)
    accumulated_train_loss = 0
    accumulated_train_accuracy = 0
    accumulated_train_dice_score = 0
    global STEP
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)                               #images
        targets = targets.float().unsqueeze(1).to(device=DEVICE)    #masks

        with torch.cuda.amp.autocast():        #mixed-precision fp16
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # metrics calc
        predictions = (predictions > 0.5).float()
        num_correct = (predictions == targets).sum()
        num_pixels = torch.numel(predictions)
        accuracy = num_correct/num_pixels*100
        dice_score = (2 * (predictions * targets).sum()) / ((predictions + targets).sum() + 1e-8)
        
        #accumulating metrics
        accumulated_train_accuracy += accuracy
        accumulated_train_dice_score += dice_score
        accumulated_train_loss += loss.item()
        if STEP % log_window_size == log_window_size - 1:
            count_window_datapoints  = (log_window_size - 1) * BATCH_SIZE + len(data)
            summary_writer.add_scalars("Loss", {"train":accumulated_train_loss/count_window_datapoints}, STEP)
            summary_writer.add_scalars("Acc", {"train":accumulated_train_accuracy/count_window_datapoints}, STEP)
            summary_writer.add_scalars("Dice", {"train":accumulated_train_dice_score/count_window_datapoints}, STEP)
            scheduler.step(accumulated_train_loss/count_window_datapoints)
            accumulated_train_accuracy = 0
            accumulated_train_dice_score = 0
            accumulated_train_loss = 0
            
        if (len(loader)-1 == batch_idx) and (not COMPILE_MODEL):
            summary_writer.add_graph(model, data)

        #update loop
        loop.set_postfix(loss=loss.item())
        STEP += 1
    check_accuracy(val_loader, model, summary_writer, STEP, loss_fn, device=DEVICE)  


def run_model(model_name):
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    summary_writer = SummaryWriter(f"runs/{model_name}")

    model = UNET(input_channels=3, output_channels=1).to(DEVICE)
    if COMPILE_MODEL:
        compiled_model = torch.compile(model)
    else:
        compiled_model = model
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(compiled_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=150, verbose=False)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, val_loader, compiled_model, optimizer, scheduler, loss_fn, scaler, 10, summary_writer)
        checkpoint = {
            "state_dict": compiled_model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        save_checkpoint(checkpoint, f"Checkpoints/{model_name}.pth.tar")
        summary_writer.flush()
        
    summary_writer.close()
        # save__predictions_as_imgs(val_loader, model)

def print_check():
    print(COMPILE_MODEL)

def main():
    global COMPILE_MODEL
    # run_model(f"compiled_model_1")
    # run_model(f"compiled_model_2")
    COMPILE_MODEL = False
    if COMPILE_MODEL == False:
        run_model(f"no_compiled_model_1")
        run_model(f"no_compiled_model_2")
if __name__ == "__main__":
    main()