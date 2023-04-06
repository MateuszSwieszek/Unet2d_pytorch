import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch
import torch.nn as nn
from model import UNET
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_preds_as_imgs
)

class Model_Runner():
    def __init__(self,
                 loss_fn, 
                 summary_writer, 
                 model_name, 
                 device, 
                 batch_size, 
                 num_epochs, 
                 num_workers, 
                 image_height, 
                 image_width, 
                 train_img_dir, 
                 train_mask_dir, 
                 val_img_dir, 
                 val_mask_dir,
                 lr,
                 log_window_size=10, 
                 pin_memory=True,
                 load_model=False, 
                 compile_model=True):
        
        self.batch_size                   = batch_size
        self.num_epochs                   = num_epochs
        self.num_workers                  = num_workers
        self.image_height                 = image_height
        self.image_width                  = image_width
        self.train_img_dir                = train_img_dir
        self.train_mask_dir               = train_mask_dir
        self.val_img_dir                  = val_img_dir
        self.val_mask_dir                 = val_mask_dir
        self.pin_memory                   = pin_memory
        self.load_model                   = load_model
        self.compile_model                = compile_model
        self.learning_rate                = lr
        self.loss_fn                      = loss_fn  
        self.summary_writer               = summary_writer 
        self.model_name                   = model_name
        self.device                       = device
        self.accumulated_train_loss       = 0
        self.accumulated_train_accuracy   = 0
        self.accumulated_train_dice_score = 0
        self.step                         = 0
        self.scaler                       = torch.cuda.amp.GradScaler()
        self.log_window_size              = log_window_size
        self.optimizer                    = None    
        self.train_loader                 = None
        self.val_loader                   = None
        self.scheduler                    = None

    def init_loaders(self):

        train_transform = A.Compose(
            [
                A.Resize(height=self.image_height, width=self.image_width),
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
                A.Resize(height=self.image_height, width=self.image_width),
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
        
        self.train_loader, self.val_loader = get_loaders(
            self.train_img_dir,
            self.train_mask_dir,
            self.val_img_dir,
            self.val_mask_dir,
            self.batch_size,
            train_transform,
            val_transform,
            self.num_workers,
            self.pin_memory
        )

    def train_fn(self, model):
        loop = tqdm(self.train_loader)
        
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=self.device)                               #images
            targets = targets.float().unsqueeze(1).to(device=self.device)    #masks

            with torch.cuda.amp.autocast():        #mixed-precision fp16
                predictions = model(data)
                loss = self.loss_fn(predictions, targets)

            #backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # metrics calc
            predictions = (predictions > 0.5).float()
            num_correct = (predictions == targets).sum()
            num_pixels = torch.numel(predictions)
            accuracy = num_correct/num_pixels*100
            dice_score = (2 * (predictions * targets).sum()) / ((predictions + targets).sum() + 1e-8)
            
            #accumulating metrics
            self.accumulated_train_accuracy += accuracy
            self.accumulated_train_dice_score += dice_score
            self.accumulated_train_loss += loss.item()
            if self.step % self.log_window_size == self.log_window_size - 1:
                count_window_datapoints  = (self.log_window_size - 1) * self.batch_size + len(data)
                self.summary_writer.add_scalars("Loss", {"train":self.accumulated_train_loss/count_window_datapoints}, self.step)
                self.summary_writer.add_scalars("Acc", {"train":self.accumulated_train_accuracy/count_window_datapoints}, self.step)
                self.summary_writer.add_scalars("Dice", {"train":self.accumulated_train_dice_score/count_window_datapoints}, self.step)
                self.scheduler.step(self.accumulated_train_loss/count_window_datapoints)
                self.accumulated_train_accuracy = 0
                self.accumulated_train_dice_score = 0
                self.accumulated_train_loss = 0
                
            if (len(self.train_loader)-1 == batch_idx) and (not self.compile_model):
                self.summary_writer.add_graph(model, data)

            #update loop
            loop.set_postfix(loss=loss.item())
            self.step += 1
        check_accuracy(self.val_loader, model, self.summary_writer, self.step, self.loss_fn, device=self.device) 

    def run_model(self, model):
        
        model.to(self.device)
        if self.compile_model:
            model = torch.compile(model)

        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=150, verbose=False)

        for epoch in range(self.num_epochs):
            self.train_fn(model)
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }

            save_checkpoint(checkpoint, f"Checkpoints/{self.model_name}.pth.tar")
            self.summary_writer.flush()
            
        self.summary_writer.close()
            # save__predictions_as_imgs(val_loader, model)

def main():
    model_name = "compiled_model_1"
    summary_writer = SummaryWriter(f"runs/{model_name}")
    loss_fn = nn.BCEWithLogitsLoss()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    BATCH_SIZE = 32
    NUM_EPOCHS = 500
    NUM_WORKERS = 1
    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 240
    LEARNING_RATE = 1e-3
    TRAIN_IMG_DIR = "data/train"
    TRAIN_MASK_DIR = "data/train_masks"
    VAL_IMG_DIR = "data/train"
    VAL_MASK_DIR = "data/train_masks"
    model = UNET(input_channels=3, output_channels=1).to(device)
    runner = Model_Runner(loss_fn, summary_writer, model_name, device, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS,IMAGE_HEIGHT,IMAGE_WIDTH,
                          TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, LEARNING_RATE,10)
    runner.init_loaders()
    runner.run_model(model)


if __name__ == "__main__":
    main()