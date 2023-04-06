import torch
import torchvision
from dataset import CaravanDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="Checkpoints/checkpoint.pth.tar"):
    print("==> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(
        train_dir,
        train_mask_dir,
        val_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory=True
):
    train_ds = CaravanDataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    val_ds = CaravanDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def check_accuracy(loader, model, summary_writer, step, loss_fn, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    loss = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y = y.float().unsqueeze(1).to(device=device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            loss += loss_fn(y,preds).item()
    summary_writer.add_scalars("Acc", {"val":num_correct/num_pixels}, step)
    summary_writer.add_scalars("Dice", {"val":dice_score/len(loader)}, step)
    summary_writer.add_scalars("Loss", {"val":loss/len(loader)}, step)
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}, dice score: {dice_score/len(loader):.2f}")
    model.train()

def save_preds_as_imgs( loader, model, folder="predictions", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/mask_{idx}.png")

    model.tarin()