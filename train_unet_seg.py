import os
import glob
import argparse
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
    Set random seeds to make results more reproducible across runs.
    Note: complete determinism on GPU may still require extra flags,
    but this already stabilizes most sources of randomness.
    """
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleUNet(nn.Module):
    """
    Small U-Net for binary segmentation.
    Input:  RGB image (3 channels)
    Output: 1 channel logits map (later converted to probability with sigmoid)
    """
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        
        """
            Standard U-Net conv block:
            Conv -> BN -> ReLU -> Conv -> BN -> ReLU
            """
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        #Encoder (downsampling path)
        self.enc1 = block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        #Bottleneck (deepest level)
        self.bott = block(base * 4, base * 8)
        
        #Decoder (upsampling path) + skip connections
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = block(base * 8, base * 4) #concat channels: up + skip

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = block(base * 2, base)
        
        # Final 1x1 conv to produce 1-channel logits.
        self.out = nn.Conv2d(base, out_ch, 1)
    
    """
        Forward pass with skip connections.
        We store encoder activations (e1,e2,e3) and concatenate them with
        decoder features after each upsampling step.
        """
    def forward(self, x):
        #Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bott(self.pool3(e3))   #Bottleneck
        
        #Decoder stage3
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1) #skip connection
        d3 = self.dec3(d3)
        
        #Decoder stage2
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        #Decoder stage1
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output logits (no sigmoid here because we use BCEWithLogitsLoss)
        return self.out(d1)


class RGBMaskDataset(Dataset):
    """
    Dataset class optimized for your training files:
    - Image: file_name.jpg
    - Mask:  file_name_mask.png
    """
    def __init__(self, images_dir, masks_dir, img_size=256, limit=None):
        self.img_size = img_size
        self.items = []
        
        # 1. Search for all .jpg images recursively in the nyu2_train folder
        img_paths = sorted(glob.glob(os.path.join(images_dir, "**", "*.jpg"), recursive=True))
        if limit:
            img_paths = img_paths[:limit]
        
        # 2. Pair each .jpg with its corresponding _mask.png
        # Convert masks_dir to absolute path to avoid ambiguity
        abs_masks_dir = os.path.abspath(masks_dir)

        for ip in img_paths:
            # Get filename without extension (e.g., "115")
            name_no_ext = os.path.splitext(os.path.basename(ip))[0]
            
            # Check for both "115.png" and "115_mask.png"
            possible_masks = [
                os.path.join(abs_masks_dir, name_no_ext + ".png"),
                os.path.join(abs_masks_dir, name_no_ext + "_mask.png")
            ]
            
            for mp in possible_masks:
                if os.path.exists(mp):
                    self.items.append((ip, mp))
                    break # Found it, move to next image
        
        if not self.items:
            raise RuntimeError(
                f"No paired (JPG, MASK) found in {images_dir}.\n"
                "Check if masks in your output folder have the suffix '_mask.png'."
            )

        print(f"[INFO] Successfully paired {len(self.items)} training samples.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ip, mp = self.items[idx]
        
        # Load image (BGR to RGB) and mask (Grayscale)
        img = cv2.imread(ip)
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize both to the U-Net input size
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Normalization and Tensor conversion (C, H, W)
        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        msk = (msk.astype(np.float32) / 255.0)[None, ...] # Add channel dimension

        return torch.from_numpy(img), torch.from_numpy(msk)
"""
    Soft Dice loss computed on probabilities (sigmoid output).
    - logits: raw model outputs (B,1,H,W)
    - targets: ground truth masks in {0,1} (B,1,H,W)
    """
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))     # Compute Dice per batch element (sum over H,W)
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


@torch.no_grad()  
def eval_dice(model, loader, device): 
    """Evaluate Dice score on a dataloader by thresholding probabilities at 0.5.  
    Returns the average Dice over batches.
    """
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()

        inter = (pred * y).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + y.sum(dim=(2, 3)) + 1e-6
        dice = (2 * inter / union).mean().item()

        total += dice
        n += 1
    return total / max(1, n)

#Training loop
def main():
    ap = argparse.ArgumentParser(description="Train U-Net for binary segmentation (RGB -> mask).")
    ap.add_argument("--images", required=True, help="RGB folder containing *_colors.png (recursive).")
    ap.add_argument("--masks", required=True, help="Masks folder containing *_mask.png.")
    ap.add_argument("--out", default="seg_unet_best.pth", help="Output checkpoint (.pth).")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None, help="Optional: limit number of samples.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    
    # Create dataset of paired RGB and mask samples
    ds = RGBMaskDataset(args.images, args.masks, img_size=args.img_size, limit=args.limit)
    
    # Create shuffled indices and split into train/validation
    idx = list(range(len(ds)))
    random.shuffle(idx)
    split = int(len(idx) * (1 - args.val_split))
    tr_idx, va_idx = idx[:split], idx[split:]
    
    # Build PyTorch Subset objects
    tr = torch.utils.data.Subset(ds, tr_idx)
    va = torch.utils.data.Subset(ds, va_idx)
    
    # DataLoaders: shuffle train, keep val fixe
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize model, optimizer, and losses
    model = SimpleUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()   # BCEWithLogitsLoss expects raw logits (no sigmoid in model output)
    
    # Track the best validation Dice score and save checkpoint when improved
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tr_loader:   # Training over batches
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)  # Combined loss: BCE + Dice (common for segmentation)

            opt.zero_grad()
            loss.backward()
            opt.step()

        val_dice = eval_dice(model, va_loader, device)     # Validation Dice after each epoch
        print(f"[EPOCH {ep:03d}] val_dice={val_dice:.4f}")
        
        # Save best model checkpoint
        if val_dice > best:
            best = val_dice
            torch.save({"model": model.state_dict(), "img_size": args.img_size}, args.out)
            print(f"[INFO] Saved best checkpoint -> {args.out}")

    print(f"[INFO] Training done. Best val_dice={best:.4f}")


if __name__ == "__main__":
    main()
