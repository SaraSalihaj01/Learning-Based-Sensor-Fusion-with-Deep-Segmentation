import os, glob, argparse
import cv2
import numpy as np
import torch
import torch.nn as nn

# Model (same as training) 
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        def C(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        self.enc1 = C(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = C(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = C(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.bott = C(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = C(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = C(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = C(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bott(self.pool3(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

def dice_score(pred_u8, gt_u8, eps=1e-6):
    pred = (pred_u8 > 0).astype(np.uint8)
    gt   = (gt_u8 > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2.0 * inter + eps) / (denom + eps))

def find_pairs(images_dir, masks_dir):
    rgb_paths = sorted(glob.glob(os.path.join(images_dir, "**", "*_colors.png"), recursive=True))
    pairs = []
    for rp in rgb_paths:
        base = os.path.basename(rp).replace("_colors.png", "")
        mp = os.path.join(masks_dir, f"{base}_mask.png")
        if os.path.exists(mp):
            pairs.append((rp, mp, base))
    return pairs

@torch.no_grad()
def predict_mask(model, device, bgr, img_size=256, thr=0.5):
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = (x.astype(np.float32) / 255.0).transpose(2,0,1)
    x = torch.from_numpy(x)[None].to(device)
    logits = model(x)
    prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    prob = cv2.resize(prob, (w,h), interpolation=cv2.INTER_LINEAR)
    mask = (prob > thr).astype(np.uint8) * 255
    return mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to best_model.pth")
    ap.add_argument("--images", required=True, help="Folder with *_colors.png (test set)")
    ap.add_argument("--masks", required=True, help="Folder with *_mask.png (test GT)")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5, help="Sigmoid threshold for prediction")
    ap.add_argument("--save-dir", default=None, help="Optional: save predictions & overlays here")
    ap.add_argument("--max-save", type=int, default=50, help="How many examples to save")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device)

    model = SimpleUNet().to(device)
    #checkpoint may store either {"model": state_dict} or directly state_dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    pairs = find_pairs(args.images, args.masks)
    if not pairs:
        raise RuntimeError("No paired (RGB, MASK) found. Check folders and filenames.")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Paired test samples: {len(pairs)}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    dices = []
    saved = 0

    for rp, mp, base in pairs:
        bgr = cv2.imread(rp)
        gt  = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            continue

        pred = predict_mask(model, device, bgr, img_size=args.img_size, thr=args.thr)
        d = dice_score(pred, gt)
        dices.append(d)

        if args.save_dir and saved < args.max_save:
            # save prediction mask
            cv2.imwrite(os.path.join(args.save_dir, f"{base}_pred.png"), pred)
            # save overlay for report
            overlay = bgr.copy()
            col = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(overlay, 0.7, col, 0.3, 0)
            cv2.imwrite(os.path.join(args.save_dir, f"{base}_overlay.png"), overlay)
            saved += 1

    mean_dice = float(np.mean(dices)) if dices else 0.0
    print(f"[RESULT] TEST mean Dice = {mean_dice:.4f}")

if __name__ == "__main__":
    main()
