import os
import glob
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# Fusion dataset builder (Segmentation -> Canny(mask) + Depth + optional YOLO)
# ------------------------------------------------------------
# This script:
# 1) Reads paired RGB frames (*_colors.png) and depth frames (*_depth.png)
# 2) Computes visual features from a predicted segmentation mask (U-Net)
# 3) Computes depth-based "ultrasonic" features from a ROI in the depth map
# 4) Optionally adds YOLO detection features + depth-in-box distance
# 5) Generates a CSV dataset for downstream RF / ML fusion training
# ============================================================

#Configuration
FOV_H_DEG = 70
US_CONE_DEG = 15
ROI_Y_LOW, ROI_Y_HIGH = 0.55, 0.95
CANNY_LOW, CANNY_HIGH = 60, 150
EDGE_DENSITY_THR = 0.16
D0 = 0.6

YOLO_WEIGHTS = "yolov8n.pt"
YOLO_MODEL = None
OBSTACLE_CLASSES = {
    "person","chair","couch","sofa","dining table","table","bench","bed",
    "tv","refrigerator","potted plant","suitcase","backpack","bottle","book","laptop","box"
}


# -------------UNet (same as training/runtime)---------------------
# U-Net used to predict a binary obstacle mask from an RGB frame.
# The network outputs 1-channel logits (sigmoid -> probability).
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        # A small conv block used in encoder/decoder (Conv-BN-ReLU x2)
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        # Encoder
        self.enc1 = block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = block(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = block(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.bott = block(base*4, base*8)
        
        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = block(base*2, base)

        self.out = nn.Conv2d(base, out_ch, 1)    # Output: logits map

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
    
        b  = self.bott(self.pool3(e3)) # Bottleneck

        # Decoder path + skip concatenations
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

#Segmentation model I/O + inference 
def load_seg_model(path: str, device: str):
    """
    Load a trained U-Net checkpoint.
    Expects a dict with:
      - "model": state_dict
      - optional "img_size": input size used during training
    """
    ckpt = torch.load(path, map_location=device)
    model = SimpleUNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    img_size = int(ckpt.get("img_size", 256))
    return model, img_size


@torch.no_grad()
def predict_mask(frame_bgr: np.ndarray, model, device: str, img_size: int, thr: float) -> np.ndarray:
    """
    Predict a binary mask from an input BGR frame.
    Returns a uint8 mask (0 or 255) resized back to the original resolution.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.0).transpose(2, 0, 1)
    x = torch.from_numpy(inp)[None, ...].to(device)  #Forward pass
    logits = model(x)
    # Probability map -> resize back -> threshold -> uint8 0/255
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return (prob > thr).astype(np.uint8) * 255


# ROI + Utilities 
def roi_bounds(w, h, fov_deg=FOV_H_DEG, cone_deg=US_CONE_DEG):
    """
    Compute a rectangular ROI approximating the ultrasonic sensor cone:
    - narrow horizontal band around the image center (based on cone/fov ratio)
    - vertical span focused near the bottom of the image (ROI_Y_LOW..ROI_Y_HIGH)
    Returns (x1,y1,x2,y2) in pixel coordinates.
    """
    k = int((w * (cone_deg / fov_deg)) / 2)
    x1, x2 = w // 2 - k, w // 2 + k
    y1, y2 = int(ROI_Y_LOW * h), int(ROI_Y_HIGH * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    return x1, y1, x2, y2


def visual_cues_from_mask(frame, mask_u8, roi):
    x1, y1, x2, y2 = roi
    """
    Extracts visual features by combining the AI(U-Net) segmentation mask with 
    classical Canny edges to maintain consistency with the RF model training.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     #Compute Canny edges on the ORIGINAL grayscale image.This preserves the texture and detail information the RF model expects.
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges_raw = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    refined_edges = cv2.bitwise_and(edges_raw, mask_u8)  #Use the U-Net segmentation mask as a spatial filter. We keep edges only where the AI predicts an actual obstacle.

    dens = refined_edges[y1:y2, x1:x2].mean() / 255.0    #Calculate edge density within the specified ROI. Dividing by 255.0 converts the mean pixel intensity to a [0, 1] range.
    
    #Generate visual features for the Random Forest input vector.
    # w_v: continuous weight [0,1], f_v: binary flag based on density threshold.
    w_v = float(np.clip(dens / 0.5, 0, 1))          
    f_v = 1 if dens > EDGE_DENSITY_THR else 0

    return f_v, w_v, dens, refined_edges


def ultrasonic_from_depth(depth, roi):
    """
    Compute depth-based features from the depth ROI:
    - Converts depth to meters if needed (heuristic: if max > 50 -> mm)
    - Uses 10th percentile depth as a conservative distance estimate (d_u)
    - Computes a confidence-like probability P_u = exp(-d_u / D0)
    Returns (d_u, P_u). If invalid depth, returns (inf, 0).
    """
    y1, y2, x1, x2 = roi
    if depth is None:
        return np.inf, 0.0
    d = depth[y1:y2, x1:x2]
    if d is None:
        return np.inf, 0.0

    if d.dtype not in (np.float32, np.float64):
        d = d.astype(np.float32)
        if d.max() > 50:
            d *= 1e-3  # mm -> m

    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.inf, 0.0

    d_u = float(np.percentile(d[valid], 10))
    P_u = float(np.exp(-min(d_u, 5.0) / D0))
    return d_u, P_u

#Frame generators (RGB + Depth) 
def frames_from_folder(folder, pattern="*_colors.png"):
    """
    Yield (path, BGR image) for RGB frames under 'folder' recursively.
    """
    paths = sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    print(f"[INFO] RGB found: {len(paths)}")
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            yield p, img


def depth_frames(folder, pattern="*_depth.png"):
    """
    Yield (path, depth image) for depth frames under 'folder' recursively.
    Depth is read unchanged (keeps uint16 if present).
    """
    paths = sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    print(f"[INFO] DEPTH found: {len(paths)}")
    for p in paths:
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        yield p, d


# YOLO helpers
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def depth_in_box(depth, box, percentile=10):
    """
    Estimate object distance inside a detection box using depth map.
    Uses a percentile to be robust to outliers (similar to ultrasonic_from_depth).
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(depth.shape[1] - 1, x2); y2 = min(depth.shape[0] - 1, y2)
    d = depth[y1:y2, x1:x2]
    if d.size == 0:
        return np.inf

    if d.dtype not in (np.float32, np.float64):
        d = d.astype(np.float32)
        if d.max() > 50:
            d *= 1e-3

    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.inf
    return float(np.percentile(d[valid], percentile))

#Main: build CSV dataset
def main():
    """
    For each paired RGB+depth frame:
    - compute ROI
    - get segmentation mask and visual features
    - get depth features (d_u, P_u)
    - optionally compute YOLO features (and depth-based distance per detection)
    - set label from depth distance threshold
    - append row to CSV
    """
    global YOLO_MODEL

    ap = argparse.ArgumentParser(description="Create RF fusion dataset using Segmentation->Canny(mask).")
    ap.add_argument("--images", required=True, help="RGB folder with *_colors.png (recursive).")
    ap.add_argument("--depth", required=True, help="Depth folder with *_depth.png (recursive).")
    ap.add_argument("--seg_model", required=True, help="Trained segmentation checkpoint (.pth).")
    ap.add_argument("--seg_thr", type=float, default=0.5, help="Segmentation threshold.")
    ap.add_argument("--use_yolo", action="store_true", help="Enable YOLO features.")
    ap.add_argument("--yolo_conf", type=float, default=0.4, help="YOLO min confidence.")
    ap.add_argument("--yolo_dist", type=float, default=0.8, help="Distance threshold for obstacle (m).")
    ap.add_argument("--yolo_iou_roi", type=float, default=0.10, help="Min overlap with ROI.")
    ap.add_argument("--out_csv", default="fusion_dataset_seg.csv", help="Output CSV filename.")
    ap.add_argument("--label_thr", type=float, default=0.8, help="Label=1 if d_u < label_thr else 0.")
    args = ap.parse_args()
    

    # Device selection + load segmentation network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model, seg_img_size = load_seg_model(args.seg_model, device)
    print(f"[INFO] Seg model loaded on {device}, img_size={seg_img_size}")
    
    # Load YOLO only if requested
    if args.use_yolo:
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise SystemExit(f"[ERROR] YOLO not available. Install with: pip install ultralytics\nDetails: {e}")
        YOLO_MODEL = YOLO(YOLO_WEIGHTS)
        print("[INFO] YOLO model loaded.")

    rgb_gen = frames_from_folder(args.images)
    dep_gen = depth_frames(args.depth)

    rows = []
    frame_id = 0
    
    for (rgb_path, frame), (dep_path, depth) in zip(rgb_gen, dep_gen):
        h, wimg = frame.shape[:2]
        x1, y1, x2, y2 = roi_bounds(wimg, h)
        roi_box = (x1, y1, x2, y2)

        # Segmentation mask -> Canny(mask) -> visual features
        mask_u8 = predict_mask(frame, seg_model, device, seg_img_size, args.seg_thr)
        f_v, w_v, edge_density, _ = visual_cues_from_mask(frame, mask_u8, roi_box)
        # Depth-based ultrasonic-like features
        d_u, P_u = ultrasonic_from_depth(depth, (y1, y2, x1, x2))
        if not np.isfinite(d_u):
            frame_id += 1
            continue

        # YOLO features (optional)
        yolo_has_det = 0
        yolo_conf = 0.0
        yolo_dist = np.inf
        yolo_is_obst = 0

        if YOLO_MODEL is not None:
            res = YOLO_MODEL.predict(frame, verbose=False)[0]
            yolo_objs = []
            for b, c, conf in zip(res.boxes.xyxy.cpu().numpy(),
                                  res.boxes.cls.cpu().numpy().astype(int),
                                  res.boxes.conf.cpu().numpy()):
                if conf < args.yolo_conf:
                    continue
                cls_name = res.names[c]
                bx1, by1, bx2, by2 = map(int, b)
                box = (bx1, by1, bx2, by2)
                if iou(roi_box, box) >= args.yolo_iou_roi:
                    yolo_objs.append((cls_name, float(conf), box))

            if yolo_objs:
                det_rows = []
                for cls_name, conf, box in yolo_objs:
                    d_obj = depth_in_box(depth, box, percentile=10)
                    is_obst = int((cls_name in OBSTACLE_CLASSES) and (d_obj < args.yolo_dist))
                    det_rows.append((cls_name, conf, d_obj, is_obst))

                det_rows.sort(key=lambda r: r[1], reverse=True)
                _, top_conf, top_dist, top_isobst = det_rows[0]
                yolo_has_det = 1
                yolo_conf = float(top_conf)
                yolo_dist = float(top_dist)
                yolo_is_obst = int(top_isobst)

        # Ground-truth label from depth ROI distance 
        label = 1 if d_u < args.label_thr else 0

        rows.append({
            "frame": frame_id,
            "edge_density": float(edge_density),
            "f_v": float(f_v),
            "w_v": float(w_v),
            "d_u": float(d_u),
            "P_u": float(P_u),
            "yolo_has_det": float(yolo_has_det),
            "yolo_conf": float(yolo_conf),
            "yolo_dist": float(yolo_dist if np.isfinite(yolo_dist) else 5.0),
            "yolo_is_obst": float(yolo_is_obst),
            "label": int(label)
        })

        frame_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[INFO] Saved dataset with {len(df)} samples to {args.out_csv}")


if __name__ == "__main__":
    main()
