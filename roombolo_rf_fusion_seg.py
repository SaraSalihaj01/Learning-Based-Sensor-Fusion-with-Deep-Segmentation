
import cv2
import numpy as np
import argparse
import os
import glob
import csv
import joblib
import torch
import torch.nn as nn

# Configuration 
FOV_H_DEG = 70
US_CONE_DEG = 15
ROI_Y_LOW, ROI_Y_HIGH = 0.55, 0.95
CANNY_LOW, CANNY_HIGH = 60, 150
EDGE_DENSITY_THR = 0.16
D0 = 0.6

# YOLO (loaded only if enabled)
YOLO_MODEL = None
YOLO_WEIGHTS = "yolov8n.pt"
OBSTACLE_CLASSES = {
    "person", "chair", "couch", "sofa", "dining table", "table", "bench", "bed",
    "tv", "refrigerator", "potted plant", "suitcase", "backpack", "bottle",
    "book", "laptop", "box"
}

# Random Forest model (loaded at runtime)
RF_MODEL = None


#U-Net (same architecture used for training)
class SimpleUNet(nn.Module):
    """
    A compact U-Net for binary segmentation.
    Input:  RGB image (3 channels)
    Output: 1-channel logits (apply sigmoid to get probability map)
    """
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bott = conv_block(base * 4, base * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bott(self.pool3(e3))

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


def load_seg_model(path: str, device: str):
    """
    Load a trained segmentation checkpoint.
    The checkpoint is expected to contain:
      - "model": state_dict
      - "img_size": training resize size (optional, default 256)
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
    Predict a binary mask (uint8 0/255) at the original image resolution.

    Steps:
    - Convert BGR -> RGB
    - Resize to model input size
    - Run model to get logits -> sigmoid -> probability map
    - Resize prob map back to original size
    - Threshold to create a binary mask
    """
    h, w = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.0).transpose(2, 0, 1)  # C,H,W

    x = torch.from_numpy(inp)[None, ...].to(device)
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (prob > thr).astype(np.uint8) * 255
    return mask


#Utilities 
def roi_bounds(w, h, fov_deg=FOV_H_DEG, cone_deg=US_CONE_DEG):
    """
    Compute a central ROI approximating the ultrasound cone projection in the image.
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


def ultrasonic_sim(frame_id, mode):
    """
    Simulate an ultrasonic distance measurement when depth is not available.
    Returns (d_u, P_u)
    """
    rng = np.random.default_rng(1234 + frame_id)

    if mode == "near":
        d_u = float(rng.uniform(0.25, 0.45))
    elif mode == "far":
        d_u = float(rng.uniform(0.9, 1.5))
    else:
        block = (frame_id // 40) % 3
        if block == 0:
            d_u = float(rng.uniform(0.25, 0.45))
        elif block == 1:
            d_u = float(rng.uniform(0.9, 1.5))
        else:
            d_u = float(rng.uniform(0.35, 1.2))

    # Convert distance to a probability-like measure.
    P_u = float(np.exp(-min(d_u, 5.0) / D0))
    return d_u, P_u


def ultrasonic_from_depth(depth, roi):
    """
    Estimate an ultrasonic-like distance from the depth frame inside the ROI.
    Uses the 10th percentile for robustness (closer points dominate).
    Returns (d_u, P_u).
    """
    y1, y2, x1, x2 = roi
    if depth is None:
        return None, None

    d = depth[y1:y2, x1:x2]
    if d is None:
        return None, None

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


def frames_from_folder(folder, pattern="*_colors.png"):
    """
    Generator for RGB frames (BGR images loaded by OpenCV).
    """
    paths = sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    print(f"[INFO] RGB found: {len(paths)}")
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            yield img


def depth_frames(folder, pattern="*_depth.png"):
    """
    Generator for depth frames.
    """
    paths = sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    print(f"[INFO] DEPTH found: {len(paths)}")
    for p in paths:
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        yield d


#YOLO helpers 
def iou(a, b):
    """
    Intersection-over-Union between two boxes (x1,y1,x2,y2).
    """
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
    Compute a robust distance estimate inside a bounding box from depth map.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(depth.shape[1] - 1, x2)
    y2 = min(depth.shape[0] - 1, y2)

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


# Main 
def main():
    parser = argparse.ArgumentParser(description="RF Fusion with Segmentation -> Canny(mask)")
    parser.add_argument("--images", type=str, required=True, help="RGB folder (recursive, *_colors.png).")
    parser.add_argument("--depth", type=str, default=None, help="Depth folder (recursive, *_depth.png).")
    parser.add_argument("--mode", type=str, default="pattern", choices=["near", "far", "pattern"],
                        help="Ultrasound simulation mode if depth is not provided.")
    parser.add_argument("--save_csv", type=str, default="log_rf.csv", help="Output CSV log file.")
    parser.add_argument("--display", action="store_true", help="Show visualization window.")

    # Playback controls
    parser.add_argument("--fps", type=float, default=5.0, help="Playback speed (frames per second).")
    parser.add_argument("--step", action="store_true", help="Step mode: press a key to advance frames.")

    # YOLO options
    parser.add_argument("--use_yolo", action="store_true", help="Enable YOLO object detection features.")
    parser.add_argument("--yolo_conf", type=float, default=0.4, help="Minimum YOLO confidence.")
    parser.add_argument("--yolo_dist", type=float, default=0.8, help="Distance threshold (m) for obstacle.")
    parser.add_argument("--yolo_iou_roi", type=float, default=0.10, help="Min IoU with ROI to keep detection.")

    # Random Forest model
    parser.add_argument("--rf_model", type=str, required=True, help="Path to trained Random Forest (.joblib).")
    parser.add_argument("--rf_thresh", type=float, default=0.5, help="Obstacle probability threshold (0..1).")

    # Segmentation model
    parser.add_argument("--seg_model", type=str, required=True, help="Path to segmentation checkpoint (.pth).")
    parser.add_argument("--seg_thr", type=float, default=0.5, help="Segmentation threshold (0..1).")

    args = parser.parse_args()

    # Load segmentation model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model, seg_img_size = load_seg_model(args.seg_model, device)
    print(f"[INFO] Segmentation model loaded on {device}, img_size={seg_img_size}")

    # Load YOLO only if requested
    global YOLO_MODEL
    if args.use_yolo and YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise SystemExit(f"[ERROR] YOLO not available. Install with: pip install ultralytics\nDetails: {e}")
        YOLO_MODEL = YOLO(YOLO_WEIGHTS)
        print("[INFO] YOLO model loaded.")

    # Load Random Forest model
    global RF_MODEL
    RF_MODEL = joblib.load(args.rf_model)
    print(f"[INFO] Loaded RF model from {args.rf_model}")

    # Input generators
    rgb_gen = frames_from_folder(args.images)
    depth_gen = depth_frames(args.depth) if args.depth else None

    # CSV logging
    os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
    fcsv = open(args.save_csv, "w", newline="")
    wcsv = csv.writer(fcsv)
    wcsv.writerow([
        "frame",
        "edge_density", "f_v", "w_v",
        "d_u_m", "P_u",
        "yolo_has_det", "yolo_conf", "yolo_dist_m", "yolo_is_obst",
        "rf_prob_obstacle", "Decision"
    ])

    frame_id = 0
    paused = False

    for frame in rgb_gen:
        h, wimg = frame.shape[:2]
        x1, y1, x2, y2 = roi_bounds(wimg, h)
        roi_box = (x1, y1, x2, y2)

        # Segmentation -> mask -> Canny(mask)
        mask_u8 = predict_mask(frame, seg_model, device, seg_img_size, args.seg_thr)
        f_v, w_v, edge_density, edges = visual_cues_from_mask(frame, mask_u8, roi_box)

        # YOLO features
        yolo_has_det = 0
        yolo_conf = 0.0
        yolo_dist = np.inf
        yolo_is_obst = 0
        yolo_objs = []

        if args.use_yolo:
            res = YOLO_MODEL.predict(frame, verbose=False)[0]
            for b, c, conf in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int),
                res.boxes.conf.cpu().numpy()
            ):
                if conf < args.yolo_conf:
                    continue
                cls_name = res.names[c]
                bx1, by1, bx2, by2 = map(int, b)
                box = (bx1, by1, bx2, by2)
                if iou(roi_box, box) >= args.yolo_iou_roi:
                    yolo_objs.append((cls_name, float(conf), box))

        # Ultrasonic distance (from depth or simulated)
        if depth_gen is not None:
            try:
                depth_frame = next(depth_gen)
            except StopIteration:
                depth_frame = None
            d_u, P_u = ultrasonic_from_depth(depth_frame, (y1, y2, x1, x2))
        else:
            d_u, P_u = ultrasonic_sim(frame_id, args.mode)

        # If YOLO + depth, estimate object distance for best detection
        if args.use_yolo and (depth_gen is not None) and (depth_frame is not None):
            det_rows = []
            for cls_name, conf, box in yolo_objs:
                d_obj = depth_in_box(depth_frame, box, percentile=10)
                is_obst = int((cls_name in OBSTACLE_CLASSES) and (d_obj < args.yolo_dist))
                det_rows.append((cls_name, conf, d_obj, is_obst))

            if det_rows:
                det_rows.sort(key=lambda r: r[1], reverse=True)
                _, top_conf, top_dist, top_isobst = det_rows[0]
                yolo_has_det = 1
                yolo_conf = float(top_conf)
                yolo_dist = float(top_dist)
                yolo_is_obst = int(top_isobst)

        # Build feature vector for Random Forest (ORDER MUST MATCH TRAINING)
        feat_edge_density = float(edge_density)
        feat_f_v = float(f_v)
        feat_w_v = float(w_v)
        feat_d_u = float(d_u if np.isfinite(d_u) else 5.0)
        feat_P_u = float(P_u)
        feat_yolo_has_det = float(yolo_has_det)
        feat_yolo_conf = float(yolo_conf)
        feat_yolo_dist = float(yolo_dist if np.isfinite(yolo_dist) else 5.0)
        feat_yolo_is_obst = float(yolo_is_obst)

        feature_vector = [
            feat_edge_density,
            feat_f_v,
            feat_w_v,
            feat_d_u,
            feat_P_u,
            feat_yolo_has_det,
            feat_yolo_conf,
            feat_yolo_dist,
            feat_yolo_is_obst
        ]

        # Random Forest prediction
        # We only pass the 3 features used during training: edge_density, d_u, P_u
        reduced_feature_vector = [edge_density, d_u, P_u]
        X = np.array([reduced_feature_vector], dtype=np.float32)
        
        proba = RF_MODEL.predict_proba(X)[0]
        prob_obstacle = float(proba[1])
        decision = "OBSTACLE" if prob_obstacle > args.rf_thresh else "FREE"
        # Log to CSV
        wcsv.writerow([
            frame_id,
            f"{edge_density:.3f}", f"{feat_f_v:.3f}", f"{feat_w_v:.3f}",
            f"{feat_d_u:.3f}", f"{feat_P_u:.3f}",
            int(yolo_has_det),
            f"{yolo_conf:.3f}",
            f"{yolo_dist:.3f}" if np.isfinite(yolo_dist) else "",
            int(yolo_is_obst),
            f"{prob_obstacle:.3f}",
            decision
        ])

        # Visualization 
        if args.display:
            vis = frame.copy()
            
            #Draw only the present yolo boxes
            if args.use_yolo and yolo_objs:
                for cls_name, conf, box in yolo_objs:
                    bx1, by1, bx2, by2 = box
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                    cv2.putText(vis, f"{cls_name}", (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            # HUD (Heads-Up Display) at the top
            color = (0, 0, 255) if decision == "OBSTACLE" else (0, 200, 0)
            hud = f"d={feat_d_u:.2f}m | RF p(obj)={prob_obstacle:.2f} -> {decision}"
            cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ROI Overlay (Semi-transparent Red/Green area)
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1) #Filled rectangle
            vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
            
            # Draw ROI border
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            cv2.imshow("Roombolo AI Fusion (U-Net + RF)", vis)

        delay = 0 if args.step or paused else max(1, int(1000 / max(0.1, args.fps)))
        key = cv2.waitKey(delay) & 0xFF
        if key == ord(" "):          # Space: pause/resume
            paused = not paused
        elif key in (27, ord("q")):  # ESC or q: quit
            break

        frame_id += 1

    fcsv.close()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
