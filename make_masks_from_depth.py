import os
import glob
import argparse
import imageio.v2 as imageio  #image reading (depth maps)
import cv2
import numpy as np


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    """
    Convert the depth map to meters.

    - If the depth is stored as uint16 (often millimeters) we convert it to float32.
    - Heuristic: if the maximum value is > ~50, we assume the units are millimeters and convert to meters.
    """
    if depth is None:
        return None

    d = depth
    if d.dtype not in (np.float32, np.float64):
        d = d.astype(np.float32)
        if np.nanmax(d) > 50:     # Heuristic: if max is large, depth is likely in millimeters
            d *= 1e-3             # Convert from mm -> m
    return d


def main():
    parser = argparse.ArgumentParser(
        description="Generate binary segmentation masks from depth maps."
    )
    parser.add_argument(
        "--depth",
        required=True,
        help="Folder containing *_depth.png files (searched recursively)."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output folder where masks will be saved."
    )
    parser.add_argument(
        "--pattern",
        default="*_depth.png",
        help="Depth filename pattern (default: *_depth.png)."
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.8,
        help="Obstacle threshold in meters (depth < thr -> obstacle)."   #Pixels with depth smaller than this value are classified as obstacles.
    )
    parser.add_argument(
        "--min_valid",
        type=float,
        default=0.05,
        help="Minimum valid depth (ignore smaller/invalid depth values)."
    )
    parser.add_argument(
        "--morph",
        action="store_true",
        help="Apply morphology (open+close) to clean the mask."   #Optional flag to apply morphological filtering.
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    paths = sorted([
    p for p in glob.glob(os.path.join(args.depth, "**", args.pattern), recursive=True)  #Recursively search for all depth images matching the pattern.
    if os.path.isfile(p)
    ])

    print(f"[INFO] Found depth frames: {len(paths)}")

    for p in paths:                    #Iterate over each depth image.
        depth = imageio.imread(p)
        #Make sure depth is 2D
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        d = depth_to_meters(depth)
        if d is None:
            continue

        # Build a binary mask:
        # 255 = obstacle (depth < threshold), 0 = background/free
        valid = np.isfinite(d) & (d > args.min_valid)
        mask = np.zeros(d.shape, dtype=np.uint8)
        mask[valid & (d < args.thr)] = 255



        # Optional cleanup
        """If morphology is enabled:
           - Apply opening (remove small noise)
           - Apply closing (fill small holes)"""
        if args.morph:                                                           
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        #Save mask with corresponding filename.
        base = os.path.basename(p)
        out_name = base.replace("_depth.png", "_mask.png")
        out_path = os.path.join(args.out, out_name)
        cv2.imwrite(out_path, mask)

    print("[INFO] Done. Masks saved to:", args.out)


if __name__ == "__main__":
    main()

