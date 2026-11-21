"""
End-to-end test script for the Food Volume Estimation Pipeline.

Usage:
    python test_volume_pipeline.py --image path/to/food_image.jpg
    python test_volume_pipeline.py --image path/to/food_image.jpg --save-viz

Tests the full chain:
  YOLO detection → SAM2 segmentation → Depth Anything V2 → Scale recovery
  → Volume → Mass → Nutrition

Compares volume-based nutrition estimates against the fixed values
in class_names.py to show the difference.
"""

import argparse
import sys
import os
import time
import logging
import numpy as np
import cv2
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Food Volume Estimation Pipeline"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input food image (JPG/PNG)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="YOLO detection confidence threshold (default: 0.35)",
    )
    parser.add_argument(
        "--save-viz", action="store_true",
        help="Save visualization images (mask overlay, depth map, scale)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./test_output",
        help="Directory for saving visualizations (default: ./test_output)"
    )
    parser.add_argument(
        "--no-sam", action="store_true",
        help="Skip SAM2 (use bbox fallback) for faster testing"
    )
    parser.add_argument(
        "--no-depth", action="store_true",
        help="Skip depth estimation for faster testing"
    )
    return parser.parse_args()


def load_yolo_and_detect(image_path: str, conf: float):
    """Load YOLO model and run detection."""
    from class_names import class_names
    import onnxruntime as ort

    model_path = "./model/yolov26/best.onnx"
    if not os.path.exists(model_path):
        logger.error(f"YOLO model not found at {model_path}")
        sys.exit(1)

    logger.info(f"Loading YOLO model from {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read image: {image_path}")
        sys.exit(1)
    h_orig, w_orig = img.shape[:2]

    # Preprocess to 640x640
    img_resized = cv2.resize(img, (640, 640))
    input_tensor = img_resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, 0)

    logger.info(f"Running detection on {image_path} (conf={conf})")
    outputs = session.run(output_names, {input_name: input_tensor})
    preds = outputs[0][0]  # shape (300, 6): [x1, y1, x2, y2, score, class_id]

    scale_x = w_orig / 640.0
    scale_y = h_orig / 640.0

    detections = []
    for pred in preds:
        score = float(pred[4])
        if score >= conf:
            class_id = int(pred[5])
            x1 = max(0, min(w_orig, int(pred[0] * scale_x)))
            y1 = max(0, min(h_orig, int(pred[1] * scale_y)))
            x2 = max(0, min(w_orig, int(pred[2] * scale_x)))
            y2 = max(0, min(h_orig, int(pred[3] * scale_y)))

            c_name = class_names[class_id]["name"] if 0 <= class_id < len(class_names) else f"Class_{class_id}"
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class_name": c_name,
                "class_id": class_id,
                "confidence": score,
            })

    logger.info(f"Detected {len(detections)} food items:")
    for d in detections:
        logger.info(f"  - {d['class_name']} ({d['confidence']:.0%})")

    return detections, outputs


def compare_with_fixed_nutrition(detections, estimations):
    """Compare volume-based estimates with fixed class_names.py values."""
    from class_names import class_names

    print("\n" + "=" * 80)
    print("  COMPARISON: Volume-based vs Fixed Nutrition")
    print("=" * 80)
    print(f"{'Food':<40} {'Fixed Cal':>10} {'Est. Cal':>10} {'Est. Mass':>10}")
    print("-" * 80)

    for det, est in zip(detections, estimations):
        class_id = det["class_id"]
        fixed_cal = class_names[class_id]["nutrition"]["Calories"] if 0 <= class_id < len(class_names) else 0.0
        est_cal = est.nutrition.get("Calories", 0.0)
        est_mass = f"{est.mass_g:.1f}±{est.mass_std_g:.1f}g"

        print(f"{est.class_name:<40} {fixed_cal:>10.1f} {est_cal:>10.1f} {est_mass:>10}")

    print("-" * 80)


def save_visualizations(result, image_rgb, output_dir):
    """Save visualization images."""
    os.makedirs(output_dir, exist_ok=True)

    if result.mask_overlay is not None:
        mask_path = os.path.join(output_dir, "mask_overlay.jpg")
        cv2.imwrite(mask_path, cv2.cvtColor(result.mask_overlay, cv2.COLOR_RGB2BGR))
        logger.info(f"Mask overlay saved: {mask_path}")

    if result.depth_colored is not None:
        depth_path = os.path.join(output_dir, "depth_map.jpg")
        cv2.imwrite(depth_path, result.depth_colored)
        logger.info(f"Depth map saved: {depth_path}")

    if result.scale_overlay is not None:
        scale_path = os.path.join(output_dir, "scale_overlay.jpg")
        cv2.imwrite(scale_path, result.scale_overlay)
        logger.info(f"Scale overlay saved: {scale_path}")

    # Save per-item crops
    for i, est in enumerate(result.estimations):
        x1, y1, x2, y2 = est.bbox
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size > 0:
            crop_path = os.path.join(output_dir, f"item_{i}_{est.class_name[:20]}.jpg")
            cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    logger.info(f"All visualizations saved to {output_dir}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    # Validate input
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        sys.exit(1)

    # Load image
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        logger.error(f"Failed to read image: {args.image}")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    logger.info(f"Image loaded: {w}×{h}")

    # Step 1: YOLO detection
    t0 = time.time()
    detections, yolo_results = load_yolo_and_detect(args.image, args.conf)
    t_yolo = time.time() - t0

    if not detections:
        logger.warning("No food items detected. Exiting.")
        sys.exit(0)

    # Step 2: Run the volume pipeline
    from food_volume_pipeline import FoodVolumePipeline

    pipeline = FoodVolumePipeline()

    # Override modules if --no-sam or --no-depth flags
    if args.no_sam:
        logger.info("--no-sam: SAM2 disabled, using bbox fallback")
        pipeline._segmenter = SAM2Segmenter.__new__(SAM2Segmenter)
        pipeline._segmenter._available = False
        pipeline._segmenter._predictor = None
        pipeline._segmenter.erode_kernel = 3
        pipeline._segmenter.oversized_threshold = 0.9

    t1 = time.time()
    result = pipeline.analyze(
        image_rgb=image_rgb,
        yolo_detections=detections,
        image_path=args.image,
        generate_visualizations=args.save_viz,
    )
    t_pipeline = time.time() - t1

    # Print results
    print(pipeline.format_summary(result))

    # Timing
    print(f"\n⏱ Timing:")
    print(f"  YOLO detection:  {t_yolo:.2f}s")
    print(f"  Volume pipeline: {t_pipeline:.2f}s")
    print(f"  Total:           {t_yolo + t_pipeline:.2f}s")

    # Compare with fixed nutrition
    compare_with_fixed_nutrition(detections, result.estimations)

    # Module availability
    print(f"\n🔧 Module status:")
    print(f"  SAM2:    {'✅ loaded' if pipeline.segmenter.is_available else '❌ fallback (bbox)'}")
    print(f"  Depth:   {'✅ loaded' if pipeline.depth_estimator.is_available else '❌ unavailable'}")
    print(f"  Scale:   {result.scale_result.scale_source} "
          f"(confidence={result.scale_result.confidence})")

    # Save visualizations
    if args.save_viz:
        save_visualizations(result, image_rgb, args.output_dir)

    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
