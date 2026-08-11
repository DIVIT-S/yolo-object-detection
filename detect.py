import argparse
import os
import sys

import cv2
from ultralytics import YOLO

# 1. Locate Trained Weights Automatically (Supports default and custom run names)
WEIGHTS_PATH = 'weights/best.pt'

if not os.path.exists(WEIGHTS_PATH):
    for alternate in [
        'results/YOLOv8 Industrial Training Results/weights/best.pt',
        'runs/detect/train-3/weights/best.pt',
    ]:
        if os.path.exists(alternate):
            WEIGHTS_PATH = alternate
            break

if not os.path.exists(WEIGHTS_PATH):
    detect_dir = 'runs/detect'
    if os.path.exists(detect_dir):
        found_weights = []
        for root, _, files in os.walk(detect_dir):
            if 'best.pt' in files:
                full_path = os.path.join(root, 'best.pt')
                found_weights.append(full_path)
        if found_weights:
            WEIGHTS_PATH = max(found_weights, key=os.path.getctime)

# Fall back to the repo root best.pt if present
if not os.path.exists(WEIGHTS_PATH) and os.path.exists('best.pt'):
    WEIGHTS_PATH = 'best.pt'

if not os.path.exists(WEIGHTS_PATH):
    print(f"[ERROR] Trained weights not found at '{WEIGHTS_PATH}'.")
    print("Please ensure your 'best.pt' file is placed inside the 'runs/detect/' directory, or the repo root.")
    sys.exit(1)

print(f'[INFO] Loading model weights from: {WEIGHTS_PATH}')
model = YOLO(WEIGHTS_PATH)

# Class Index Mapping
CLASS_NAMES = {0: 'Carton', 1: 'Cement Bag', 2: 'Wood'}

# Bounding Box Color Palette (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),    # Green for Carton
    1: (255, 165, 0),  # Orange for Cement Bag
    2: (0, 165, 255),  # Cyan/Amber for Wood
}

# Use a reasonable default confidence threshold so the model does not draw
# many low-confidence boxes that appear incorrect in practice.
DEFAULT_CONF = 0.10

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}


def draw_detections(frame, results, threshold):
    """Draws bounding boxes, class labels, and confidence scores onto a frame."""
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < threshold:
                continue

            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            class_name = CLASS_NAMES.get(cls_id, 'Unknown')
            label = f'{class_name} {conf:.2f}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(y1 - 20, 0)), (x1 + w, max(y1, 20)), color, -1)
            cv2.putText(
                frame, label, (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
            )
    return frame


def run_on_image(source, save, out_dir, conf):
    """Runs detection on a single image and displays/saves the annotated result."""
    frame = cv2.imread(source)
    if frame is None:
        print(f'[ERROR] Could not read image: {source}')
        sys.exit(1)

    print(f'[INFO] Inference confidence threshold: {conf}')
    results = model(frame, conf=conf)
    frame = draw_detections(frame, results, conf)

    if save:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(source))
        cv2.imwrite(out_path, frame)
        print(f'[INFO] Saved annotated image to: {out_path}')

    cv2.imshow('YOLOv8 + OpenCV Object Detection', frame)
    print('[INFO] Press any key to close the window.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_video(source, save, out_dir, conf, is_webcam=False):
    """Runs detection on a video file or webcam stream."""
    if is_webcam:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f'[ERROR] Unable to open video source: {source if not is_webcam else "webcam"}')
        sys.exit(1)

    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    writer = None
    if save and not is_webcam:
        os.makedirs(out_dir, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(out_dir, os.path.basename(source))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f'[INFO] Saving annotated video to: {out_path}')

    print('[INFO] Starting video stream. Press "q" in the window to quit.')
    while True:
        ret, frame = cap.read()
        if not ret:
            if not is_webcam:
                print('[INFO] End of video.')
            else:
                print('[WARN] Failed to grab frame.')
            break

        results = model(frame, stream=True, conf=conf)
        frame = draw_detections(frame, results, conf)

        if writer is not None:
            writer.write(frame)

        cv2.imshow('YOLOv8 + OpenCV Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8 detection for Carton / Cement Bag / Wood — webcam, image, or video.'
    )
    parser.add_argument(
        'source', nargs='?', default=None,
        help='Path to an image or video file. Omit this to use the live webcam.',
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save the annotated output (image/video) to --out-dir instead of just displaying it.',
    )
    parser.add_argument(
        '--out-dir', default='detection_results',
        help='Directory to save annotated output when --save is set (default: detection_results/).',
    )
    parser.add_argument(
        '--conf', type=float, default=DEFAULT_CONF,
        help=f'Minimum confidence threshold for inference/drawing detections (default: {DEFAULT_CONF}).',
    )
    args = parser.parse_args()

    if args.source is None:
        run_on_video(None, args.save, args.out_dir, args.conf, is_webcam=True)
        return

    if not os.path.exists(args.source):
        print(f'[ERROR] Source not found: {args.source}')
        sys.exit(1)

    ext = os.path.splitext(args.source)[1].lower()
    if ext in IMAGE_EXTS:
        run_on_image(args.source, args.save, args.out_dir, args.conf)
    elif ext in VIDEO_EXTS:
        run_on_video(args.source, args.save, args.out_dir, args.conf, is_webcam=False)
    else:
        print(f'[ERROR] Unsupported file type: {ext}')
        sys.exit(1)


if __name__ == '__main__':
    main()