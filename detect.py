import os
import sys
import cv2
from ultralytics import YOLO

# 1. Locate Trained Weights Automatically (Supports default and custom run names)
WEIGHTS_PATH = 'runs/detect/carton_cement_wood_run/fast_30min_model/weights/best.pt'

if not os.path.exists(WEIGHTS_PATH):
  detect_dir = 'runs/detect'
  if os.path.exists(detect_dir):
    # Traverses all runs inside runs/detect to find the most recently created best.pt
    found_weights = []
    for root, _, files in os.walk(detect_dir):
      if 'best.pt' in files:
        full_path = os.path.join(root, 'best.pt')
        found_weights.append(full_path)

    if found_weights:
      WEIGHTS_PATH = max(found_weights, key=os.path.getctime)

if not os.path.exists(WEIGHTS_PATH):
  print(f"[ERROR] Trained weights not found at '{WEIGHTS_PATH}'.")
  print("Please ensure your 'best.pt' file is placed inside the 'runs/detect/' directory.")
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

# Optimal Confidence Threshold derived from F1-Confidence Curve
OPTIMAL_CONF = 0.436

def run_detection():
  """Performs real-time OpenCV object detection via camera feed."""
  # Use AVFOUNDATION backend for Apple Silicon stability
  cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

  if not cap.isOpened():
    cap = cv2.VideoCapture(0)

  if not cap.isOpened():
    print('[ERROR] Unable to open video capture device.')
    sys.exit(1)

  # Set camera capture resolution
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  print('[INFO] Starting video stream. Press "q" in the window to quit.')

  while True:
    ret, frame = cap.read()
    if not ret:
      print('[WARN] Failed to grab frame.')
      break

    # Run YOLO Inference with optimal F1 confidence threshold
    results = model(frame, stream=True, conf=OPTIMAL_CONF)

    for r in results:
      boxes = r.boxes
      for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        class_name = CLASS_NAMES.get(cls_id, 'Unknown')
        label = f'{class_name} {conf:.2f}'

        # Draw Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw Label Background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(y1 - 20, 0)), (x1 + w, max(y1, 20)), color, -1)

        # Draw Text
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow('YOLOv8 + OpenCV Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  run_detection()