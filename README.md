# Custom YOLO Object Detection

A custom YOLO-based object detection project for detecting **cartons, cement bags, and wooden objects**. The trained model can perform real-time object detection on **live video feeds**, as well as detect objects in **images and pre-recorded videos**.

### Classes

* Carton
* Cement Bag
* Wooden Object

### Features

* Custom YOLO model training
* Image detection
* Video detection
* Live webcam detection
* Bounding box and confidence score visualization


<div align="center">
  
# 🏭 Industrial Object Detection (YOLOv8s)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8s-00FFFF?style=for-the-badge)](https://docs.ultralytics.com/)
[![mAP@50](https://img.shields.io/badge/mAP%4050-97.4%25-brightgreen?style=for-the-badge)](#-performance-benchmarks)
[![Release](https://img.shields.io/badge/Release-v1.3-blue?style=for-the-badge)](https://github.com/DIVIT-S/yolo-object-detection/releases)

*A production-grade, state-of-the-art computer vision pipeline designed for automated material handling, logistics tracking, and smart warehouse automation.*

</div>

---

## 📌 Overview

This repository hosts a custom-trained **YOLOv8 Small (YOLOv8s)** model optimized to detect, classify, and localize three primary industrial materials with near-perfect accuracy:
1. 📦 **Carton** (Boxes / Packaging)
2. 🏗️ **Cement Bag** (Construction materials)
3. 🪵 **Wood** (Timber / Logs)

---

## 🧠 Model Architecture & Training Specs

* **Base Architecture:** YOLOv8s (Ultralytics)
* **Parameters:** 11.1 Million
* **GFLOPs:** 28.4
* **Input Resolution:** 640x640 pixels
* **Hardware Used:** NVIDIA Tesla T4 GPU
* **Epochs Trained:** 50

---

## 📊 Performance Benchmarks

Tested on a rigorous validation set of **618 images** containing **1,463 target instances**, the model demonstrates pixel-level edge precision and highly reliable detection.

### Overall Metrics
| Metric | Score | Accuracy |
| :--- | :---: | :---: |
| **mAP @ 0.50** | **0.974** | **97.4%** |
| **mAP @ 0.50 - 0.95** | **0.827** | **82.7%** |
| **Precision (P)** | **0.940** | **94.0%** |
| **Recall (R)** | **0.954** | **95.4%** |

### Class-Wise Perfection Metrics
| Class Name | Instances | Precision | Recall | **mAP@50** | **mAP@50-95** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🪵 **Wood** | 566 | 0.950 | 0.973 | **98.3%** | 85.0% |
| 📦 **Carton** | 396 | 0.899 | 0.960 | **97.1%** | 82.9% |
| 🏗️ **Cement Bag** | 501 | 0.971 | 0.928 | **96.8%** | 80.1% |

> **Note:** Evaluation plots including the Confusion Matrix, F1-Curves, and Precision-Recall graphs can be found in the `results/` directory.

---

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/DIVIT-S/yolo-object-detection.git
cd yolo-object-detection
pip install -r requirements.txt
```

---

## 🚀 How to Run (`detect.py`)

The repository includes a highly optimized, smart detection script (`detect.py`) that handles images, videos, and live webcam feeds seamlessly.

### 🔴 1. Live Webcam
Run the script with no arguments to initialize webcam mode. It opens your camera and draws bounding boxes in real-time.
```bash
python detect.py
```
> *Press `q` to quit the live stream.*

### 🖼️ 2. Single Image
Pass an image path. The script auto-detects the extension (`.jpg`, `.png`, etc.), runs a single-pass inference, and displays the annotated result.
```bash
python detect.py photo.jpg
```
> *Press any key to close the image window.*

### 🎥 3. Video File
Pass a video path (`.mp4`, `.avi`, etc.). The script processes it frame-by-frame and opens a live playback window.
```bash
python detect.py clip.mp4
```
> *Ends automatically when the video finishes, or press `q` to stop early.*

---

## 💾 Saving Outputs

By default, the script only *displays* the results to keep it lightweight. To write annotated images or videos to disk, use the `--save` flag.

```bash
# Save an annotated image
python detect.py photo.jpg --save

# Save an annotated video
python detect.py clip.mp4 --save
```

* **Default Save Location:** Outputs are saved to a dynamically created `outputs/` folder in your current directory (e.g., `outputs/photo.jpg`).
* **Custom Save Location:** You can override the output directory using `--out-dir`:
  ```bash
  python detect.py photo.jpg --save --out-dir custom_results/
  ```
> **Note on Webcams:** Passing `--save` with webcam mode intentionally does nothing, as live open-ended streams require a fixed duration to finalize the file. 

---

## 🏎️ Under the Hood: How it Works

The `detect.py` engine is built for maximum efficiency and zero-configuration routing:

1. **Smart Weight Loading:** On startup, the script automatically searches the `runs/detect/**/` directories for the most recently created `best.pt` file. If no local training runs are found, it gracefully falls back to the production `best.pt` weights committed in the root repository.
2. **Data-Driven Confidence:** Detections use a hardcoded `conf=0.468`. This isn't a random number—it is the exact threshold where the model's **F1 score peaked during training**, perfectly balancing precision and recall while filtering out low-confidence noise.
3. **Memory Efficient Processing:** Video and webcam feeds utilize `stream=True` internally, ensuring Python yields a generator of results rather than overloading RAM with frame arrays.
4. **Dynamic Rendering:** `draw_detections()` parses YOLO's tensor outputs (mapping IDs 0, 1, and 2 to Carton, Cement Bag, and Wood), and leverages `cv2` to draw crisp, color-coded rectangles and confidence scores before displaying or routing to `cv2.VideoWriter`.

---

