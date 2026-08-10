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

