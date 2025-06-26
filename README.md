# Re-Identification in a Single Feed

This project performs **Player Re-Identification** in a single video feed using **YOLOv8** for detection and **DeepSORT** for tracking. It processes a video, detects objects (players), assigns unique IDs, overlays tracking boxes, and generates a real-time bar chart of detections.
⚠️ Note: This runs on CPU only, and hence the performance (FPS ~ 1) is limited. For better results, consider running on a CUDA-capable GPU.

---

## 📽️ Demo Output

A sample video (`output_reid.mp4`) is generated showing bounding boxes, track IDs, and live graph overlays.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/vanditam07/Re-Identification-in-a-single-feed.git
cd Re-Identification-in-a-single-feed
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv_cpu
venv_cpu\Scripts\activate     # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```bash
pip install ultralytics opencv-python deep_sort_realtime matplotlib numpy
```

---

## ▶️ How to Run

Make sure the following files are present in the root folder:

* `reid.py`  — the main script
* `15sec_input_720p.mp4`  — sample input video
* `best.pt` — YOLOv8 model weights *(Download manually, see below)*

### Run the script:

```bash
python reid.py
```

> 💡 Press `q` to exit the video window during execution.

---

## 📦 File Outputs

* `output_reid.mp4`: Final video with tracked IDs and graph overlay
* `detection_summary.csv`: Summary of total detections, frames, average detections per frame, total runtime, and FPS

---

## 🔗 Download Model Weights

Due to GitHub's file size limit, the YOLOv8 model (`best.pt`) is not included.

Please [download the model weights here](https://your-download-link.com) and place it in the root folder.

---

## ✅ Dependencies

* Python 3.10
* ultralytics
* opencv-python
* deep\_sort\_realtime
* matplotlib
* numpy

---

## 🧠 Model Info

* **YOLOv8** is used for object detection
* **DeepSORT** is used for multi-object tracking
* Re-Identification logic is applied in a single feed (not cross-camera)

---

## 📈 Example Stats

```
Class,Total Detections,Frames,Avg Per Frame
object,5557,375,14.82

Total Time (s),Average FPS
370.41,1.01
```

---

## 🙌 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [DeepSORT Realtime](https://github.com/levan92/deep_sort_realtime)

---

## 📬 Contact

**Vandita M**

Email: [vanditam07@gmail.com](mailto:vanditam14@gmail.com)
GitHub: [@vanditam07](https://github.com/vanditam07)

---


