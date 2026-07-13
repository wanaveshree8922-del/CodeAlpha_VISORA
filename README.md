
## CodeAlpha_VISORA
 
**VISORA** is a real-time object detection and tracking web application built using YOLOv8 and DeepSORT, developed as part of my **CodeAlpha AI Internship**.

It transforms live webcam video into intelligent insights — detecting objects in real time and tracking each one with a persistent ID as it moves across the frame, all wrapped in a clean, cinematic web interface.

## ✨ Features

- **Real-time object detection** using YOLOv8
- **Multi-object tracking** with DeepSORT — every detected object keeps a consistent ID across frames
- **Live webcam feed** streamed directly to the browser
- **Custom-built UI** — dark, minimal welcome screen with a glassmorphic camera control

## 🛠️ Tech Stack
```bash
| Layer            | Technology                    |
|------------------|-------------------------------|
| Object Detection | YOLOv8 (Ultralytics)          |
| Object Tracking  | DeepSORT (deep-sort-realtime) |
| Backend          | Flask                         |
| Video Handling   | OpenCV                        |
| Frontend         | HTML, CSS, JavaScript         |
```
## ⚙️ How It Works

1. Flask opens the local webcam using OpenCV
2. Each frame is passed through YOLOv8 to detect objects
3. Detections are passed to DeepSORT, which tracks each object across frames and assigns a consistent ID
4. The annotated frame (bounding boxes + labels + IDs) is streamed to the browser as a live MJPEG feed
5. The frontend shows a welcome screen, then switches to the live detection feed when the user clicks Open Camera


## 🚀 Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/wanaveshree8922-del/CodeAlpha_VISORA.git
cd CodeAlpha_VISORA
```

**2. Create a virtual environment (Python 3.11 recommended)**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

**3. Install dependencies**
```bash
pip install ultralytics deep-sort-realtime flask opencv-python "setuptools<81"
```

**4. Run the app**
```bash
python app.py
```

**5. Open in your browser**
```
http://127.0.0.1:5000
```

## 📁 Project Structure

```
CodeAlpha_VISORA/
├── static/
│   ├── bg.png              # Background image
│   ├── style.css           # Styling
│   └── fonts/               # Custom fonts
├── templates/
│   └── index.html          # Welcome page + camera UI
├── app.py                  # Flask backend + YOLO/DeepSORT logic
└── README.md
```

## 🔮 Future Improvements

- Deploy with browser-based camera access so it works for any visitor, not just on localhost
- Add object count / analytics dashboard
- Support video file uploads in addition to live webcam

## 🙏 Acknowledgements

Built using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime), as part of the **CodeAlpha Artificial Intelligence Internship**.

```
##Author: Shreyas
