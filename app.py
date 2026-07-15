import os
from flask import Flask, render_template, Response, request
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load YOLOv8 model (nano version = fastest, good for real-time)
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

camera = None

def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)

    frame_count = 0
    last_tracks = []

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        if frame_count % 3 == 0:  # only run detection every 3rd frame
            results = model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]
                if conf > 0.5:
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, label))
            last_tracks = tracker.update_tracks(detections, frame=frame)

        for track in last_tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            label = track.get_det_class() if track.get_det_class() else "object"

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
         
         # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return ('', 204)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return {'error': 'No video uploaded'}, 400

    file = request.files['video']
    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    output_filename = 'processed_' + filename
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width, height = 640, 480  # Resize for faster processing

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    local_tracker = DeepSort(max_age=30)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))  # Resize for faster processing
        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]
            if conf > 0.5:
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, label))

        tracks = local_tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            label = track.get_det_class() if track.get_det_class() else "object"
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    return {'video_url': f'/static/processed/{output_filename}'}

if __name__ == '__main__':
    app.run(debug=True)