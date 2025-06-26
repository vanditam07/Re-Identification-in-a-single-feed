from ultralytics import YOLO
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

def plot_to_image(class_counts_dict, total_frames):
    fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
    classes = list(class_counts_dict.keys())
    counts = [class_counts_dict[c] for c in classes]

    ax.bar(classes, counts, color='orange')
    ax.set_title(f"Detections @ Frame {total_frames}")
    ax.set_ylabel("Count")
    fig.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()

    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf



class_counts = defaultdict(int)
total_frames = 0
total_time = 0  

device = 'cpu'
print(f"[INFO] Using device: {device}")

model = YOLO("best.pt").to(device)

cap = cv2.VideoCapture("15sec_input_720p.mp4")
if not cap.isOpened():
    print("[ERROR] Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("output_reid.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

tracker = DeepSort(max_age=30)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    results = model.predict(source=frame, conf=0.3, device=device, verbose=False)

    boxes = []
    detections = results[0].boxes

    if detections is not None and detections.data is not None:
        det_data = detections.data.cpu().numpy()
        for det in det_data:
            x1, y1, x2, y2, conf, cls = det[:6]
            if conf > 0.3:
                w, h = x2 - x1, y2 - y1
                boxes.append(([x1, y1, w, h], conf, 'object'))
                class_counts['object'] += 1

    total_frames += 1

    tracks = tracker.update_tracks(boxes, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        # Generate graph and overlay
graph_img = plot_to_image(class_counts, total_frames)
graph_resized = cv2.resize(graph_img, (300, 200))  # Resize to fit on video

# Paste graph into top-left corner
frame[10:210, 10:310] = graph_resized


    out.write(frame)
    cv2.imshow("Player Re-ID (CPU)", frame)

    total_time += time.time() - start_time

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n=== Detection Summary ===")
if total_frames > 0:
    for cls, count in class_counts.items():
        print(f"{cls.capitalize():<10}: {count} detections total over {total_frames} frames")
        print(f"→ Avg per frame: {count / total_frames:.2f}")
else:
    print("No frames were processed.")

# === FPS and Timing ===
avg_fps = total_frames / total_time if total_time > 0 else 0
print(f"\nTotal Time: {total_time:.2f} seconds for {total_frames} frames")
print(f"Average FPS: {avg_fps:.2f}")

# === Save to CSV ===
csv_path = "detection_summary.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Class", "Total Detections", "Frames", "Avg Per Frame"])
    for cls, count in class_counts.items():
        writer.writerow([cls, count, total_frames, f"{count / total_frames:.2f}"])
    writer.writerow([])
    writer.writerow(["Total Time (s)", "Average FPS"])
    writer.writerow([f"{total_time:.2f}", f"{avg_fps:.2f}"])

print(f"\n📄 Detection stats saved to '{csv_path}'")
