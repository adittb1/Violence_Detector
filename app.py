import os
import json
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db

# =========================
# 🔧 KONFIGURASI
# =========================
# Telegram
TELEGRAM_TOKEN = "8217219791:AAFwlw1ca3IAO9711iJZBZzwaEN6HevXeqo"
CHAT_ID = "-1002923189078"  

# Firebase
FIREBASE_CRED_JSON = "serviceAccountKey.json"
FIREBASE_DB_URL = "https://violence-detection---comvis-default-rtdb.asia-southeast1.firebasedatabase.app/"

# Model paths
YOLO_WEIGHTS = "yolo11n.pt"
VIOLENCE_MODEL_PATH = "violence_model.pth"

# Runtime
SEQ_LENGTH = 16
CONFIDENCE_THRESHOLD = 0.5   
COOLDOWN_SEC = 5
USE_FIREBASE = True 

# =========================
# 🧠 Model Conv1d + GRU
# =========================
class ModelConvGRU(nn.Module):
    def __init__(self, input_size=3, conv_channels=16, hidden_size=64, num_layers=2, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=conv_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)        
        x = self.conv1(x)             
        x = self.relu(x)
        x = x.permute(0, 2, 1)       
        out, _ = self.gru(x)           
        out = out[:, -1, :]            
        return self.fc(out)          

# =========================
# 📲 Notifikasi
# =========================
def send_alert_telegram(message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=5)
    except Exception as e:
        print("[WARN] Gagal kirim Telegram:", e)

def init_firebase():
    if not USE_FIREBASE:
        return
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED_JSON)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    except Exception as e:
        print("[WARN] Firebase init gagal:", e)

def send_to_firebase(status, camera="Camera 1"):
    if not USE_FIREBASE:
        return
    try:
        ref = db.reference("violence_alerts")
        ref.push({
            "status": status,
            "camera": camera,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        print("[WARN] Firebase push gagal:", e)

# =========================
# 📦 Utilitas fitur
# =========================
def iou(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union

def compute_overlap_count(boxes, thr=0.1):
    n = len(boxes)
    if n < 2:
        return 0.0
    cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            if iou(boxes[i], boxes[j]) > thr:
                cnt += 1
    return float(cnt)

def compute_avg_speed(prev_boxes, curr_boxes):
    # speed rata2 centroid antar frame dgn asosiasi greedy via IoU tertinggi
    if not prev_boxes or not curr_boxes:
        return 0.0
    used = set()
    speeds = []
    for pb in prev_boxes:
        best_j, best_iou = -1, 0.0
        for j, cb in enumerate(curr_boxes):
            if j in used:
                continue
            iou_v = iou(pb, cb)
            if iou_v > best_iou:
                best_iou, best_j = iou_v, j
        if best_j >= 0:
            used.add(best_j)
            # centroid distance
            pcx = (pb[0]+pb[2]) / 2.0; pcy = (pb[1]+pb[3]) / 2.0
            ccx = (curr_boxes[best_j][0]+curr_boxes[best_j][2]) / 2.0
            ccy = (curr_boxes[best_j][1]+curr_boxes[best_j][3]) / 2.0
            d = np.hypot(ccx - pcx, ccy - pcy)
            speeds.append(d)
    if not speeds:
        return 0.0
    return float(np.mean(speeds))

# =========================
# Main Realtime
# =========================
def main():
    device = torch.device("cpu")
    init_firebase()

    # Load YOLO
    print("[INFO] Load YOLO...")
    yolo = YOLO(YOLO_WEIGHTS)

    # Load Model Kekerasan
    print("[INFO] Load model kekerasan...")
    model = ModelConvGRU(input_size=3, conv_channels=16, hidden_size=64, num_layers=2, num_classes=2)
    state = torch.load(VIOLENCE_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Buffer fitur (seq_len, 3)
    feat_window = deque(maxlen=SEQ_LENGTH)

    # Track boxes untuk speed
    prev_boxes = []
    last_alert_time = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka kamera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi orang
        res = yolo(frame, imgsz=640, conf=0.5, verbose=False)[0]
        boxes = []
        person_count = 0
        for b in res.boxes:
            cls_id = int(b.cls[0].item())
            if cls_id == 0:  # person
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])
                person_count += 1
                # opsional: gambar bbox (tidak mengubah label text)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Fitur agregat per-frame
        avg_speed = compute_avg_speed(prev_boxes, boxes)
        overlap = compute_overlap_count(boxes, thr=0.1)
        prev_boxes = boxes

        feat_window.append([float(person_count), float(avg_speed), float(overlap)])

        # Default tampil "NonFight" supaya tidak ada embel-embel lain saat belum cukup sequence
        label_txt = "NonFight"

        # Jika sequence cukup → prediksi
        if len(feat_window) == SEQ_LENGTH:
            x = torch.tensor([list(feat_window)], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model(x)
                pred = int(torch.argmax(logits, dim=1).item())

            label_txt = "Fight" if pred == 1 else "NonFight"

            # Kirim alert + Firebase bila Fight (pakai cooldown)
            if label_txt == "Fight":
             now = time.time()
             if now - last_alert_time >= COOLDOWN_SEC:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ALERT] Kekerasan terdeteksi!")
                send_alert_telegram("⚠️ Violence Detected on Camera 1!")
                send_to_firebase("Fight Detected", camera="Camera 1")
                last_alert_time = now


        # Tampilkan label
        # Fight → merah, NonFight → hijau
        color = (0, 0, 255) if label_txt == "Fight" else (0, 255, 0)
        cv2.putText(frame, label_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # cv2.imshow("Deteksi Kekerasan Realtime", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        # pass

    cap.release()
    # cv2.destroyAllWindows()

# =========================
# 🚀 Run
# =========================
if __name__ == "__main__":
    main()
