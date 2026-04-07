import cv2
import threading
import time
import os
import collections
from datetime import datetime, timedelta # ⏱️ NEW: Added timedelta for 7-day math
from ultralytics import YOLO
from deepface import DeepFace

# ==========================================
# ⚙️ SYSTEM CONFIGURATION
# ==========================================
SOURCE = 1              # 1 is your computer's camera (640x480 @ 30fps) ✓ TESTED
MODEL_VERSION = 'yolov8m.pt'  # The default medium brain
EVIDENCE_FOLDER = "evidence"
LOG_FILE = "security_log.txt"

if not os.path.exists(EVIDENCE_FOLDER):
    os.makedirs(EVIDENCE_FOLDER)

# ==========================================
# 🛑 MAXIMUM SECURITY CLASSIFICATION
# ==========================================
# Tier 2: The Optimized Threat List (No Waste)
IMPROVISED_WEAPONS = [34, 38, 39, 40, 41, 42, 43, 44, 65, 66, 75, 76]

DANGEROUS_EMOTIONS = ["angry", "fear", "disgust"]

# ==========================================
# 🧠 MULTI-PERSON TRACKING + EMOTION QUEUE
# ==========================================
# Tracks store per-person state so we can identify repeat visitors and
# only run DeepFace when a tracked person is carrying a weapon.
tracks = {}  # track_id -> state dict
next_track_id = 0

# A small queue of face crops waiting for emotion analysis.
# Only weapon carriers are enqueued.
face_crop_queue = collections.deque(maxlen=10)

# Threading + synchronization
lock = threading.Lock()
running = True

# Global display state (most severe emotion among weapon carriers)
current_emotion = "scanning..."

# Tracking settings
MAX_MISSED_FRAMES = 15               # remove people not seen for this many frames
EMOTION_UPDATE_INTERVAL = 1        # seconds between emotion updates for a track

# Anti-Flicker Filter for Weapons
weapon_counter = 0
TRIGGER_THRESHOLD = 5   # Require 5 consecutive frames to trigger alarm
COOLDOWN = 0

# ==========================================
# 📝 LOGGING, AUDIO, & AUTO-DELETER
# ==========================================
def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

# 🚨 Audio Alarm Function
def sound_alarm():
    """Plays a loud warning beep without freezing the video feed."""
    try:
        import winsound
        winsound.Beep(2500, 1000)
    except ImportError:
        print('\a')

# 🧹 7-DAY ROLLING AUTO-DELETER
def auto_clean_storage(days_to_keep=7):
    """Deletes any images and log entries older than 7 days."""
    cutoff_datetime = datetime.now() - timedelta(days=days_to_keep)
    cutoff_timestamp = time.time() - (days_to_keep * 24 * 60 * 60)

    # 1. Clean up old images
    try:
        files = [os.path.join(EVIDENCE_FOLDER, f) for f in os.listdir(EVIDENCE_FOLDER) if f.endswith('.jpg')]
        for file_path in files:
            if os.path.getmtime(file_path) < cutoff_timestamp:
                os.remove(file_path)
                print(f"🧹 Auto-Deleter: Removed expired evidence -> {file_path}")
    except Exception:
        pass

    # 2. Clean up old text logs
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()
            kept_lines = []
            for line in lines:
                try:
                    date_str = line[1:20]
                    log_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    if log_date >= cutoff_datetime:
                        kept_lines.append(line)
                except ValueError:
                    kept_lines.append(line)
            with open(LOG_FILE, 'w') as f:
                f.writelines(kept_lines)
    except Exception:
        pass

# ==========================================
# 🚧 UTILITY HELPERS
# ==========================================
def iou(boxA, boxB):
    """Intersection over Union for two boxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def expand_box(box, pad, frame_shape):
    """Expand a box by pad pixels while staying in frame bounds."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return (x1, y1, x2, y2)


def extract_face_crop(frame, person_box, pad=30):
    """Crop a face region inside a person bounding box (or return the person crop)."""
    px1, py1, px2, py2 = expand_box(person_box, pad, frame.shape)
    crop = frame[py1:py2, px1:px2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        fx1 = max(0, fx - pad)
        fy1 = max(0, fy - pad)
        fx2 = min(crop.shape[1], fx + fw + pad)
        fy2 = min(crop.shape[0], fy + fh + pad)
        return crop[fy1:fy2, fx1:fx2]

    return crop


def update_tracks(person_boxes):
    """Match detections to existing tracks and create new ones."""
    global next_track_id

    # Reset match flag for all existing tracks
    for t in tracks.values():
        t["matched"] = False

    # Try to associate each detection with an existing track
    for box in person_boxes:
        best_id = None
        best_iou = 0.0
        for tid, t in tracks.items():
            i = iou(box, t["bbox"])
            if i > best_iou:
                best_iou = i
                best_id = tid

        if best_id is not None and best_iou > 0.25:
            # update existing track
            tracks[best_id]["bbox"] = box
            tracks[best_id]["missed"] = 0
            tracks[best_id]["last_seen"] = time.time()
            tracks[best_id]["matched"] = True
        else:
            # new track
            tracks[next_track_id] = {
                "bbox": box,
                "missed": 0,
                "last_seen": time.time(),
                "has_weapon": False,
                "weapon_names": [],
                "current_emotion": "scanning...",
                "emotion_history": collections.deque(maxlen=7),
                "last_emotion_time": 0,
                "proximity": "far",
                "matched": True,
            }
            next_track_id += 1

    # Remove old tracks
    to_remove = [tid for tid, t in tracks.items() if t.get("missed", 0) > MAX_MISSED_FRAMES]
    for tid in to_remove:
        del tracks[tid]

    # Increment missed counter for unmatched tracks
    for t in tracks.values():
        if not t.get("matched"):
            t["missed"] += 1


def compute_global_emotion():
    """Return the most severe emotion among weapon carriers."""
    candidate = "scanning..."

    for t in tracks.values():
        if not t.get("has_weapon"):
            continue
        em = t.get("current_emotion", "scanning...")
        if em in DANGEROUS_EMOTIONS:
            return em
        if candidate == "scanning..." and em not in (None, ""):
            candidate = em

    return candidate


# ==========================================
# 🧵 BACKGROUND THREAD: EMOTION AI
# ==========================================
def emotion_worker():
    global current_emotion

    while running:
        item = None
        with lock:
            if face_crop_queue:
                item = face_crop_queue.popleft()

        if item is None:
            time.sleep(0.05)
            continue

        track_id, img = item
        try:
            # 💡 THE AUTO-LIGHTING HACK (CLAHE)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_face = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

            objs = DeepFace.analyze(enhanced_face, actions=['emotion'], enforce_detection=False, silent=True)

            # 🛠️ THE RAW SCORE HACK (Lowered Threshold)
            emotion_scores = objs[0]['emotion']
            if emotion_scores['angry'] > 25.0:
                detected_emotion = "angry"
            elif emotion_scores['fear'] > 25.0:
                detected_emotion = "fear"
            elif emotion_scores['disgust'] > 20.0:
                detected_emotion = "disgust"
            else:
                detected_emotion = objs[0]['dominant_emotion']

            with lock:
                track = tracks.get(track_id)
                if track is not None:
                    track['emotion_history'].append(detected_emotion)
                    track['current_emotion'] = collections.Counter(track['emotion_history']).most_common(1)[0][0]
                    track['last_emotion_time'] = time.time()

        except Exception:
            pass

        with lock:
            current_emotion = compute_global_emotion()


# ==========================================
# 🚀 MAIN SECURITY ENGINE
# ==========================================
print(f"Loading AI Brain ({MODEL_VERSION})...")
weapon_model = YOLO(MODEL_VERSION)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

PERSON_CLASS_ID = 0
if 'person' in weapon_model.names:
    PERSON_CLASS_ID = weapon_model.names.index('person')

# Start the emotion worker thread
t = threading.Thread(target=emotion_worker)
t.daemon = True
t.start()

cap = cv2.VideoCapture(SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("System ARMED and Monitoring...")
log_event("SYSTEM STARTED - MONITORING ACTIVE")

# Run the cleaner once right when the system turns on
auto_clean_storage()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image

    # ---------------------------------------------------------
    # A. OBJECT DETECTION (People + Weapons)
    # ---------------------------------------------------------
    results = weapon_model(frame, verbose=False, conf=0.45)

    person_boxes = []
    weapon_detections = []  # (cls_id, x1, y1, x2, y2)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id == PERSON_CLASS_ID:
                person_boxes.append((x1, y1, x2, y2))
            elif cls_id in IMPROVISED_WEAPONS:
                weapon_detections.append((cls_id, x1, y1, x2, y2))

    with lock:
        update_tracks(person_boxes)
        for t in tracks.values():
            t['has_weapon'] = False
            t['weapon_names'] = []

        # Assign weapons to the nearest person
        for cls_id, wx1, wy1, wx2, wy2 in weapon_detections:
            best_tid = None
            best_iou = 0.0
            for tid, t in tracks.items():
                i = iou((wx1, wy1, wx2, wy2), t['bbox'])
                if i > best_iou:
                    best_iou = i
                    best_tid = tid

            if best_tid is not None and best_iou > 0.05:
                t = tracks[best_tid]
                t['has_weapon'] = True
                t['weapon_names'].append(weapon_model.names[cls_id].upper())

        now = time.time()
        for tid, t in tracks.items():
            x1, y1, x2, y2 = t['bbox']
            height = y2 - y1

            if height > 180:
                t['proximity'] = 'close'
            elif height > 80:
                t['proximity'] = 'medium'
            else:
                t['proximity'] = 'far'

            is_close = t.get('proximity') == 'close'
            if (t.get('has_weapon') or is_close) and (now - t.get('last_emotion_time', 0)) > EMOTION_UPDATE_INTERVAL:
                crop = extract_face_crop(frame, t['bbox'])
                if crop is not None:
                    face_crop_queue.append((tid, crop))

    # Compute global state
    with lock:
        current_emotion = compute_global_emotion()
        weapon_carriers = [t for t in tracks.values() if t.get('has_weapon')]
        any_proximity_alert = any(t.get('proximity') == 'close' for t in weapon_carriers)

    if weapon_detections:
        weapon_counter += 1
    else:
        weapon_counter = 0

    # ---------------------------------------------------------
    # C. THE TIERED THREAT LOGIC (The Brain)
    # ---------------------------------------------------------
    system_status = "SAFE"
    status_color = (0, 255, 0)
    weapon_box_color = (0, 165, 255)  # Default weapons to Orange

    weapon_names = set()
    with lock:
        for t in weapon_carriers:
            weapon_names.update(t.get('weapon_names', []))

    weapon_names_str = ", ".join(sorted(weapon_names))

    if weapon_counter > TRIGGER_THRESHOLD:
        if any_proximity_alert or current_emotion in DANGEROUS_EMOTIONS:
            system_status = "!!! CRITICAL THREAT: ARMED & HOSTILE !!!"
            status_color = (0, 0, 255)  # RED
            weapon_box_color = (0, 0, 255)  # 🩸 Force weapon boxes to RED

            if COOLDOWN == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{EVIDENCE_FOLDER}/ATTACK_{timestamp}.jpg"
                cv2.imwrite(filename, frame)

                log_event(f"CRITICAL: Armed Hostile ({weapon_names_str}). Emotion: {current_emotion}")
                print(f"📸 EVIDENCE SAVED: {filename}")

                threading.Thread(target=sound_alarm, daemon=True).start()
                threading.Thread(target=auto_clean_storage, daemon=True).start()

                COOLDOWN = 30

        else:
            system_status = "CAUTION: Dual-Use Item Detected"
            status_color = (0, 165, 255)  # ORANGE

    elif current_emotion in DANGEROUS_EMOTIONS and any_proximity_alert:
        system_status = "WARNING: Aggressive Behavior"
        status_color = (0, 255, 255)  # YELLOW

    if COOLDOWN > 0:
        COOLDOWN -= 1

    # ---------------------------------------------------------
    # D. DISPLAY DASHBOARD & WEAPONS
    # ---------------------------------------------------------
    with lock:
        for tid, t in tracks.items():
            x1, y1, x2, y2 = t['bbox']
            color = (0, 255, 0)
            label = f"ID {tid}"
            if t.get('has_weapon'):
                color = (0, 0, 255)
                label += " (WEAPON)"
            if t.get('current_emotion'):
                label += f" | {t['current_emotion']}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 🖌️ Draw the weapons with names and dynamic colors
    for cls_id, x1, y1, x2, y2 in weapon_detections:
        item_name = weapon_model.names[cls_id].upper()
        cv2.rectangle(frame, (x1, y1), (x2, y2), weapon_box_color, 2)
        cv2.putText(frame, item_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, weapon_box_color, 2)

    # Dashboard Banner
    cv2.rectangle(frame, (0, 0), (1280, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"STATUS: {system_status}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow("Security Core - Improvised Threat Mode", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_event("SYSTEM SHUTDOWN")
running = False
cap.release()
cv2.destroyAllWindows()