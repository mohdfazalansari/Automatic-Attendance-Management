import cv2
import numpy as np
import sqlite3
import json
import time
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==========================================
# 1. CONFIGURATION & THRESHOLDS
# ==========================================
MAX_RETRIES = 5
MIN_FACE_WIDTH = 40
STRICT_THRESHOLD = 0.35
FRAME_SKIP = 2  

# MASTER TIME CONFIGURATION
TOTAL_CLASS_TIME_SECONDS = 60 # 1 minute
REQUIRED_ATTENDANCE_SECONDS = TOTAL_CLASS_TIME_SECONDS * 0.80  

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_database():
    try:
        conn = sqlite3.connect('attendance_system.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name, embedding FROM students")
        data = [(row[0], np.array(json.loads(row[1]))) for row in cursor.fetchall()]
        conn.close()
        return data
    except Exception as e:
        print(f"Database Error: {e}")
        return []

def recognize_student(face_crop, known_students, embedder):
    try:
        face_crop = cv2.resize(face_crop, (160, 160))
        face_crop = np.expand_dims(face_crop, axis=0)
        live_embedding = embedder.embeddings(face_crop)[0]
        
        best_match_name = "Unknown"
        lowest_distance = 1.0
        
        for name, db_embedding in known_students:
            distance = cosine(live_embedding, db_embedding)
            if distance < lowest_distance and distance < STRICT_THRESHOLD:
                lowest_distance = distance
                best_match_name = name
                
        return best_match_name
    except Exception as e:
        print(f"Recognition Error: {e}")
        return "Unknown"

# ==========================================
# 3. INITIALIZATION
# ==========================================
print("Loading Models & Database (Local V2)...")
detector = MTCNN() 
embedder = FaceNet()
tracker = DeepSort(max_age=30, n_init=1, max_iou_distance=0.7)
known_students = load_database()

if not known_students:
    print("WARNING: No students found in database!")

track_to_name = {}
track_retries = {}
attendance_log = {} 

# ==========================================
# 4. MAIN VIDEO LOOP
# ==========================================
cap = cv2.VideoCapture(0)
print("Camera initialized. Waiting for teacher to start class...")

prev_frame_time = 0
frame_count = 0

# --- NEW STATE VARIABLES ---
class_started = False
class_start_time = 0
time_left = TOTAL_CLASS_TIME_SECONDS

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab camera frame.")
        break
        
    frame_count += 1
    
    if frame_count % FRAME_SKIP != 0:
        continue
        
    frame = cv2.resize(frame, (640, 480))
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    current_time = time.time()
    time_diff = current_time - prev_frame_time
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_frame_time = current_time
    
    # --- MASTER TIMER LOGIC ---
    if class_started:
        elapsed_class_time = current_time - class_start_time
        time_left = int(TOTAL_CLASS_TIME_SECONDS - elapsed_class_time)
        
        # --- AUTO-END FEATURE ---
        if time_left <= 0:
            print("\n[INFO] Time is up! Automatically ending class...")
            time_left = 0
            break # This instantly breaks the while loop and jumps to the report!
    else:
        time_left = TOTAL_CLASS_TIME_SECONDS
        
    # --- A. DETECT FACES ---
    results = detector.detect_faces(img_rgb)
    bbs = []
    for result in results:
        x, y, w, h = result['box']
        if result['confidence'] > 0.50: 
            bbs.append(([max(0, x), max(0, y), w, h], result['confidence'], "face"))
            
    # --- B. UPDATE TRACKER ---
    tracks = tracker.update_tracks(bbs, frame=frame)
    
    # --- C. RECOGNIZE & ATTENDANCE LOGIC ---
    active_students_this_frame = [] 
    
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1 = max(0, int(ltrb[0])), max(0, int(ltrb[1]))
        x2, y2 = min(frame.shape[1], int(ltrb[2])), min(frame.shape[0], int(ltrb[3]))
        box_width, box_height = x2 - x1, y2 - y1
        
        if box_width <= 0 or box_height <= 0:
            continue

        needs_recognition = (
            track_id not in track_to_name or 
            (track_to_name[track_id] == "Unknown" and track_retries.get(track_id, 0) < MAX_RETRIES)
        )
        
        if needs_recognition:
            if box_width >= MIN_FACE_WIDTH:
                face_crop = img_rgb[y1:y2, x1:x2]
                if face_crop.size > 0:
                    best_name = recognize_student(face_crop, known_students, embedder)
                    
                    if best_name != "Unknown":
                        track_to_name[track_id] = best_name
                    else:
                        track_retries[track_id] = track_retries.get(track_id, 0) + 1
                        if track_retries[track_id] >= MAX_RETRIES:
                            track_to_name[track_id] = "Unknown"
            else:
                track_to_name[track_id] = "Approaching..."

        display_name = track_to_name.get(track_id, "Unknown")
        
        # ---------------------------------------------------------
        # ATTENDANCE ENTRY LOGIC (Only logs if class started!)
        # ---------------------------------------------------------
        if display_name not in ["Unknown", "Approaching..."]:
            active_students_this_frame.append(display_name)
            
            if class_started:
                if display_name not in attendance_log:
                    attendance_log[display_name] = {"entry_time": current_time, "total_time": 0.0}
                    print(f"[{time.strftime('%H:%M:%S')}] {display_name} entered the frame.")
                    
                elif attendance_log[display_name]["entry_time"] is None:
                    attendance_log[display_name]["entry_time"] = current_time
                    print(f"[{time.strftime('%H:%M:%S')}] {display_name} returned to the frame.")
                    
                current_session_time = current_time - attendance_log[display_name]["entry_time"]
                live_total_time = attendance_log[display_name]["total_time"] + current_session_time
                display_text = f"ID:{track_id} {display_name} ({int(live_total_time)}s)"
            else:
                # If class hasn't started, don't accumulate time
                display_text = f"ID:{track_id} {display_name} (Waiting...)"
        else:
            display_text = f"ID:{track_id} {display_name}"

        color = (0, 255, 0) if display_name not in ["Unknown", "Approaching..."] else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, display_text, (x1, max(20, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---------------------------------------------------------
    # ATTENDANCE EXIT LOGIC (Only triggers if class started!)
    # ---------------------------------------------------------
    if class_started:
        for student in attendance_log:
            if student not in active_students_this_frame and attendance_log[student]["entry_time"] is not None:
                time_spent = current_time - attendance_log[student]["entry_time"]
                attendance_log[student]["total_time"] += time_spent
                attendance_log[student]["entry_time"] = None
                print(f"[{time.strftime('%H:%M:%S')}] {student} left. Total Time so far: {int(attendance_log[student]['total_time'])} seconds.")

    # =========================================================
    # ON-SCREEN UI DRAWING
    # =========================================================
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    frame_width = frame.shape[1]
    
    # --- START PROMPT UI ---
    if not class_started:
        # Blinking effect for the "Start" prompt
        if int(current_time * 2) % 2 == 0:
            cv2.putText(frame, "Press 's' to START Class", (frame_width // 2 - 130, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    timer_color = (0, 255, 0) if time_left > 30 else (0, 0, 255) 
    cv2.putText(frame, f"Time Left: {time_left}s", (frame_width - 180, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)

    cv2.imshow("Model V2 - Local Tracker", frame)
    
    # --- KEYPRESS LOGIC ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and not class_started:
        class_started = True
        class_start_time = time.time()
        print(f"\n[INFO] Class started at {time.strftime('%H:%M:%S')}!")

# ==========================================
# FINAL ATTENDANCE PRINT OUT & LOGIC
# ==========================================
print("\n" + "="*50)
print("FINAL ATTENDANCE REPORT")
print("="*50)

for student, data in attendance_log.items():
    if data["entry_time"] is not None:
        data["total_time"] += (time.time() - data["entry_time"])
        
    total_seconds = int(data['total_time'])
    
    if total_seconds >= REQUIRED_ATTENDANCE_SECONDS:
        status = "PRESENT"
    else:
        status = "ABSENT"
        
    print(f"{student.ljust(15)} | Time: {str(total_seconds).rjust(3)}s / {int(TOTAL_CLASS_TIME_SECONDS)}s | Status: {status}")
print("="*50)

cap.release()
cv2.destroyAllWindows()