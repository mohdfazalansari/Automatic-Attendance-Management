import cv2
import numpy as np
import sqlite3
import json
import time
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from deep_sort_realtime.deepsort_tracker import DeepSort

print("Loading Models & Database...")

# 1. Initialize Models
detector = MTCNN() 
embedder = FaceNet()

# UPGRADE: n_init=1 makes boxes appear instantly. max_iou_distance=0.7 handles fast movement.
tracker = DeepSort(max_age=30, n_init=1, max_iou_distance=0.7)

# 2. Load Known Students
conn = sqlite3.connect('attendance_system.db')
cursor = conn.cursor()
cursor.execute("SELECT name, embedding FROM students")
known_students = [(row[0], np.array(json.loads(row[1]))) for row in cursor.fetchall()]
conn.close()

# 3. Cache & Config
track_to_name = {}
track_retries = {}
MAX_RETRIES = 5
MIN_FACE_WIDTH = 40      # Lowered to accept smaller/distant faces
STRICT_THRESHOLD = 0.35  

# 4. Initialize Camera
cap = cv2.VideoCapture(0)
print("Starting live tracking... Press 'q' to quit.")

# FPS Calculation Variables
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Check webcam connection.")
        break
        
    # Resize and convert color
    frame = cv2.resize(frame, (640, 480))
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    # 5. Detect ALL Faces
    results = detector.detect_faces(img_rgb)
    bbs = []
    
    for result in results:
        x, y, w, h = result['box']
        confidence = result['confidence']
        
        # UPGRADE: Ultra-low confidence threshold to combat backlighting
        if confidence > 0.50: 
            # Ensure coordinates don't go negative
            x, y = max(0, x), max(0, y)
            bbs.append(([x, y, w, h], confidence, "face"))
            
    # 6. Update Tracker with ALL detected faces
    tracks = tracker.update_tracks(bbs, frame=frame)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        
        # Safely extract and bound coordinates to frame size
        x1, y1 = max(0, int(ltrb[0])), max(0, int(ltrb[1]))
        x2, y2 = min(frame.shape[1], int(ltrb[2])), min(frame.shape[0], int(ltrb[3]))
        
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Skip if bounding box is invalid/collapsed
        if box_width <= 0 or box_height <= 0:
            continue

        # 7. Recognition Logic
        needs_recognition = (
            track_id not in track_to_name or 
            (track_to_name[track_id] == "Unknown" and track_retries.get(track_id, 0) < MAX_RETRIES)
        )
        
        if needs_recognition and box_width >= MIN_FACE_WIDTH:
            
            face_crop = img_rgb[y1:y2, x1:x2]
            
            # Double check crop validity
            if face_crop.size > 0:
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
                    
                    if best_match_name != "Unknown":
                        track_to_name[track_id] = best_match_name
                    else:
                        track_retries[track_id] = track_retries.get(track_id, 0) + 1
                        if track_retries[track_id] >= MAX_RETRIES:
                            track_to_name[track_id] = "Unknown"
                except Exception as e:
                    print(f"Error processing face crop: {e}")
                    
        elif needs_recognition and box_width < MIN_FACE_WIDTH:
            track_to_name[track_id] = "Approaching..."

        # 8. Draw Graphics
        display_name = track_to_name.get(track_id, "Unknown")
        color = (0, 255, 0) if display_name not in ["Unknown", "Approaching..."] else (0, 165, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id} {display_name}", (x1, max(20, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw FPS Counter
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Live Deep SORT Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()