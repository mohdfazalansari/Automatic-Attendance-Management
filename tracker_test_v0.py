#.\attendance_env\Scripts\activate
import cv2
import numpy as np
import sqlite3
import json
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from deep_sort_realtime.deepsort_tracker import DeepSort


print("Loading Models & Database...")
detector = MTCNN()
embedder = FaceNet()
tracker = DeepSort(max_age=30, n_init=3) # Initialize Deep SORT

# Load Known Students from Database
conn = sqlite3.connect('attendance_system.db')
cursor = conn.cursor()
cursor.execute("SELECT name, embedding FROM students")
known_students = [(row[0], np.array(json.loads(row[1]))) for row in cursor.fetchall()]
conn.close()

# The Dictionary Cache
track_to_name = {}

# Initialize Local Webcam
cap = cv2.VideoCapture(0)
print("Starting live tracking... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Downscale for performance
    frame = cv2.resize(frame, (640, 480))
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. Face Detection
    results = detector.detect_faces(img_rgb)
    
    # Format detections for Deep SORT: ([left, top, w, h], confidence, detection_class)
    bbs = []
    for result in results:
        x, y, w, h = result['box']
        confidence = result['confidence']
        # Ensure no negative coordinates
        x, y = max(0, x), max(0, y)
        bbs.append(([x, y, w, h], confidence, "face"))
        
    # 2. Update Deep SORT Tracks
    tracks = tracker.update_tracks(bbs, frame=frame)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb() # Left, Top, Right, Bottom bounding box
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        
        # Ensure coordinates are within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # 3. The Bypass Logic
        if track_id not in track_to_name:
            # We don't know who this is yet, run FaceNet
            face_crop = img_rgb[y1:y2, x1:x2]
            
            if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                face_crop = cv2.resize(face_crop, (160, 160))
                face_crop = np.expand_dims(face_crop, axis=0)
                
                live_embedding = embedder.embeddings(face_crop)[0]
                
                best_match_name = "Unknown"
                lowest_distance = 1.0
                strict_threshold = 0.40 
                
                for name, db_embedding in known_students:
                    distance = cosine(live_embedding, db_embedding)
                    if distance < lowest_distance and distance < strict_threshold:
                        lowest_distance = distance
                        best_match_name = name
                
                # Save the identity to the dictionary cache
                track_to_name[track_id] = best_match_name
        
        # Pull the name from the cache
        display_name = track_to_name.get(track_id, "Unknown")
        
        # Draw Bounding Box and Name
        color = (0, 255, 0) if display_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id} {display_name}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Live Deep SORT Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()