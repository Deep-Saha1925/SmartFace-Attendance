# Face Recognition Attendance System (Webcam Based)
import cv2
import face_recognition
import pickle
import numpy as np
import csv
from datetime import datetime, timedelta
import os
from db import get_connection
from config import ATTENDANCE_ELAPSED_MINUTES

# ------------------- Configuration -------------------
ENCODE_FILE = "EncodeFile.p"
CSV_FILE = "students.csv"

# ------------------- Load Encodings -------------------
if not os.path.exists(ENCODE_FILE):
    raise FileNotFoundError("EncodeFile.p not found. Please run encoding script first.")

with open(ENCODE_FILE, "rb") as f:
    encodeListKnown, studentIds = pickle.load(f)
print(f"[INFO] Loaded {len(encodeListKnown)} encodings successfully.")

# ------------------- Helper Functions -------------------
def get_student_info(student_id):
    """Fetch student info from MySQL; fallback to CSV."""
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students WHERE id = %s", (student_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            return row
    except Exception as e:
        print("[WARN] DB fetch failed, falling back to CSV:", e)

    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["id"] == str(student_id):
                    return row
    return None


def update_attendance(student_id):
    """Update attendance in MySQL and CSV with elapsed time check."""
    updated_row = None
    now = datetime.now()

    threshold = timedelta(minutes=ATTENDANCE_ELAPSED_MINUTES)

    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students WHERE id = %s", (student_id,))
        row = cursor.fetchone()

        if row:
            last_time = datetime.strptime(str(row["last_attendance_time"]), "%Y-%m-%d %H:%M:%S")
            if now - last_time >= threshold:
                new_total = int(row["total_attendance"]) + 1
                cursor.execute("""
                    UPDATE students 
                    SET total_attendance = %s, last_attendance_time = %s 
                    WHERE id = %s
                """, (new_total, now.strftime("%Y-%m-%d %H:%M:%S"), student_id))
                conn.commit()
                print(f"[DB] Attendance updated for ID {student_id}")
            else:
                print(f"[SKIP] ID {student_id} marked too recently. Waiting for next interval.")
            updated_row = row
        cursor.close()
        conn.close()
    except Exception as e:
        print("[WARN] DB update failed, falling back to CSV:", e)

    # ------------------- CSV Backup Update -------------------
    if os.path.exists(CSV_FILE):
        rows = []
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["id"] == str(student_id):
                    last_time = datetime.strptime(row["last_attendance_time"].strip(), "%Y-%m-%d %H:%M:%S")
                    if now - last_time >= threshold:
                        row["total_attendance"] = str(int(row["total_attendance"]) + 1)
                        row["last_attendance_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[CSV] Attendance updated for ID {student_id}")
                    else:
                        print(f"[CSV-SKIP] ID {student_id} recently marked.")
                    updated_row = row
                rows.append(row)
        if updated_row:
            with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    return updated_row

cap = cv2.VideoCapture(0)
print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        continue

    # Resize for faster recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect and encode faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
        match_index = np.argmin(face_distances)

        if matches[match_index]:
            student_id = studentIds[match_index]
            student_info = get_student_info(student_id)
            name = student_info["name"] if student_info else "Unknown"

            update_attendance(student_id)

            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam closed.")