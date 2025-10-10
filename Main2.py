# Face Recognition Attendance System (Static Image Based)
import cv2
import face_recognition
import pickle
import numpy as np
import csv
from datetime import datetime
import os
from db import get_connection
from config import ATTENDANCE_ELAPSED_MINUTES

ENCODE_FILE = "EncodeFile.p"
CSV_FILE = "students.csv"
PHOTO_FILE = "./TEST/test_photo.jpg"

if not os.path.exists(ENCODE_FILE):
    raise FileNotFoundError("EncodeFile2.p not found. Please run encoding script first.")

with open(ENCODE_FILE, "rb") as f:
    encodeListKnown, studentIds = pickle.load(f)
print("Encodings loaded successfully.")

def get_student_info(student_id):
    """Fetch student info from DB, fallback to CSV."""
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
        print("DB fetch failed, using CSV:", e)

    # Fallback to CSV
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["id"] == str(student_id):
                    return row
    return None


def update_attendance(student_id):
    """Update attendance in both MySQL and CSV."""
    updated_row = None
    secondsElapsed = 0
    now = datetime.now()

    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM students WHERE id = %s", (student_id,))
        row = cursor.fetchone()

        if row:
            last_time = datetime.strptime(str(row["last_attendance_time"]), "%Y-%m-%d %H:%M:%S")
            secondsElapsed = (now - last_time).total_seconds()
            minutesElapsed = secondsElapsed / 60

            if minutesElapsed > ATTENDANCE_ELAPSED_MINUTES:
                new_total = int(row["total_attendance"]) + 1
                cursor.execute("""
                    UPDATE students 
                    SET total_attendance = %s, last_attendance_time = %s 
                    WHERE id = %s
                """, (new_total, now.strftime("%Y-%m-%d %H:%M:%S"), student_id))
                conn.commit()
                print(f"✅ Attendance updated in DB for ID {student_id}")
            updated_row = row
        cursor.close()
        conn.close()
    except Exception as e:
        print("⚠️ DB update failed, falling back to CSV:", e)

    # --- CSV Update (Backup) ---
    if os.path.exists(CSV_FILE):
        rows = []
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                if row["id"] == str(student_id):
                    last_time = datetime.strptime(row["last_attendance_time"].strip(), "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (now - last_time).total_seconds()
                    minutesElapsed = secondsElapsed / 60
                    if minutesElapsed > ATTENDANCE_ELAPSED_MINUTES:
                        row["total_attendance"] = str(int(row["total_attendance"]) + 1)
                        row["last_attendance_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
                    updated_row = row
        if updated_row:
            with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    return updated_row, secondsElapsed

img = cv2.imread(PHOTO_FILE)
if img is None:
    print(f"⚠️ Could not read image: {PHOTO_FILE}")
    exit()

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect all faces
face_locations = face_recognition.face_locations(rgb_img)
face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

if len(face_locations) == 0:
    print("⚠️ No faces detected in the image.")

for face_encoding, face_location in zip(face_encodings, face_locations):
    matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
    face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
    if len(face_distances) == 0:
        continue

    match_index = np.argmin(face_distances)

    if matches[match_index]:
        student_id = studentIds[match_index]
        studentInfo, elapsed = update_attendance(student_id)
        name = studentInfo["name"] if studentInfo else "Unknown"
        print(f"✅ Attendance marked for: {name} (ID: {student_id})")

        # Draw rectangle
        top, right, bottom, left = face_location
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        top, right, bottom, left = face_location
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(img, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

cv2.imshow("Face Recognition Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()