# This file is for taking attendance through web came

import cv2
import face_recognition
import pickle
import numpy as np
import csv
from datetime import datetime
import os

# Load encodings
ENCODE_FILE = "EncodeFile.p"
CSV_FILE = "students.csv"
with open(ENCODE_FILE, "rb") as f:
    encodeListKnown, studentIds = pickle.load(f)

def get_student_info(student_id):
    if not os.path.exists(CSV_FILE):
        return None
    with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] == str(student_id):
                return row
    return None

def update_attendance(student_id):
    if not os.path.exists(CSV_FILE):
        return None, 0
    rows = []
    updated_row = None
    secondsElapsed = 0
    with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if row["id"] == str(student_id):
                last_time = datetime.strptime(row["last_attendance_time"].strip(), "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - last_time).total_seconds()
                if secondsElapsed > 80:
                    row["total_attendance"] = str(int(row["total_attendance"]) + 1)
                    row["last_attendance_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                updated_row = row
    if updated_row:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return updated_row, secondsElapsed

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        continue

    # Resize frame for faster recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
        match_index = np.argmin(face_distances)

        if matches[match_index]:
            student_id = studentIds[match_index]
            studentInfo, elapsed = update_attendance(student_id)
            name = studentInfo["name"] if studentInfo else "Unknown"

            # Draw rectangle
            top, right, bottom, left = [v*4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()