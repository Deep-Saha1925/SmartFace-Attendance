# This file is for taking attendance through static images

import cv2
import face_recognition
import pickle
import numpy as np
import csv
from datetime import datetime
import os

ENCODE_FILE = "EncodeFile2.p"
CSV_FILE = "students.csv"
PHOTO_FILE = "test_photo.jpg"

# Load encodings
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

# ------------------- Process Image -------------------
img = cv2.imread(PHOTO_FILE)
if img is None:
    print(f"⚠️ Could not read image: {PHOTO_FILE}")
    exit()

scale_factor = 1  # keep 1 for full size, or use 0.5 for large images
if scale_factor != 1:
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    img = cv2.resize(img, (width, height))

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect all faces in the image
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

        # Draw rectangle
        top, right, bottom, left = face_location
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Unknown face
        top, right, bottom, left = face_location
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(img, "Unknown", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Face Recognition Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()