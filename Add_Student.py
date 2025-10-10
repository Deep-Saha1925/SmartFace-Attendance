import cv2
import os
import csv
from datetime import datetime
from db import get_connection 

save_folder = "Recent_Images"
os.makedirs(save_folder, exist_ok=True)
csv_file = "students copy.csv"

student_id = input("Enter Student ID: ").strip()
name = input("Enter Name: ").strip()
major = input("Enter Major: ").strip()
starting_year = input("Enter Starting Year: ").strip()
total_attendance = input("Enter Total Attendance: ").strip()
standing = input("Enter Standing: ").strip()
year = input("Enter Year: ").strip()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

print("Press 's' to capture and save face image. Press 'q' to quit.")
image_saved = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        margin = int(0.2 * max(w, h))
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, frame.shape[1])
        y2 = min(y + h + margin, frame.shape[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(faces) == 0:
            print("No face detected. Try again.")
            continue

        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        margin = int(0.2 * max(w, h))
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, frame.shape[1])
        y2 = min(y + h + margin, frame.shape[0])
        face_img = frame[y1:y2, x1:x2]
        filename = f"{save_folder}/{student_id}.jpg"
        cv2.imwrite(filename, face_img)
        print(f"Face image saved as {filename}")
        image_saved = True
        break

    elif key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()

if image_saved:
    last_attendance_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            student_id,
            name,
            major,
            starting_year,
            total_attendance,
            standing,
            year,
            last_attendance_time
        ])
    print("Student data added to CSV.")

    try:
        connection = get_connection()
        cursor = connection.cursor()
        sql = """
            INSERT INTO students 
            (id, name, major, starting_year, total_attendance, standing, year, last_attendance_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            student_id,
            name,
            major,
            starting_year,
            total_attendance,
            standing,
            year,
            last_attendance_time
        )
        cursor.execute(sql, values)
        connection.commit()
        print("Student data added to MySQL database.")
    except Exception as e:
        print("Database Error:", e)
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
else:
    print("No image saved. CSV and DB not updated.")