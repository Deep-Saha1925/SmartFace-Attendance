import cv2
import os
import csv
from datetime import datetime

save_folder = "Recent_Images"
os.makedirs(save_folder, exist_ok=True)
csv_file = "students.csv"

# Prompt for student details
student_id = input("Enter Student ID: ")
name = input("Enter Name: ")
major = input("Enter Major: ")
starting_year = input("Enter Starting Year: ")
total_attendance = input("Enter Total Attendance: ")
standing = input("Enter Standing: ")
year = input("Enter Year: ")

# Load Haar Cascade for face detection
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

    # Draw rectangles around faces (with margin)
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

# Update CSV if image was saved
if image_saved:
    last_attendance_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, 'a', newline='') as f:
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
    print("Student data updated in CSV.")
else:
    print("No image saved. CSV not updated.")