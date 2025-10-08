import cv2
import os

save_folder = "Recent_Images"
os.makedirs(save_folder, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

print("Press 's' to capture and save face image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(faces) == 0:
            print("No face detected. Try again.")
            continue
        # Find the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        face_img = frame[y:y + h, x:x + w]
        roll_no = input("Enter Roll Number: ")
        filename = f"{save_folder}/{roll_no}.jpg"
        cv2.imwrite(filename, face_img)
        print(f"Face image saved as {filename}")

    elif key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()