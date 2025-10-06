import cv2
import face_recognition
import os
import pickle

# Folder with student images
IMAGES_PATH = "Images"
ENCODE_FILE = "EncodeFile.p"

# Read all images and get IDs
imageList = []
studentIds = []
for file in os.listdir(IMAGES_PATH):
    img = cv2.imread(os.path.join(IMAGES_PATH, file))
    if img is None:
        print(f"Could not read {file}, skipping...")
        continue
    imageList.append(img)
    studentIds.append(os.path.splitext(file)[0])

# Function to generate encodings
def findEncodings(imagesList):
    encodeList = []
    for idx, img in enumerate(imagesList):
        try:
            if img.dtype != 'uint8':
                img = img.astype('uint8')
            if len(img.shape) != 3 or img.shape[2] != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(rgb_img)
            if enc:
                encodeList.append(enc[0])
            else:
                print(f"No face found in image {idx}, skipping...")
        except Exception as e:
            print(f"Error encoding image {idx}: {e}")
    return encodeList

# Generate encodings
print("Encoding started...")
encodeListKnown = findEncodings(imageList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
# Save encodings to pickle
with open(ENCODE_FILE, "wb") as f:
    pickle.dump(encodeListKnownWithIds, f)
print("Encoding completed and saved to EncodeFile.p")