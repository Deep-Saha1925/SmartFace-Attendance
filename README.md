# Face Recognition Attendance System

A real-time **Face Recognition Attendance System** that uses AI & Machine Learning to automatically recognize faces and mark attendance. This project integrates with **MySQL** for database management, **Firebase** for cloud storage, and **Cloudinary** for image hosting.

---

## Features

- **Real-Time Face Recognition**: Detect and recognize faces using your webcam
- **Static Image Recognition**: Also supports attendance marking from static images
- **Automatic Attendance Tracking**: Automatically updates attendance records in MySQL database
- **Dual Storage System**: 
  - **MySQL Database**: Primary storage for student data and attendance records
  - **CSV Backup**: Local CSV file as backup for student data
- **Cloud Integration**:
  - **Firebase**: Stores credentials and cloud configuration
  - **Cloudinary**: Stores student face images in the cloud
- **Smart Attendance Logic**: Prevents duplicate attendance within a configurable time threshold (default: 1 minute)
- **Interactive UI**: Displays real-time face detection with bounding boxes and student names

---

## Project Structure

```
REAL-TIME-FACE-ATTENDANCE-SYSTEM/
├── Add_Student.py           # Script to add new students with face capture
├── EncodeGenerator.py       # Generates face encodings from Images folder
├── Main.py                  # Real-time webcam-based attendance system
├── Main2.py                 # Static image-based attendance system
├── config.py                # Configuration file loading environment variables
├── db.py                    # MySQL database connection module
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (credentials)
├── students.csv             # Local CSV backup of student data
├── EncodeFile.p             # Pickle file storing face encodings
├── haarcascade_frontalface_default.xml  # OpenCV face detection cascade
├── Images/                  # Folder containing student face images
├── Recent_Images/           # Folder for newly captured student images
├── Resources/               # UI resources (backgrounds, mode images)
└── TEST/                    # Testing scripts and sample data
```

---

## Prerequisites

### 1. Python Installation
- **Python 3.10 or higher** is required

### 2. MySQL Database
- Install MySQL Server on your machine
- Create a database named `face_recognition`
- Create a table named `students` with the following schema:

```sql
CREATE TABLE students (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    major VARCHAR(100),
    starting_year VARCHAR(10),
    total_attendance INT DEFAULT 0,
    standing VARCHAR(50),
    year VARCHAR(20),
    last_attendance_time DATETIME
);
```

### 3. Cloudinary Account
- Create a free account at [cloudinary.com](https://cloudinary.com)
- Get your Cloud Name, API Key, and API Secret

### 4. Firebase Project (Optional)
- Create a project at [firebase.google.com](https://firebase.google.com)
- Download the service account JSON key file

---

## Installation

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd REAL-TIME-FACE-ATTENDANCE-SYSTEM
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv myenv
myenv\Scripts\activate    # On Windows
# source myenv/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root with the following variables:

```env
# Cloudinary Configuration
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Firebase Configuration
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/firebase-adminsdk.json

# Student Data Path
STUDENT_DATA_PATH=students.csv

# MySQL Database Configuration
MYSQL_USERNAME=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_HOST=localhost
MYSQL_DATABASE=face_recognition

# Attendance Settings (in minutes)
ATTENDANCE_ELAPSED_MINUTES=1
```

---

## How to Use

### Step 1: Prepare Student Images
Place all student face images in the `Images/` folder. 
- Image filename should be the student ID (e.g., `12345.jpg`)
- Supported formats: JPG, PNG
- Use clear, well-lit photos for best recognition

### Step 2: Generate Face Encodings
Run the encoding generator to create the face recognition model:
```bash
python EncodeGenerator.py
```
This will create `EncodeFile.p` containing all face encodings.

### Step 3: Add New Students
To add a new student with face capture:
```bash
python Add_Student.py
```
Follow the prompts to enter:
- Student ID
- Name
- Major
- Starting Year
- Total Attendance (default: 0)
- Standing
- Year

The script will open your webcam - press **'s'** to capture the face, or **'q'** to quit.

### Step 4: Run the Attendance System

#### Option A: Real-Time Webcam Mode
```bash
python Main.py
```
- Opens your default webcam
- Detects and recognizes faces in real-time
- Press **'q'** to quit
- Attendance is automatically marked when a recognized face is detected

#### Option B: Static Image Mode
```bash
python Main2.py
```
- Processes a static image for attendance
- Edit `PHOTO_FILE` in `Main2.py` to point to your image
- Useful for batch processing or testing

---

## How It Works

1. **Face Detection**: Uses OpenCV's Haar Cascade classifier to detect faces
2. **Face Encoding**: Uses the `face_recognition` library to generate 128-dimensional face encodings
3. **Face Matching**: Compares detected faces against known encodings using Euclidean distance
4. **Attendance Update**: 
   - Checks the last attendance time
   - Only marks attendance if elapsed time exceeds the threshold (default: 1 minute)
   - Updates both MySQL database and CSV backup

---

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `ATTENDANCE_ELAPSED_MINUTES` | Minimum minutes between attendance marks | 1 |
| `MYSQL_HOST` | MySQL server hostname | localhost |
| `MYSQL_DATABASE` | Database name | face_recognition |

---

## Troubleshooting

### No face detected
- Ensure proper lighting on the face
- Face should be clearly visible and facing the camera
- Remove glasses or hats that might obscure facial features

### Encoding errors
- Make sure all images in `Images/` folder are valid image files
- Delete corrupted images and re-run `EncodeGenerator.py`

### Database connection errors
- Verify MySQL is running
- Check credentials in `.env` file
- Ensure the database and table exist

### Module not found errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

---

## Dependencies

- `opencv-python` - Computer vision and image processing
- `face-recognition` - Face detection and recognition
- `firebase-admin` - Firebase integration
- `cloudinary` - Cloud image storage
- `python-dotenv` - Environment variable management
- `cvzone` - Computer vision utilities
- `numpy` - Numerical computing
- `mysql-connector-python` - MySQL database connector
- `requests` - HTTP requests

---

## License

This project is for educational purposes. Please ensure compliance with privacy laws and regulations when using face recognition technology.

---

## Author

**Deep Saha** - Created for educational and demonstration purposes.