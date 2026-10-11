import cv2
import numpy as np
import os

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
nose_cascade = cv2.CascadeClassifier("D:/complete web dev/Web D/Face Detection/haarcascade_mcs_nose.xml")
mouth_cascade = cv2.CascadeClassifier("D:/complete web dev/Web D/Face Detection/haarcascade_mcs_mouth.xml")

# Check if cascades loaded correctly
if nose_cascade.empty():
    print("❌ Error: Nose cascade not loaded!")
if mouth_cascade.empty():
    print("❌ Error: Mouth cascade not loaded!")

# Load face recognizer and trained data
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_path = "face_trainer.yml"

if os.path.exists(model_path):
    recognizer.read(model_path)
    print("✅ Face recognizer model loaded!")
else:
    print("⚠️ Face recognizer model not found! Train the model first.")

# Dictionary of known persons (IDs -> Names)
known_faces = {1: "Pranav", 2: "Om", 3: "Akshay"}  # Update with actual names

def detect_and_recognize_faces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Face (Blue)

        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Recognize face
        id_, confidence = recognizer.predict(roi_gray)
        confidence_text = f"{round(100 - confidence, 2)}%"
        
        if confidence < 60:  # Confidence threshold (lower = more strict)
            name = known_faces.get(id_, "Unknown")
        else:
            name = "Unknown"

        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        """  # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Eyes (Green)

        # Detect nose
        nose = nose_cascade.detectMultiScale(roi_gray, 1.1, 5)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)  # Nose (Red)

        # Detect mouth
        mouths = mouth_cascade.detectMultiScale(roi_gray, 1.3, 15)
        for (mx, my, mw, mh) in mouths:
            if my > h // 2:  # Ensure it's below the nose
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)  # Mouth (Yellow) """

    return img

# Initialize webcam
video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    _, img = video_capture.read()
    
    if img is None:
        print("Error: Could not read from webcam.")
        break

    # Apply detection & recognition
    img = detect_and_recognize_faces(img)

    # Show the output
    cv2.imshow("Face Recognition & Feature Detection", img)

    # Exit on 'g' key press
    if cv2.waitKey(1) & 0xFF == ord('g'):
        break

video_capture.release()
cv2.destroyAllWindows()
