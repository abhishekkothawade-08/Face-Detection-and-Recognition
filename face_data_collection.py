import cv2
import os

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create dataset folder if it doesn't exist
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Ask for user ID
user_id = input("Enter a numeric ID for this person: ")
user_folder = os.path.join(dataset_path, user_id)

if not os.path.exists(user_folder):
    os.makedirs(user_folder)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

print("📷 Capturing face images... Look at the camera.")

count = 0
while True:
    _, img = video_capture.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray_img[y:y+h, x:x+w]
        file_path = os.path.join(user_folder, f"{count}.jpg")
        cv2.imwrite(file_path, face_img)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, f"Capturing {count}/100", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Face Data Collection", img)

    if count >= 100:  # Capture 100 images per person
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

print(f"✅ Face data collection complete! Images saved in: {user_folder}")
