import cv2
import os
import numpy as np

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

dataset_path = "dataset"
faces, labels = [], []

# Load images and labels
print("🔄 Training face recognizer...")
for user_id in os.listdir(dataset_path):
    user_folder = os.path.join(dataset_path, user_id)
    
    if not os.path.isdir(user_folder):
        continue

    for img_name in os.listdir(user_folder):
        img_path = os.path.join(user_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue
        
        faces.append(img)
        labels.append(int(user_id))

# Train the recognizer
recognizer.train(faces, np.array(labels))

# Save the trained model
recognizer.save("face_trainer.yml")
print("✅ Training complete! Model saved as 'face_trainer.yml'.")
