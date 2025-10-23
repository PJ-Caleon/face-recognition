import face_recognition
import cv2
import os
import pickle

# Automatically detect dataset folder relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "../dataset")  # adjust if your dataset is elsewhere

known_encodings = []
known_names = []

# Loop through each image in the dataset folder
for filename in os.listdir(dataset_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(dataset_path, filename)
        print(f"Processing {filename}...")
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            encoding = encodings[0]
            # Use filename (without extension) as label instead of hardcoding your name
            name = os.path.splitext(filename)[0]
            known_encodings.append(encoding)
            known_names.append(name)
        else:
            print(f"No face found in {filename}, skipping...")

known_names = ["pj" for _ in known_names]

# Save encodings to a file
data = {"encodings": known_encodings, "names": known_names}

model_path = os.path.join(base_dir, "face_model.pickle")
with open(model_path, "wb") as f:
    pickle.dump(data, f)

print(f"✅ Model trained and saved as {model_path}")
