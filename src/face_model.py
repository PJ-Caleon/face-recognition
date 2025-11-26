import face_recognition
import cv2
import os
import pickle

# folder: dataset/
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "../dataset")

known_encodings = []
known_names = []

for filename in os.listdir(dataset_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(dataset_path, filename)
        print(f"Processing {filename}...")
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            encoding = encodings[0]

            raw_name = os.path.splitext(filename)[0]   # pj1
            # convert pj1 → pj, steven3 → steven
            name = ''.join([c for c in raw_name if not c.isdigit()]).lower()

            known_encodings.append(encoding)
            known_names.append(name)
        else:
            print(f"No face found in {filename}, skipping...")

data = {"encodings": known_encodings, "names": known_names}

model_path = os.path.join(base_dir, "face_model.pickle")
with open(model_path, "wb") as f:
    pickle.dump(data, f)

print(f"✅ Model trained and saved as {model_path}")
