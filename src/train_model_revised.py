import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from tensorflow.lite.python.interpreter import Interpreter

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "../dataset")
model_path = os.path.join(base_dir, "face_model_tflite.pickle")

# Load TFLite FaceNet model
interpreter = Interpreter("facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mediapipe face detector
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Data holders
known_embeddings = []
known_names = []

def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype("float32")
    face_img = (face_img - 127.5) / 128.0  # normalize
    face_img = np.expand_dims(face_img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], face_img)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    return embedding / np.linalg.norm(embedding)  # L2 normalize


for filename in os.listdir(dataset_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(dataset_path, filename)
        print(f"Processing {filename}...")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {filename}, skipping...")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            h, w, _ = img.shape
            x1 = max(int(bbox.xmin * w), 0)
            y1 = max(int(bbox.ymin * h), 0)
            x2 = min(x1 + int(bbox.width * w), w)
            y2 = min(y1 + int(bbox.height * h), h)
            face = rgb[y1:y2, x1:x2]

            if face.size == 0:
                print(f"No valid face in {filename}, skipping...")
                continue

            embedding = get_embedding(face)

            # Extract name from filename: remove digits
            raw_name = os.path.splitext(filename)[0]
            name = ''.join([c for c in raw_name if not c.isdigit()]).lower()

            known_embeddings.append(embedding)
            known_names.append(name)
        else:
            print(f"No face detected in {filename}, skipping...")


data = {"embeddings": known_embeddings, "names": known_names}

with open(model_path, "wb") as f:
    pickle.dump(data, f)

print(f"Model trained and saved as {model_path}")
