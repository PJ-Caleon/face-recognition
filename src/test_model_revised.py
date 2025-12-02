import cv2
import pickle
import numpy as np
import mediapipe as mp
# from tflite_runtime.interpreter import Interpreter
from tensorflow.lite.python.interpreter import Interpreter


interpreter = Interpreter(model_path="facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


with open("face_model_tflite.pickle", "rb") as f:
    data = pickle.load(f)

known_embeddings = data["embeddings"]
known_names = data["names"]

# Mediapipe
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def get_embed(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype("float32")
    face_img = (face_img - 127.5) / 128.0
    face_img = np.expand_dims(face_img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], face_img)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    return embedding / np.linalg.norm(embedding)  # L2 normalize

# ----------------------
# Compare embeddings
# ----------------------
def find_match(embedding, known_embeddings, known_names, threshold=0.4):
    distances = [np.linalg.norm(embedding - e) for e in known_embeddings]
    best_idx = np.argmin(distances)
    if distances[best_idx] < threshold:
        return known_names[best_idx], distances[best_idx]
    else:
        return "Unknown", None


video = cv2.VideoCapture(0)
print("Starting Camera... Press 'q' to Quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb_frame)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = x1 + int(bbox.width * w)
            y2 = y1 + int(bbox.height * h)

            face = rgb_frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            embedding = get_embed(face)
            name, distance = find_match(embedding, known_embeddings, known_names)

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name.capitalize(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if name != "Unknown":
                print(f"User:  {name.capitalize()}, Distance: {distance:.3f}")
            else:
                print("Unkown")


    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
