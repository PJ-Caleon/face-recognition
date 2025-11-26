import cv2
import face_recognition
import pickle
import serial
import time

# ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
# time.sleep(2)  # wait for Arduino to reset

# Load the trained face model
with open("face_model.pickle", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Initialize webcam
video = cv2.VideoCapture(0)

print("🎥 Starting camera... Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame from BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encodings directly
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin()

        # Check if the best match is within tolerance
        if face_distances[best_match_index] < 0.3:  # adjust this threshold if needed
            name = known_names[best_match_index]
        else:
            name = "Unknown"

        print(f"Best match: {known_names[best_match_index]}, Distance: {face_distances[best_match_index]:.3f}")



        # Draw box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Add label
        label = name.capitalize()

        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if name != "Unknown":
            print(f"✅ Detected: Hello {name.capitalize()}")
        else:
            print("❌ Unknown face detected")



    # Show the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
