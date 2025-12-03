import cv2
import time
import os

name = "pj"

def capture_photos(duration_seconds=120, interval_seconds=5, output_dir="generated_dataset"):
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start video capture (0 = default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Camera started. Taking photos...")
    start_time = time.time()
    photo_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Show live camera feed
        cv2.imshow("Camera - Press 'q' to quit early", frame)

        # Time check
        if time.time() - start_time >= photo_count * interval_seconds:
            filename = os.path.join(output_dir, f"{name}{photo_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            photo_count += 1

        # Stop after duration
        if time.time() - start_time >= duration_seconds:
            print("Finished capturing photos.")
            break

        # Allow user to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting early.")
            break

    cap.release()
    cv2.destroyAllWindows()

capture_photos(duration_seconds=2000, interval_seconds=1)
