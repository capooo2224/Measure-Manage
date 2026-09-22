import os
import time
import cv2

# Configuration
MODEL_PATH = "face_detection_yunet_2023mar.onnx"
OUTPUT_DIR = "faces_db"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

# Retrieve webcam dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize YuNet detector compatible with OpenCV 5.0+
detector = cv2.FaceDetectorYN.create(
    model=MODEL_PATH,
    config="",
    input_size=(frame_width, frame_height),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000,
)

print("Displaying video feed...")
print("Press 's' to save detected faces to 'faces_db/' folder")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame")
        break

    # Perform face detection
    _, faces = detector.detect(frame)

    current_faces = []

    if faces is not None:
        for face in faces:
            # Extract bounding box coordinates (x, y, w, h)
            box = list(map(int, face[:4]))
            x, y, w, h = box[0], box[1], box[2], box[3]

            # Ensure coordinates stay within frame bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame_width, x + w), min(frame_height, y + h)

            # Store cropped face region
            if w > 0 and h > 0:
                face_crop = frame[y1:y2, x1:x2]
                current_faces.append(face_crop)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video Feed - YuNet Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 's' to save cropped faces to disk
    if key == ord("s"):
        if current_faces:
            for idx, face_img in enumerate(current_faces):
                timestamp = int(time.time())
                # Resize to standard dimensions for database consistency
                resized_face = cv2.resize(face_img, (112, 112))
                filename = os.path.join(
                    OUTPUT_DIR, f"face_{timestamp}_{idx}.png"
                )
                cv2.imwrite(filename, resized_face)
                print(f"Saved: {filename}")
        else:
            print("No faces detected to save.")

    # Press 'q' to exit
    elif key == ord("esc"):
        print("Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Done!")