import glob
import os
import time
import cv2

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(BASE_DIR, "face_recognition_sface_2021dec.onnx")

EXPECTED_DIR = os.path.join(BASE_DIR, "Expected_faces_db")
TEMP_DIR = os.path.join(BASE_DIR, "faces_db")

os.makedirs(EXPECTED_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize YuNet & SFace
detector = cv2.FaceDetectorYN.create(
    model=YUNET_PATH, config="", input_size=(frame_w, frame_h), score_threshold=0.8
)
recognizer = cv2.FaceRecognizerSF.create(model=SFACE_PATH, config="")

MATCH_THRESHOLD = 0.363
SAVE_COOLDOWN = 3.0
LOG_COOLDOWN = 5.0

last_save_time = 0
last_log_time = {}

# Structure: { "Student_ID": {"id": "...", "name": "...", "features": [<vector_1>, <vector_2>]} }
expected_embeddings = {}


def load_expected_faces():
    """
    Scans Expected_faces_db/ for 'STUDENTID_NAME_VARIANT.jpg' (flat structure).
    Supports multiple reference images per Student ID.
    """
    expected_embeddings.clear()

    if not os.path.exists(EXPECTED_DIR):
        print(f"[ERROR] Directory '{EXPECTED_DIR}' does not exist!")
        return

    files = [
        f
        for f in os.listdir(EXPECTED_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for fname in files:
        fpath = os.path.join(EXPECTED_DIR, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        # Parse filename using underscores (e.g., 250430348_john_glasses.jpg)
        base_name = os.path.splitext(fname)[0]
        parts = base_name.split("_")

        student_id = parts[0] if len(parts) > 0 else "N/A"
        person_name = parts[1] if len(parts) > 1 else base_name
        variant = parts[2] if len(parts) > 2 else "default"

        # Detect face in reference photo
        ref_detector = cv2.FaceDetectorYN.create(
            model=YUNET_PATH,
            config="",
            input_size=(img.shape[1], img.shape[0]),
            score_threshold=0.3,
        )
        _, ref_faces = ref_detector.detect(img)

        if ref_faces is not None and len(ref_faces) > 0:
            aligned = recognizer.alignCrop(img, ref_faces[0])
            feat = recognizer.feature(aligned)

            # Initialize entry for new student
            if student_id not in expected_embeddings:
                expected_embeddings[student_id] = {
                    "id": student_id,
                    "name": person_name,
                    "features": [],
                }

            expected_embeddings[student_id]["features"].append(feat)
            print(
                f"[Loaded Reference] ID: {student_id} | Name: {person_name} | Variant: {variant}"
            )
        else:
            print(
                f"[WARNING] No face detected in reference image: 'Expected_faces_db/{fname}'"
            )


def cleanup_temp_faces():
    """Deletes temporary captured images in faces_db on shutdown."""
    files = glob.glob(os.path.join(TEMP_DIR, "*"))
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                os.remove(f)
            except Exception:
                pass
    print("Cleaned up temporary 'faces_db' directory.")


load_expected_faces()
print(f"\nTotal student profiles loaded: {len(expected_embeddings)}")
print("Monitoring video feed... Press 'ESC' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _, faces = detector.detect(frame)
    current_time = time.time()

    if faces is not None:
        for face in faces:
            box = list(map(int, face[:4]))
            x, y, w, h = box[0], box[1], box[2], box[3]

            aligned_face = recognizer.alignCrop(frame, face)
            face_feature = recognizer.feature(aligned_face)

            max_score = 0.0
            matched_profile = None

            # Compare against all features stored for each student profile
            for student_id, profile in expected_embeddings.items():
                for ref_feature in profile["features"]:
                    score = recognizer.match(
                        face_feature,
                        ref_feature,
                        cv2.FaceRecognizerSF_FR_COSINE,
                    )
                    if score > max_score:
                        max_score = score
                        if score >= MATCH_THRESHOLD:
                            matched_profile = profile

            if matched_profile is not None:
                color = (0, 255, 0)
                student_id = matched_profile["id"]
                person_name = matched_profile["name"]
                display_label = f"ID: {student_id} | {person_name}"

                # Terminal Attendance Logging
                if (
                    student_id not in last_log_time
                    or (current_time - last_log_time[student_id]) > LOG_COOLDOWN
                ):
                    print(
                        f"[LOG | PRESENT] ID: {student_id} | Name: {person_name} | Score: {max_score:.2f} | Status: YES"
                    )
                    last_log_time[student_id] = current_time
            else:
                color = (0, 0, 255)
                display_label = "Unknown"

                # Temporary save for unknown detections
                if (current_time - last_save_time) > SAVE_COOLDOWN:
                    last_save_time = current_time
                    timestamp = int(current_time)
                    save_path = os.path.join(
                        TEMP_DIR, f"unknown_{timestamp}.png"
                    )
                    cv2.imwrite(save_path, aligned_face)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{display_label} ({max_score:.2f})",
                (x, max(15, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    cv2.imshow("Attendance Feed", frame)

    if (cv2.waitKey(1) & 0xFF) == 27:
        print("\nExiting and performing cleanup...")
        break

cap.release()
cv2.destroyAllWindows()
cleanup_temp_faces()
print("Done!")