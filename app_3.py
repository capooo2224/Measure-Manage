from pathlib import Path
import time
import cv2
import numpy as np

# Base Directories & Files
BASE_DIR = Path(__file__).resolve().parent
YUNET_PATH = str(BASE_DIR / "face_detection_yunet_2023mar.onnx")
SFACE_PATH = str(BASE_DIR / "face_recognition_sface_2021dec.onnx")

EXPECTED_DIR = BASE_DIR / "Expected_faces_db"
TEMP_DIR = BASE_DIR / "faces_db"

EXPECTED_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Camera Initialization
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit(1)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Models
detector = cv2.FaceDetectorYN.create(
    model=YUNET_PATH, config="", input_size=(frame_w, frame_h), score_threshold=0.8
)
recognizer = cv2.FaceRecognizerSF.create(model=SFACE_PATH, config="")

# Parameters
MATCH_THRESHOLD = 0.363
SAVE_COOLDOWN = 3.0
LOG_COOLDOWN = 5.0

last_save_time = 0
last_log_time = {}

# Matrices for Fast Vectorized Comparison
profile_metadata = []  # Index maps to {"id": ..., "name": ...}
feature_matrix = None  # Shape: (N, 128) L2-normalized vectors


def load_expected_faces():
    """Scans Expected_faces_db for 'STUDENTID_NAME_VARIANT.jpg' and builds a feature matrix."""
    global feature_matrix
    profile_metadata.clear()
    features_list = []

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = [
        p for p in EXPECTED_DIR.iterdir() if p.suffix.lower() in valid_exts
    ]

    ref_detector = cv2.FaceDetectorYN.create(
        model=YUNET_PATH, config="", input_size=(300, 300), score_threshold=0.3
    )

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Parse Filename Format: STUDENTID_NAME_VARIANT.jpg
        parts = img_path.stem.split("_")
        student_id = parts[0] if len(parts) > 0 else "N/A"
        person_name = parts[1] if len(parts) > 1 else img_path.stem
        variant = parts[2] if len(parts) > 2 else "default"

        # Reuse single detector instance
        ref_detector.setInputSize((img.shape[1], img.shape[0]))
        _, ref_faces = ref_detector.detect(img)

        if ref_faces is not None and len(ref_faces) > 0:
            aligned = recognizer.alignCrop(img, ref_faces[0])
            feat = recognizer.feature(aligned).flatten()

            # L2-Normalize vector for cosine similarity via dot product
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm

            features_list.append(feat)
            profile_metadata.append({"id": student_id, "name": person_name})

            print(
                f"[Loaded Reference] ID: {student_id} | Name: {person_name} | Variant: {variant}"
            )
        else:
            print(
                f"[WARNING] No face detected in reference photo: '{img_path.name}'"
            )

    if features_list:
        feature_matrix = np.array(features_list)  # (N, 128)
    else:
        feature_matrix = None


def cleanup_temp_faces():
    """Cleans temporary captures from faces_db."""
    for item in TEMP_DIR.iterdir():
        if item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                item.unlink()
            except Exception:
                pass
    print("Cleaned up temporary 'faces_db' directory.")


load_expected_faces()
print(f"\nTotal reference features loaded: {len(profile_metadata)}")
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
            live_feat = recognizer.feature(aligned_face).flatten()

            # Normalize live feature vector
            norm = np.linalg.norm(live_feat)
            if norm > 0:
                live_feat = live_feat / norm

            max_score = 0.0
            matched_profile = None

            # Vectorized Cosine Similarity (Dot product against normalized matrix)
            if feature_matrix is not None and len(feature_matrix) > 0:
                scores = np.dot(feature_matrix, live_feat)
                best_idx = np.argmax(scores)
                max_score = float(scores[best_idx])

                if max_score >= MATCH_THRESHOLD:
                    matched_profile = profile_metadata[best_idx]

            if matched_profile is not None:
                color = (0, 255, 0)
                student_id = matched_profile["id"]
                person_name = matched_profile["name"]
                display_label = f"ID: {student_id} | {person_name}"

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

                if (current_time - last_save_time) > SAVE_COOLDOWN:
                    last_save_time = current_time
                    save_path = TEMP_DIR / f"unknown_{int(current_time)}.png"
                    cv2.imwrite(str(save_path), aligned_face)

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