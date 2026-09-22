from pathlib import Path
import time
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
YUNET_PATH = str(BASE_DIR / "face_detection_yunet_2023mar.onnx")
SFACE_PATH = str(BASE_DIR / "face_recognition_sface_2021dec.onnx")
EXPECTED_DIR = BASE_DIR / "Expected_faces_db"
TEMP_DIR = BASE_DIR / "faces_db"

EXPECTED_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)


class FaceTrackerEngine:

    def __init__(self, entry_threshold=0.363, exit_threshold=0.310):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("[ERROR] Could not open webcam.")

        # Read dimensions directly from webcam
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize YuNet detector and SFace recognizer
        self.detector = cv2.FaceDetectorYN.create(
            YUNET_PATH, "", (self.w, self.h), score_threshold=0.8
        )
        self.recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

        self.ENTRY_THRESHOLD = entry_threshold
        self.EXIT_THRESHOLD = exit_threshold
        self.SAVE_COOLDOWN = 3.0
        self.last_save_time = 0

        self.logged_students = set()
        self.active_states = {}
        self.profile_metadata = []
        self.feature_matrix = None

        self.load_expected_faces()

    def _extract_feature(self, img, face):
        """Helper to align a detected face and calculate its normalized 128D feature vector."""
        aligned = self.recognizer.alignCrop(img, face)
        feat = self.recognizer.feature(aligned).flatten()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat, aligned

    def load_expected_faces(self):
        """Scans the reference folder and computes feature embeddings for all known faces."""
        self.profile_metadata.clear()
        features_list = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

        ref_detector = cv2.FaceDetectorYN.create(
            YUNET_PATH, "", (300, 300), score_threshold=0.3
        )

        for img_path in EXPECTED_DIR.iterdir():
            if img_path.suffix.lower() not in valid_exts:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Parse filename pattern: ID_Name_Variant.ext
            parts = img_path.stem.split("_")
            student_id = parts[0] if parts else "N/A"
            person_name = parts[1] if len(parts) > 1 else img_path.stem

            ref_detector.setInputSize((img.shape[1], img.shape[0]))
            _, ref_faces = ref_detector.detect(img)

            if ref_faces is not None and len(ref_faces) > 0:
                feat, _ = self._extract_feature(img, ref_faces[0])
                features_list.append(feat)
                self.profile_metadata.append(
                    {"id": student_id, "name": person_name}
                )

        if features_list:
            self.feature_matrix = np.array(features_list)

    def get_all_expected_students(self):
        """Returns a unique list of all expected students loaded from disk."""
        unique = {p["id"]: p["name"] for p in self.profile_metadata}
        return [{"id": sid, "name": name} for sid, name in unique.items()]

    def process_next_frame(self):
        """Captures frame, matches faces against vector database, draws overlay, returns frame & logs."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        _, faces = self.detector.detect(frame)
        current_time = time.time()
        newly_logged = None

        if faces is not None:
            for face in faces:
                x, y, w, h = map(int, face[:4])
                live_feat, aligned_face = self._extract_feature(frame, face)

                max_score, candidate = 0.0, None

                # Compute cosine similarity dot product across all known feature vectors
                if self.feature_matrix is not None and len(self.feature_matrix):
                    scores = np.dot(self.feature_matrix, live_feat)
                    best_idx = np.argmax(scores)
                    max_score = float(scores[best_idx])
                    candidate = self.profile_metadata[best_idx]

                is_matched = False

                if candidate:
                    sid, name = candidate["id"], candidate["name"]
                    was_active = self.active_states.get(sid, False)

                    # Hysteresis threshold checking to prevent UI flickering
                    threshold = (
                        self.EXIT_THRESHOLD
                        if was_active
                        else self.ENTRY_THRESHOLD
                    )
                    is_matched = max_score >= threshold
                    self.active_states[sid] = is_matched

                    if is_matched:
                        color = (0, 255, 0)
                        label = f"ID: {sid} | {name}"

                        if sid not in self.logged_students:
                            self.logged_students.add(sid)
                            newly_logged = {"id": sid, "name": name}

                if not is_matched:
                    color = (0, 0, 255)
                    label = "Unknown"

                    # Save unknown faces with a 3-second cooldown rate limit
                    if current_time - self.last_save_time > self.SAVE_COOLDOWN:
                        self.last_save_time = current_time
                        cv2.imwrite(
                            str(TEMP_DIR / f"unknown_{int(current_time)}.png"),
                            aligned_face,
                        )

                # Draw bounding box and information text
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({max_score:.2f})",
                    (x, max(15, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        return frame, newly_logged

    def cleanup(self):
        """Releases hardware resources and deletes cached unknown face crops."""
        self.cap.release()
        for item in TEMP_DIR.glob("*"):
            try:
                item.unlink()
            except Exception:
                pass