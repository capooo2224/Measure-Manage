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

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.detector = cv2.FaceDetectorYN.create(
            model=YUNET_PATH,
            config="",
            input_size=(self.frame_w, self.frame_h),
            score_threshold=0.8,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=SFACE_PATH, config=""
        )

        self.ENTRY_THRESHOLD = entry_threshold
        self.EXIT_THRESHOLD = exit_threshold
        self.SAVE_COOLDOWN = 3.0
        self.last_save_time = 0

        self.logged_students = set()
        self.active_states = {}
        self.profile_metadata = []
        self.feature_matrix = None

        self.load_expected_faces()

    def load_expected_faces(self):
        self.profile_metadata.clear()
        features_list = []

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths = [
            p for p in EXPECTED_DIR.iterdir() if p.suffix.lower() in valid_exts
        ]

        ref_detector = cv2.FaceDetectorYN.create(
            model=YUNET_PATH,
            config="",
            input_size=(300, 300),
            score_threshold=0.3,
        )

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            parts = img_path.stem.split("_")
            student_id = parts[0] if len(parts) > 0 else "N/A"
            person_name = parts[1] if len(parts) > 1 else img_path.stem
            variant = parts[2] if len(parts) > 2 else "default"

            ref_detector.setInputSize((img.shape[1], img.shape[0]))
            _, ref_faces = ref_detector.detect(img)

            if ref_faces is not None and len(ref_faces) > 0:
                aligned = self.recognizer.alignCrop(img, ref_faces[0])
                feat = self.recognizer.feature(aligned).flatten()

                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat = feat / norm

                features_list.append(feat)
                self.profile_metadata.append(
                    {"id": student_id, "name": person_name}
                )

                print(
                    f"[Loaded Reference] ID: {student_id} | Name: {person_name} | Variant: {variant}"
                )

        if features_list:
            self.feature_matrix = np.array(features_list)

    def process_next_frame(self):
        """Captures frame, runs detection/recognition, returns annotated BGR frame and newly logged student info."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        _, faces = self.detector.detect(frame)
        current_time = time.time()
        newly_logged = None

        if faces is not None:
            for face in faces:
                box = list(map(int, face[:4]))
                x, y, w, h = box[0], box[1], box[2], box[3]

                aligned_face = self.recognizer.alignCrop(frame, face)
                live_feat = self.recognizer.feature(aligned_face).flatten()

                norm = np.linalg.norm(live_feat)
                if norm > 0:
                    live_feat = live_feat / norm

                max_score = 0.0
                candidate_profile = None

                if (
                    self.feature_matrix is not None
                    and len(self.feature_matrix) > 0
                ):
                    scores = np.dot(self.feature_matrix, live_feat)
                    best_idx = np.argmax(scores)
                    max_score = float(scores[best_idx])
                    candidate_profile = self.profile_metadata[best_idx]

                is_matched = False

                if candidate_profile is not None:
                    student_id = candidate_profile["id"]
                    person_name = candidate_profile["name"]

                    was_active = self.active_states.get(student_id, False)
                    if was_active:
                        is_matched = max_score >= self.EXIT_THRESHOLD
                    else:
                        is_matched = max_score >= self.ENTRY_THRESHOLD

                    self.active_states[student_id] = is_matched

                    if is_matched:
                        color = (0, 255, 0)
                        display_label = f"ID: {student_id} | {person_name}"

                        if student_id not in self.logged_students:
                            print(
                                f"[LOG | PRESENT] ID: {student_id} | Name: {person_name} | Score: {max_score:.2f}"
                            )
                            self.logged_students.add(student_id)
                            newly_logged = {
                                "id": student_id,
                                "name": person_name,
                            }

                if not is_matched:
                    color = (0, 0, 255)
                    display_label = "Unknown"

                    if (
                        current_time - self.last_save_time
                    ) > self.SAVE_COOLDOWN:
                        self.last_save_time = current_time
                        save_path = (
                            TEMP_DIR / f"unknown_{int(current_time)}.png"
                        )
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

        return frame, newly_logged

    def cleanup(self):
        self.cap.release()
        for item in TEMP_DIR.iterdir():
            if item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    item.unlink()
                except Exception:
                    pass
        print("Cleaned up temporary 'faces_db' directory.")