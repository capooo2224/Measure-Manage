import csv
from datetime import datetime
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

        # Anti-False Positive Timing Controls
        self.LOG_VERIFICATION_DELAY = 2.0  # Seconds face must match before logging
        self.match_timestamps = {}        # Tracks {student_id: first_matched_timestamp}

        self.logged_students = set()
        self.active_states = {}
        self.profile_metadata = []
        self.feature_matrix = None

        # Group & CSV State
        self.selected_group_folder = None
        self.active_csv_path = None  # Created ONCE per app execution

        # Dual Timer State
        self.initial_timer_duration = 0  # Timer 1 in seconds
        self.grace_timer_duration = 0    # Timer 2 (Late) in seconds
        self.timer_start_time = None
        self.current_phase = "idle"       # "on_time", "late", "expired", "idle"
        self.is_running = False

    @staticmethod
    def get_available_groups():
        """Scans Expected_faces_db for group subdirectories."""
        return [
            d.name
            for d in EXPECTED_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def set_timers(self, init_hrs: float = 0, init_mins: float = 0, grace_hrs: float = 0, grace_mins: float = 0):
        """Sets durations for both the initial timer (Present) and second timer (Late)."""
        self.initial_timer_duration = max(0.0, (init_hrs * 3600.0) + (init_mins * 60.0))
        self.grace_timer_duration = max(0.0, (grace_hrs * 3600.0) + (grace_mins * 60.0))

    def select_group(self, group_name: str):
        """Loads group embeddings and metadata."""
        group_dir = EXPECTED_DIR / group_name
        if not group_dir.exists() or not group_dir.is_dir():
            raise ValueError(f"[ERROR] Group directory '{group_name}' does not exist.")

        self.selected_group_folder = group_dir
        self.load_expected_faces(group_dir)

    def start_session(self, group_name: str, init_hrs: float = 0, init_mins: float = 0, grace_hrs: float = 0, grace_mins: float = 0):
        """Initializes attendance session, reuses single CSV file per app launch, and starts timers."""
        self.select_group(group_name)
        self.set_timers(init_hrs=init_hrs, init_mins=init_mins, grace_hrs=grace_hrs, grace_mins=grace_mins)

        if self.active_csv_path is None:
            self.init_new_session_csv()

        self.timer_start_time = time.time()
        self.current_phase = "on_time" if self.initial_timer_duration > 0 else ("late" if self.grace_timer_duration > 0 else "expired")
        self.is_running = True if self.current_phase != "expired" else False

    def init_new_session_csv(self):
        """Creates a timestamped CSV log file inside the active group folder once per session."""
        if not self.selected_group_folder:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_{timestamp}.csv"
        self.active_csv_path = self.selected_group_folder / filename

        with open(self.active_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Student ID", "Name", "Status", "Timestamp"])

    def sort_csv_alphabetically(self, sort_by_col: int = 1):
        """Reads the active CSV file and re-writes it sorted alphabetically.

        sort_by_col: 0 for Student ID, 1 for Name (default).
        """
        if not self.active_csv_path or not self.active_csv_path.exists():
            return

        with open(self.active_csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)

        if not header or not rows:
            return

        # Sort rows case-insensitively by the specified column (Name = index 1)
        rows.sort(key=lambda x: x[sort_by_col].lower() if len(x) > sort_by_col else "")

        # Overwrite file with sorted contents
        with open(self.active_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def clear_csv_contents(self):
        """Resets the CSV headers if user discards attendance."""
        if self.active_csv_path and self.active_csv_path.exists():
            with open(self.active_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Student ID", "Name", "Status", "Timestamp"])

    def discard_session(self):
        """Stops active session, clears recorded data in current CSV file."""
        self.is_running = False
        self.current_phase = "idle"
        self.clear_csv_contents()
        self.logged_students.clear()
        self.active_states.clear()
        self.match_timestamps.clear()

    def _log_attendance(self, student_id: str, student_name: str, status: str = "Present"):
        """Helper method to log student attendance to the active CSV file."""
        if not self.active_csv_path or not self.active_csv_path.exists():
            return

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.active_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([student_id, student_name, status, time_str])

    def update_timer_phase(self):
        """Evaluates elapsed time and updates current phase (on_time -> late -> expired)."""
        if not self.is_running or self.timer_start_time is None:
            return

        elapsed = time.time() - self.timer_start_time

        if self.current_phase == "on_time":
            if elapsed >= self.initial_timer_duration:
                if self.grace_timer_duration > 0:
                    self.current_phase = "late"
                else:
                    self.current_phase = "expired"
                    self.is_running = False

        elif self.current_phase == "late":
            total_allowed = self.initial_timer_duration + self.grace_timer_duration
            if elapsed >= total_allowed:
                self.current_phase = "expired"
                self.is_running = False

    def get_remaining_time(self):
        """Returns remaining seconds for current active phase and the phase name."""
        if not self.is_running or self.timer_start_time is None or self.current_phase in ("idle", "expired"):
            return 0.0, "expired"

        elapsed = time.time() - self.timer_start_time

        if self.current_phase == "on_time":
            rem = max(0.0, self.initial_timer_duration - elapsed)
            return rem, "on_time"
        elif self.current_phase == "late":
            total_allowed = self.initial_timer_duration + self.grace_timer_duration
            rem = max(0.0, total_allowed - elapsed)
            return rem, "late"

        return 0.0, "expired"

    def _extract_feature(self, img, face):
        """Helper to align face and extract normalized feature vector."""
        aligned = self.recognizer.alignCrop(img, face)
        feat = self.recognizer.feature(aligned).flatten()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat, aligned

    def load_expected_faces(self, group_folder: Path = None):
        """Computes feature embeddings for expected faces."""
        target_dir = group_folder or self.selected_group_folder or EXPECTED_DIR
        self.profile_metadata.clear()
        features_list = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

        ref_detector = cv2.FaceDetectorYN.create(
            YUNET_PATH, "", (300, 300), score_threshold=0.50
        )

        for img_path in target_dir.iterdir():
            if img_path.suffix.lower() not in valid_exts:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

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
        else:
            self.feature_matrix = None

    def get_all_expected_students(self):
        """Returns unique list of expected students."""
        unique = {p["id"]: p["name"] for p in self.profile_metadata}
        return [{"id": sid, "name": name} for sid, name in unique.items()]

    def auto_log_dismissal(self):
        """Marks unlogged students as 'Absent' and sorts the CSV file."""
        all_students = self.get_all_expected_students()
        dismissed_count = 0

        for student in all_students:
            sid = student["id"]
            name = student["name"]
            if sid not in self.logged_students:
                self._log_attendance(sid, name, status="Absent")
                dismissed_count += 1

        # Sort CSV alphabetically by Name before concluding
        self.sort_csv_alphabetically(sort_by_col=1)

        self.is_running = False
        self.current_phase = "expired"
        return dismissed_count

    def process_next_frame(self):
        """Captures frame, detects faces, requires 1-second continuous match before logging."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        self.update_timer_phase()

        _, faces = self.detector.detect(frame)
        current_time = time.time()
        newly_logged = None
        currently_detected_ids = set()

        if faces is not None:
            for face in faces:
                x, y, w, h = map(int, face[:4])
                live_feat, aligned_face = self._extract_feature(frame, face)

                max_score, candidate = 0.0, None

                if self.feature_matrix is not None and len(self.feature_matrix):
                    scores = np.dot(self.feature_matrix, live_feat)
                    best_idx = np.argmax(scores)
                    max_score = float(scores[best_idx])
                    candidate = self.profile_metadata[best_idx]

                is_matched = False

                if candidate:
                    sid, name = candidate["id"], candidate["name"]
                    was_active = self.active_states.get(sid, False)

                    threshold = (
                        self.EXIT_THRESHOLD
                        if was_active
                        else self.ENTRY_THRESHOLD
                    )
                    is_matched = max_score >= threshold
                    self.active_states[sid] = is_matched

                    if is_matched:
                        currently_detected_ids.add(sid)
                        color = (0, 255, 0) if self.current_phase == "on_time" else (0, 255, 255)
                        label = f"ID: {sid} | {name}"

                        if self.is_running and sid not in self.logged_students:
                            # Start initial match timer if not already tracking
                            if sid not in self.match_timestamps:
                                self.match_timestamps[sid] = current_time

                            # Log only if match has lasted continuously for at least 1 second
                            elapsed_match = current_time - self.match_timestamps[sid]
                            if elapsed_match >= self.LOG_VERIFICATION_DELAY:
                                self.logged_students.add(sid)
                                status_str = "Present" if self.current_phase == "on_time" else "Late"
                                self._log_attendance(sid, name, status=status_str)
                                newly_logged = {"id": sid, "name": name, "status": status_str}
                                self.match_timestamps.pop(sid, None)

                if not is_matched:
                    color = (0, 0, 255)
                    label = "Unknown"

                    if current_time - self.last_save_time > self.SAVE_COOLDOWN:
                        self.last_save_time = current_time
                        cv2.imwrite(
                            str(TEMP_DIR / f"unknown_{int(current_time)}.png"),
                            aligned_face,
                        )

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

        # Clear match timer if face leaves the frame or stops matching before 1 second
        for sid in list(self.match_timestamps.keys()):
            if sid not in currently_detected_ids:
                del self.match_timestamps[sid]

        return frame, newly_logged

    def cleanup(self):
        """Releases resources."""
        self.cap.release()
        for item in TEMP_DIR.glob("*"):
            try:
                item.unlink()
            except Exception:
                pass