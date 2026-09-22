import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

from app_6 import FaceTrackerEngine


class ScrollableFrame(tk.Frame):
    """A frame that scrolls. Put your widgets in `.body`, not in this."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(self, highlightthickness=0)
        self._bar = tk.Scrollbar(
            self, orient="vertical", command=self._canvas.yview
        )
        self._canvas.configure(yscrollcommand=self._bar.set)

        self._bar.grid(row=0, column=1, sticky="ns")
        self._canvas.grid(row=0, column=0, sticky="nsew")

        self.body = tk.Frame(self._canvas)
        self._window = self._canvas.create_window(
            (0, 0), window=self.body, anchor="nw"
        )

        self.body.bind("<Configure>", self._on_body)
        self._canvas.bind("<Configure>", self._on_canvas)

    def _on_body(self, _event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas(self, event):
        self._canvas.itemconfig(self._window, width=event.width)


class Application:

    def __init__(self, root):
        self.root = root
        self.root.title("M&M Attendance Tracker")
        self.root.geometry("1600x900")
        self.root.minsize(1024, 600)

        # Initialize Backend Engine
        self.engine = FaceTrackerEngine()

        # Dictionary to store widget references per student ID: {student_id: tk.Label}
        self.student_labels = {}

        # Window Grid Setup
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Main Containers
        self.frame1 = tk.Frame(root, relief="groove", bd=2, bg="black")
        self.frame1.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.frame2 = tk.Frame(root, relief="groove", bd=2)
        self.frame2.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)

        # Video Surface Label
        self.video_label = tk.Label(self.frame1, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # Sidebar Grid Layout
        self.frame2.columnconfigure(0, weight=1)
        self.frame2.rowconfigure(1, weight=1)  # Present Scrollbox
        self.frame2.rowconfigure(3, weight=1)  # Log Scrollbox

        # Section 1: Present Panel
        self.w_label1 = tk.Label(
            self.frame2, text="Attendance", font=("Arial", 12, "bold"), anchor="w"
        )
        self.w_label1.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 2))

        self.scrollframe1 = ScrollableFrame(self.frame2)
        self.scrollframe1.grid(
            row=1, column=0, sticky="nsew", padx=5, pady=(0, 5)
        )

        self.separator1 = ttk.Separator(self.frame2, orient="horizontal")
        self.separator1.grid(row=2, column=0, sticky="ew", pady=5)

        # Section 2: Log Panel
        self.label2 = tk.Label(
            self.frame2, text="Log", font=("Arial", 12, "bold"), anchor="w"
        )
        self.label2.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 2))

        self.scrollframe2 = ScrollableFrame(self.frame2)
        self.scrollframe2.grid(
            row=4, column=0, sticky="nsew", padx=5, pady=(0, 10)
        )

        # Load Roster into Present Box
        self.populate_expected_roster()

        # Start Video Refresh Loop
        self.update_feed()

        # Exit Protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def populate_expected_roster(self):
        """Loads all pre-registered students into the Present panel marked as Absent (Red)."""
        expected_students = self.engine.get_all_expected_students()

        for student in expected_students:
            student_id = student["id"]
            name = student["name"]

            lbl = tk.Label(
                self.scrollframe1.body,
                text=f"🔴  {student_id} - {name}",
                anchor="w",
                font=("Arial", 10),
            )
            lbl.pack(fill="x", padx=10, pady=2)

            self.student_labels[student_id] = {
                "label": lbl,
                "name": name,
                "status": False,
            }

    def update_feed(self):
        frame, newly_logged = self.engine.process_next_frame()

        if frame is not None:
            # Update circle indicator from Red to Green on match
            if newly_logged is not None:
                student_id = newly_logged["id"]

                if student_id in self.student_labels:
                    record = self.student_labels[student_id]
                    if not record["status"]:
                        record["label"].configure(
                            text=f"🟢  {student_id} - {record['name']}"
                        )
                        record["status"] = True

                        # Log event to bottom scrollbox
                        log_lbl = tk.Label(
                            self.scrollframe2.body,
                            text=f"[LOG] {student_id} marked Present",
                            anchor="w",
                            font=("Arial", 9),
                            fg="green",
                        )
                        log_lbl.pack(fill="x", padx=10, pady=1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)

            target_w = self.video_label.winfo_width()
            target_h = self.video_label.winfo_height()

            if target_w > 10 and target_h > 10:
                img = img.resize(
                    (target_w, target_h), Image.Resampling.BILINEAR
                )

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(15, self.update_feed)

    def on_close(self):
        self.engine.cleanup()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()