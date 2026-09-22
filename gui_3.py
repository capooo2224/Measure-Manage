import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

from app_7 import FaceTrackerEngine


class ScrollableFrame(tk.Frame):
    """Reusable scrollable container holding child widgets inside `.body`."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.body = tk.Frame(self.canvas)
        self.window = self.canvas.create_window(
            (0, 0), window=self.body, anchor="nw"
        )

        self.body.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.window, width=e.width),
        )


class Application:

    def __init__(self, root):
        self.root = root
        self.root.title("M&M Attendance Tracker")
        self.root.geometry("1600x900")
        self.root.minsize(1024, 600)

        self.engine = FaceTrackerEngine()
        self.student_labels = {}

        # 2-Column Responsive Layout Setup (Column 0: 75% Video, Column 1: 25% Side Panel)
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.frame1 = tk.Frame(root, relief="groove", bd=2, bg="black")
        self.frame1.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.frame2 = tk.Frame(root, relief="groove", bd=2)
        self.frame2.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)

        self.video_label = tk.Label(self.frame1, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # Configure Sidebar Grid
        self.frame2.columnconfigure(0, weight=1)
        self.frame2.rowconfigure(1, weight=1)
        self.frame2.rowconfigure(3, weight=1)

        # Panel 1: Present Roster
        tk.Label(
            self.frame2, text="Attendance", font=("Arial", 12, "bold"), anchor="w"
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 2))
        self.scrollframe1 = ScrollableFrame(self.frame2)
        self.scrollframe1.grid(
            row=1, column=0, sticky="nsew", padx=5, pady=(0, 5)
        )

        ttk.Separator(self.frame2, orient="horizontal").grid(
            row=2, column=0, sticky="ew", pady=5
        )

        # Panel 2: Live Activity Log
        tk.Label(
            self.frame2, text="Log", font=("Arial", 12, "bold"), anchor="w"
        ).grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 2))
        self.scrollframe2 = ScrollableFrame(self.frame2)
        self.scrollframe2.grid(
            row=4, column=0, sticky="nsew", padx=5, pady=(0, 10)
        )

        self.populate_expected_roster()
        self.update_feed()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def populate_expected_roster(self):
        """Populates the Present roster with red indicators for all expected students."""
        for student in self.engine.get_all_expected_students():
            sid, name = student["id"], student["name"]
            lbl = tk.Label(
                self.scrollframe1.body,
                text=f"🔴  {sid} - {name}",
                anchor="w",
                font=("Arial", 10),
            )
            lbl.pack(fill="x", padx=10, pady=2)
            self.student_labels[sid] = {
                "label": lbl,
                "name": name,
                "status": False,
            }

    def update_feed(self):
        """Main event loop: fetches frame, updates UI status, scales and renders video stream."""
        frame, newly_logged = self.engine.process_next_frame()

        if frame is not None:
            # Update circle indicator from Red to Green on new match event
            if newly_logged and newly_logged["id"] in self.student_labels:
                sid = newly_logged["id"]
                rec = self.student_labels[sid]

                if not rec["status"]:
                    rec["label"].configure(text=f"🟢  {sid} - {rec['name']}")
                    rec["status"] = True

                    tk.Label(
                        self.scrollframe2.body,
                        text=f"[LOG] {sid} marked Present",
                        anchor="w",
                        font=("Arial", 9),
                        fg="green",
                    ).pack(fill="x", padx=10, pady=1)

            # Convert OpenCV frame (BGR) to PIL format (RGB) for Tkinter canvas
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
            if w > 10 and h > 10:
                img = img.resize((w, h), Image.Resampling.BILINEAR)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Queue next frame in 15 milliseconds
        self.root.after(15, self.update_feed)

    def on_close(self):
        self.engine.cleanup()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()