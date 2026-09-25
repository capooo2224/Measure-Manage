import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

# Import backend engine from app_11_3
from app_13 import FaceTrackerEngine


class ScrollableFrame(tk.Frame):
    """A frame that scrolls. Put your widgets in `.body`, not in this."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._canvas = tk.Canvas(self, highlightthickness=0)
        self._bar = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._bar.set)
        self._bar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self.body = tk.Frame(self._canvas)
        self._window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._on_body)

    def _on_body(self, _event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))


class Application:
    def __init__(self, root):
        self.root = root
        root.title("Attendance Tracker System")
        
        # Maximize window across full screen dynamically
        try:
            root.state('zoomed')
        except Exception:
            root.geometry("1024x768")

        # Configure Grid Layout Weight for Full Resizing
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=0)
        root.grid_columnconfigure(0, weight=3)
        root.grid_columnconfigure(1, weight=1)

        # Initialize Backend Engine
        try:
            self.engine = FaceTrackerEngine()
        except RuntimeError as e:
            messagebox.showerror("Error", str(e))
            root.destroy()
            return

        self.selected_group = tk.StringVar(value="")
        self.active_group = None  # Tracks the currently locked group session
        self.timer_running = False
        self.roster_labels = {}

        # --- Top Section: Live Video and Group Selection Sidebar ---
        self.Camera = tk.Label(root, bg="black")
        self.Camera.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.CSV_file_select = ScrollableFrame(root)
        self.CSV_file_select.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # --- Bottom Section: Controls & Inputs ---
        self.control_frame = tk.Frame(root, relief="groove", bd=2)
        self.control_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        self.control_frame.grid_columnconfigure(1, weight=1)

        # Dual Timer Inputs (On-Time & Late Grace Period)
        timer_frame = tk.Frame(self.control_frame)
        timer_frame.pack(side="left", padx=15, pady=10)

        # Timer 1: On-Time
        tk.Label(timer_frame, text="Timer 1 - On Time (hrs : mins):", font=("Helvetica", 8, "bold")).pack(anchor="w")
        input_container1 = tk.Frame(timer_frame)
        input_container1.pack(anchor="w", pady=2)

        self.input_hours = tk.Entry(input_container1, width=4)
        self.input_hours.insert(0, "0")
        self.input_hours.pack(side="left")

        tk.Label(input_container1, text=" hrs ").pack(side="left")

        self.input_mins = tk.Entry(input_container1, width=4)
        self.input_mins.insert(0, "10")  # Default to 10 mins
        self.input_mins.pack(side="left")

        tk.Label(input_container1, text=" mins").pack(side="left")

        # Timer 2: Late Grace Period
        tk.Label(timer_frame, text="Timer 2 - Late Grace (hrs : mins):", font=("Helvetica", 8, "bold"), fg="#b35900").pack(anchor="w", pady=(5, 0))
        input_container2 = tk.Frame(timer_frame)
        input_container2.pack(anchor="w", pady=2)

        self.input_grace_hours = tk.Entry(input_container2, width=4)
        self.input_grace_hours.insert(0, "0")
        self.input_grace_hours.pack(side="left")

        tk.Label(input_container2, text=" hrs ").pack(side="left")

        self.input_grace_mins = tk.Entry(input_container2, width=4)
        self.input_grace_mins.insert(0, "5")  # Default to 5 mins late buffer
        self.input_grace_mins.pack(side="left")

        tk.Label(input_container2, text=" mins").pack(side="left")

        # Session Buttons: Start | End | Discard
        self.Start = tk.Button(self.control_frame, text="Start", command=self.on_Start, bg="#e1e1e1", width=10, height=2)
        self.Start.pack(side="left", padx=5)

        self.End = tk.Button(self.control_frame, text="End", command=self.on_End, bg="#e1e1e1", width=10, height=2)
        self.End.pack(side="left", padx=5)

        self.Discard = tk.Button(self.control_frame, text="Discard", command=self.on_Discard, bg="#ffcccc", width=10, height=2)
        self.Discard.pack(side="left", padx=5)

        # Progress / Timer Status Bar
        progress_container = tk.Frame(self.control_frame)
        progress_container.pack(side="left", fill="x", expand=True, padx=20)

        self.label2 = tk.Label(progress_container, text="Status: Ready", anchor="w")
        self.label2.pack(fill="x", pady=2)

        self.countdown = ttk.Progressbar(progress_container, orient="horizontal", mode="determinate")
        self.countdown.pack(fill="x", pady=2)

        # Dismiss Panel
        dismiss_frame = tk.Frame(self.control_frame)
        dismiss_frame.pack(side="right", padx=15, pady=10)

        self.dismiss_title = tk.Label(dismiss_frame, text="Dismiss students?", font=("Helvetica", 9, "bold"))
        self.dismiss_title.pack(anchor="e")

        self.button1 = tk.Button(dismiss_frame, text="Dismiss", command=self.on_button1, bg="#e1e1e1", width=12, height=2)
        self.button1.pack(anchor="e", pady=2)

        # Populate Available Database Groups
        self.load_group_selection_ui()

        # Handle Window Close Gracefully
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start Video Feed Loop
        self.update_video_feed()

    def load_group_selection_ui(self):
        """Populates side list with group radio buttons and auto-displays students upon selection."""
        groups = self.engine.get_available_groups()

        for widget in self.CSV_file_select.body.winfo_children():
            widget.destroy()

        title = tk.Label(self.CSV_file_select.body, text="Select Group:", font=("Helvetica", 10, "bold"))
        title.pack(anchor="w", padx=5, pady=5)

        if not groups:
            no_grp = tk.Label(self.CSV_file_select.body, text="No group folders\nfound in\nExpected_faces_db", fg="red")
            no_grp.pack(anchor="w", padx=5, pady=5)
            return

        self.selected_group.set(groups[0])
        for grp in groups:
            rb = tk.Radiobutton(
                self.CSV_file_select.body,
                text=grp,
                value=grp,
                variable=self.selected_group,
                command=self.on_group_selected,
                font=("Helvetica", 10)
            )
            rb.pack(anchor="w", padx=5, pady=2)

        self.student_list_frame = tk.Frame(self.CSV_file_select.body)
        self.student_list_frame.pack(fill="both", expand=True, padx=5, pady=10)

        self.on_group_selected()

    def on_group_selected(self):
        """Loads and lists student names/IDs under the selected group, preventing swaps during active sessions."""
        new_group = self.selected_group.get()

        # Block group swapping while a timer session is running
        if self.timer_running and self.active_group and new_group != self.active_group:
            self.selected_group.set(self.active_group)  # Revert radio selection back
            messagebox.showwarning(
                "Action Restricted",
                "You cannot swap sections while a session timer is running.\n\nPlease 'End' or 'Discard' the current timer before changing sections."
            )
            return

        if not new_group:
            return

        try:
            self.engine.select_group(new_group)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        for widget in self.student_list_frame.winfo_children():
            widget.destroy()

        self.roster_labels.clear()

        ttk.Separator(self.student_list_frame, orient="horizontal").pack(fill="x", pady=5)
        
        lbl_roster = tk.Label(self.student_list_frame, text=f"Student Roster ({new_group}):", font=("Helvetica", 9, "bold"))
        lbl_roster.pack(anchor="w", pady=(0, 5))

        students = self.engine.get_all_expected_students()
        if not students:
            tk.Label(self.student_list_frame, text="No student images found.", fg="gray").pack(anchor="w")
            return

        for s in students:
            txt = f"• {s['id']} - {s['name']}"
            lbl = tk.Label(self.student_list_frame, text=txt, font=("Helvetica", 9), anchor="w")
            lbl.pack(anchor="w", pady=1)
            self.roster_labels[s["id"]] = lbl

    def on_Start(self):
        """Starts dual-timer session and uses single CSV file created for this app session."""
        group_name = self.selected_group.get()
        if not group_name:
            messagebox.showwarning("Warning", "Please select a group database first.")
            return

        try:
            hrs = float(self.input_hours.get()) if self.input_hours.get() else 0.0
            mins = float(self.input_mins.get()) if self.input_mins.get() else 0.0
            
            grace_hrs = float(self.input_grace_hours.get()) if self.input_grace_hours.get() else 0.0
            grace_mins = float(self.input_grace_mins.get()) if self.input_grace_mins.get() else 0.0

            if hrs < 0 or mins < 0 or grace_hrs < 0 or grace_mins < 0 or (hrs == 0 and mins == 0 and grace_hrs == 0 and grace_mins == 0):
                raise ValueError
        except ValueError:
            messagebox.showwarning("Warning", "Please enter valid non-negative numbers for timer inputs.")
            return

        try:
            self.engine.start_session(
                group_name=group_name,
                init_hrs=hrs, init_mins=mins,
                grace_hrs=grace_hrs, grace_mins=grace_mins
            )
            self.timer_running = True
            self.active_group = group_name  # Lock section during run
            
            initial_secs = self.engine.initial_timer_duration
            self.countdown["maximum"] = initial_secs if initial_secs > 0 else self.engine.grace_timer_duration
            self.countdown["value"] = self.countdown["maximum"]
            
            # NOTE: Removed the loop that previously cleared/reset self.roster_labels back to black text!
            # Existing statuses ([Present], [Late], [Absent]) will now persist and continue updating live.

            messagebox.showinfo(
                "Session Started", 
                f"Logging attendance for group: '{group_name}'\nOn-time Timer: {int(hrs)}h {int(mins)}m\nLate Grace Timer: {int(grace_hrs)}h {int(grace_mins)}m."
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_End(self):
        """Manually stops session timers, sorts CSV entries alphabetically, and keeps recorded entries."""
        if not self.engine.is_running and not self.timer_running:
            return

        # Sort CSV rows alphabetically by student name
        self.engine.sort_csv_alphabetically(sort_by_col=1)

        self.engine.is_running = False
        self.timer_running = False
        self.active_group = None
        self.label2.config(text="Status: Session Paused/Ended (Saved & Sorted)", fg="black")
        self.countdown["value"] = 0
        messagebox.showinfo("Session Ended", "Attendance session paused.\nLogged records have been sorted alphabetically and saved to the session CSV.")

    def on_Discard(self):
        """Clears logged attendance from current CSV file."""
        if not self.engine.active_csv_path:
            messagebox.showwarning("Warning", "No active session log to discard.")
            return

        confirm = messagebox.askyesno(
            "Discard Attendance?",
            "Warning: All logged attendance in the current CSV file will be cleared.\nDo you want to proceed?",
            icon="warning"
        )

        if confirm:
            self.engine.discard_session()
            self.timer_running = False
            self.active_group = None
            self.countdown["value"] = 0
            self.label2.config(text="Status: Attendance Records Cleared", fg="black")
            
            # Reset UI labels
            self.on_group_selected()
            messagebox.showinfo("Discarded", "Current session attendance records were cleared.")

    def on_button1(self):
        """Dismiss Button: Auto-logs all unmarked group members as 'Absent'."""
        if not self.engine.selected_group_folder:
            messagebox.showwarning("Warning", "No active group session. Select a group and click Start first.")
            return

        dismissed_count = self.engine.auto_log_dismissal()
        self.timer_running = False
        self.active_group = None
        self.countdown["value"] = 0
        self.label2.config(text="Status: Session Dismissed", fg="black")

        # Update remaining unlogged labels in UI to show Absent
        for sid, lbl in self.roster_labels.items():
            if sid not in self.engine.logged_students:
                student_name = [s['name'] for s in self.engine.get_all_expected_students() if s['id'] == sid][0]
                lbl.config(text=f"• {sid} - {student_name} [Absent]", fg="red")

        messagebox.showinfo("Dismissed", f"Session ended.\nMarked {dismissed_count} student(s) as Absent.")

    def update_video_feed(self):
        """Main Tkinter update loop for camera feed, roster UI updates, and dual-timer progress."""
        frame, newly_logged = self.engine.process_next_frame()

        if frame is not None:
            cw = self.Camera.winfo_width()
            ch = self.Camera.winfo_height()

            if cw > 10 and ch > 10:
                cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_image)
                img = img.resize((cw, ch), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.Camera.imgtk = imgtk
                self.Camera.configure(image=imgtk)

        # Highlight newly scanned students directly on sidebar
        if newly_logged:
            sid = newly_logged["id"]
            status = newly_logged["status"]
            if sid in self.roster_labels:
                lbl = self.roster_labels[sid]
                student_name = newly_logged["name"]
                color = "green" if status == "Present" else "#b35900"
                lbl.config(text=f"• {sid} - {student_name} [{status}]", fg=color)

        if self.timer_running:
            rem_time, phase = self.engine.get_remaining_time()

            hrs_left = int(rem_time // 3600)
            mins_left = int((rem_time % 3600) // 60)
            secs_left = int(rem_time % 60)

            if phase == "on_time":
                self.countdown["maximum"] = max(1.0, self.engine.initial_timer_duration)
                self.countdown["value"] = rem_time
                self.label2.config(
                    text=f"Phase: On-Time Logging | Time Remaining: {hrs_left:02d}h {mins_left:02d}m {secs_left:02d}s",
                    fg="black"
                )
            elif phase == "late":
                self.countdown["maximum"] = max(1.0, self.engine.grace_timer_duration)
                self.countdown["value"] = rem_time
                self.label2.config(
                    text=f"Phase: LATE Logging | Time Remaining: {hrs_left:02d}h {mins_left:02d}m {secs_left:02d}s",
                    fg="#b35900"
                )
            else:
                self.timer_running = False
                self.countdown["value"] = 0
                self.label2.config(text="Status: Timers Expired (Logging Paused)", fg="red")
                messagebox.showinfo("Timer Expired", "Both On-Time and Late Grace timers have expired. Face logging is paused.")

        self.root.after(30, self.update_video_feed)

    def on_close(self):
        """Clean up video camera and temporary files on close."""
        self.engine.cleanup()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()