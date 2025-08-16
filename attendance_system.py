import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from threading import Thread, Lock
from keras.models import load_model
import mysql.connector
from datetime import datetime, date
import os
import time
import dotenv

# Load environment variables
dotenv.load_dotenv()
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "attendance_system")

# Global variables
label_dict = {0: "Aishwarya", 1: "Anoop", 2: "Amma"}
last_logged = {}
attendance_cooldown = 30  # seconds
frame_skip = 3  # Process every 3rd frame
recognition_active = False
camera_lock = Lock()

# Load face detection cascade
try:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
except:
    print("Error loading Haar cascade. Using fallback path.")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def get_db_connection():
    """Create a new database connection per thread"""
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )


def log_attendance(name):
    """Log attendance with thread-safe connection and cooldown check"""
    global last_logged

    current_time = time.time()
    # Check if we should log (new name or cooldown expired)
    if name in last_logged and current_time - last_logged[name] < attendance_cooldown:
        return

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        today = date.today()

        cursor.execute("SELECT * FROM logs WHERE name=%s AND date=%s", (name, today))
        result = cursor.fetchone()

        if not result:
            now = datetime.now()
            log_time = now.strftime('%H:%M:%S')
            cursor.execute("INSERT INTO logs (name, date, time) VALUES (%s, %s, %s)",
                           (name, today, log_time))
            conn.commit()
            print(f"{name} logged at {now}")
            last_logged[name] = current_time
        else:
            print(f"{name} is already logged today.")
            last_logged[name] = current_time

    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


def recognize_face():
    """Face recognition with face detection and cooldown management"""
    global recognition_active

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    frame_count = 0
    recognition_active = True

    while recognition_active:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y + h, x:x + w]

            # Preprocess for model
            img = cv2.resize(face_roi, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict using model
            try:
                pred = np.argmax(model.predict(img, verbose=0))
                name = label_dict[pred]

                # Draw rectangle and name
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Log attendance
                Thread(target=log_attendance, args=(name,), daemon=True).start()

            except Exception as e:
                print(f"Prediction error: {e}")

        cv2.imshow("Real-Time Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognition_active = False


def start_camera_thread():
    """Start recognition thread with thread safety"""
    global recognition_active

    if recognition_active:
        print("Recognition is already running")
        return

    with camera_lock:
        Thread(target=recognize_face, daemon=True).start()


def view_logs():
    """Display attendance logs in GUI window"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM logs ORDER BY date DESC, time DESC")
        rows = cursor.fetchall()

        # Create new window
        log_window = tk.Toplevel(root)
        log_window.title("Attendance Logs")
        log_window.geometry("800x600")

        # Create treeview with scrollbar
        frame = ttk.Frame(log_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tree = ttk.Treeview(frame, columns=("ID", "Name", "Date", "Time"), show="headings")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)

        tree.heading("ID", text="ID")
        tree.heading("Name", text="Name")
        tree.heading("Date", text="Date")
        tree.heading("Time", text="Time")

        tree.column("ID", width=50, anchor=tk.CENTER)
        tree.column("Name", width=150, anchor=tk.W)
        tree.column("Date", width=100, anchor=tk.CENTER)
        tree.column("Time", width=100, anchor=tk.CENTER)

        for row in rows:
            tree.insert("", tk.END, values=row)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # Configure resizing
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Add export button
        export_btn = ttk.Button(log_window, text="Export to CSV",
                                command=lambda: export_logs(rows))
        export_btn.pack(pady=10)

    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


def export_logs(rows):
    """Export logs to CSV file"""
    from csv import writer
    from datetime import datetime

    filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        with open(filename, 'w', newline='') as f:
            csv_writer = writer(f)
            csv_writer.writerow(["ID", "Name", "Date", "Time"])
            csv_writer.writerows(rows)
        print(f"Logs exported to {filename}")
    except Exception as e:
        print(f"Export failed: {e}")


def on_closing():
    """Handle window closing event"""
    global recognition_active
    recognition_active = False
    root.destroy()


# Load model with error handling
try:
    model = load_model('best_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Create main application window
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.state('zoomed')
root.protocol("WM_DELETE_WINDOW", on_closing)

# Set background image
try:
    bg_img = Image.open("bg.jpg")
    bg_img = bg_img.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
    bg_photo = ImageTk.PhotoImage(bg_img)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
except Exception as e:
    print(f"Background image error: {e}. Using solid color.")
    root.configure(bg="#dfc5c4")

# Application title
title = tk.Label(root, text="Face Recognition Attendance System",
                 font=("Helvetica", 30, "bold"), fg="white", bg="#4a7abc")
title.pack(pady=40)

# Button style
style = {"font": ("Arial", 18), "bg": "#4a7abc", "fg": "white",
         "width": 20, "height": 2, "bd": 0, "highlightthickness": 0}

# Buttons
btn_frame = tk.Frame(root, bg="")
btn_frame.pack(pady=20)

start_btn = tk.Button(btn_frame, text="Start Recognition",
                      command=start_camera_thread, **style)
start_btn.grid(row=0, column=0, padx=20, pady=10)

log_btn = tk.Button(btn_frame, text="View Attendance Logs",
                    command=view_logs, **style)
log_btn.grid(row=0, column=1, padx=20, pady=10)

exit_btn = tk.Button(btn_frame, text="Exit System",
                     command=on_closing, **style)
exit_btn.grid(row=1, column=0, columnspan=2, pady=20)

# Status bar
status_var = tk.StringVar()
status_var.set("System Ready")
status_bar = tk.Label(root, textvariable=status_var, bd=1, relief=tk.SUNKEN,
                      anchor=tk.W, font=("Arial", 10), bg="lightgray")
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

root.mainloop()