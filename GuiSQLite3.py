import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import cv2
from keras.models import load_model
import numpy as np
from threading import Thread
from datetime import datetime, date
import sqlite3

conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        log_date TEXT NOT NULL,
        log_time TEXT NOT NULL
    )
''')
conn.commit()
conn.close()

model = load_model('best_model.h5')

label_dict = {0: "Aishwarya", 1: "Anoop", 2: "Amma"}

last_logged_name = None

def log_user_once_per_day(name):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    today = date.today().strftime('%Y-%m-%d')


    cursor.execute("SELECT * FROM logs WHERE name=? AND log_date=?", (name, today))
    result = cursor.fetchone()

    if result:
        print(f"{name} has already been logged today.")
    else:
        now = datetime.now()
        log_time = now.strftime('%H:%M:%S')

        cursor.execute("INSERT INTO logs (name, log_date, log_time) VALUES (?, ?, ?)", (name, today, log_time))
        conn.commit()
        print(f"Logged in: {name} at {now.strftime('%Y-%m-%d %H:%M:%S')}")

    conn.close()

def recognize_face():
    global last_logged_name
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = np.argmax(model.predict(img))
        name = label_dict[pred]

        x, y = 100, 150
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)
        log_user_once_per_day(name)

        cv2.imshow("Real-Time Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_camera_thread():
    Thread(target=recognize_face).start()

def view_logs():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    conn.close()


root = tk.Tk()
root.title("Face Recognition App")
root.state('zoomed')

bg_img = Image.open("bg.jpg")
bg_img = bg_img.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
bg_photo = ImageTk.PhotoImage(bg_img)


bg_label = Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)


title = Label(root, text="Face Recognition System", font=("Helvetica", 30, "bold"),
              fg="white", bg="#dfc5c4")
title.pack(pady=40)


style = {"font": ("Arial", 20, "bold"), "bg": "#dfc5c4", "fg": "white", "padx": 20, "pady": 10}


start_btn = Button(root, text="Start Recognition", command=start_camera_thread, **style)
start_btn.pack(pady=20)


exit_btn = Button(root, text="Exit", command=root.quit, font=("Arial", 18),
                  bg="#dfc5c4", fg="white", padx=20, pady=10)
exit_btn.pack(pady=10)

log_btn = Button(root, text="View Logs", command=view_logs, **style)
log_btn.pack(pady=10)


root.mainloop()
