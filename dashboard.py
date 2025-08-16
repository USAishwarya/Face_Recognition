"""
app_dashboard_recog.py

Unified UI: Dashboard <-> Recognition
Requires:
 - customtkinter
 - matplotlib
 - mysql-connector-python
 - tensorflow / keras (for your model)
 - opencv-python
 - pillow
"""

import os
import time
import threading
from datetime import datetime, timedelta, date
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import mysql.connector
import cv2
import numpy as np
from keras.models import load_model
from threading import Thread, Lock
import dotenv

# ---------------- Config / Env ----------------
dotenv.load_dotenv()  # optional .env support
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "attendance_system")
TABLE_NAME = os.getenv("TABLE_NAME", "logs")  # your table with columns (id,name,date,time)

MODEL_PATH = "best_model.h5"   # path to your trained model

# ---------------- Globals ----------------
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

camera_lock = Lock()
recognition_active = False
frame_skip = 3
attendance_cooldown = 30
last_logged = {}

label_dict = {0: "Aishwarya", 1: "Anoop", 2: "Amma"}  # update per your classes

# ---------------- DB Helpers ----------------
def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )

def fetch_attendance(limit=500):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT id, name, date, time FROM {TABLE_NAME} ORDER BY date DESC, time DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        print("fetch_attendance error:", e)
        return []

def fetch_today_activities(limit=10):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT name, time FROM {TABLE_NAME} WHERE date = CURDATE() ORDER BY time DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        print("fetch_today_activities error:", e)
        return []

def fetch_stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE date = CURDATE()")
        checks_today = cur.fetchone()[0] or 0
        cur.execute(f"SELECT COUNT(DISTINCT name) FROM {TABLE_NAME} WHERE date = CURDATE()")
        unique_today = cur.fetchone()[0] or 0
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE YEARWEEK(date,1) = YEARWEEK(CURDATE(),1)")
        checks_week = cur.fetchone()[0] or 0
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE MONTH(date)=MONTH(CURDATE()) AND YEAR(date)=YEAR(CURDATE())")
        checks_month = cur.fetchone()[0] or 0
        cur.close()
        conn.close()
        return {"unique_today": unique_today, "checks_today": checks_today, "checks_week": checks_week, "checks_month": checks_month}
    except Exception as e:
        print("fetch_stats error:", e)
        return {"unique_today":0, "checks_today":0, "checks_week":0, "checks_month":0}

def fetch_weekly_counts(days=7):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        start_date = (date.today() - timedelta(days=days-1)).strftime("%Y-%m-%d")
        cur.execute(f"SELECT date, COUNT(*) FROM {TABLE_NAME} WHERE date >= %s GROUP BY date ORDER BY date ASC", (start_date,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        counts = {r[0].strftime("%Y-%m-%d") if isinstance(r[0], (date,)) else str(r[0]): r[1] for r in rows}
        result = []
        for i in range(days):
            d = date.today() - timedelta(days=days-1-i)
            ds = d.strftime("%Y-%m-%d")
            result.append((ds, counts.get(ds, 0)))
        return result
    except Exception as e:
        print("fetch_weekly_counts error:", e)
        return [( (date.today() - timedelta(days=days-1-i)).strftime("%Y-%m-%d"), 0) for i in range(days)]

# ---------------- Attendance logging (thread-safe) ----------------
def log_attendance(name):
    """Insert record only once per day; each call uses its own DB connection."""
    global last_logged
    now_ts = time.time()
    if name in last_logged and now_ts - last_logged[name] < attendance_cooldown:
        return
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        today = date.today().strftime("%Y-%m-%d")
        cur.execute(f"SELECT id FROM {TABLE_NAME} WHERE name=%s AND date=%s LIMIT 1", (name, today))
        if cur.fetchone() is None:
            now = datetime.now()
            cur.execute(f"INSERT INTO {TABLE_NAME} (name, date, time) VALUES (%s, %s, %s)", (name, today, now.strftime("%H:%M:%S")))
            conn.commit()
            print(f"[LOGGED] {name} at {now}")
        else:
            print(f"[SKIP] {name} already logged today")
        last_logged[name] = now_ts
    except Exception as e:
        print("log_attendance error:", e)
    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass

# ---------------- Model load ----------------
try:
    model = load_model(MODEL_PATH)
    print("Model loaded")
except Exception as e:
    print("Model load error:", e)
    model = None

# ---------------- Face recognition ----------------
def recognize_loop():
    """Run capture+recognition in background thread."""
    global recognition_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera open failed")
        recognition_active = False
        return

    frame_count = 0
    recognition_active = True

    # use Haar cascade if available
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while recognition_active:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            try:
                img = cv2.resize(face_roi, (224,224))
                img = img/255.0
                img = np.expand_dims(img, axis=0)
                if model is not None:
                    pred = np.argmax(model.predict(img, verbose=0))
                    name = label_dict.get(pred, "Unknown")
                else:
                    name = "Unknown"
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                # log attendance in separate thread
                Thread(target=log_attendance, args=(name,), daemon=True).start()
            except Exception as e:
                print("Recognition error:", e)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognition_active = False

def start_recognition():
    global recognition_active
    if recognition_active:
        messagebox.showinfo("Recognition", "Recognition already running")
        return
    Thread(target=recognize_loop, daemon=True).start()
    recognition_status_var.set("Running")

def stop_recognition():
    global recognition_active
    recognition_active = False
    recognition_status_var.set("Stopped")

# ---------------- UI ----------------
root = ctk.CTk()
root.title("Face Recognition Attendance System")
root.geometry("1200x760")
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Sidebar
sidebar = ctk.CTkFrame(root, width=200, corner_radius=0)
sidebar.grid(row=0, column=0, sticky="nswe", rowspan=2)
ctk.CTkLabel(sidebar, text="NEO", font=("Helvetica", 20, "bold"), text_color="#2E86C1").pack(pady=(20,10), anchor="w", padx=16)
ctk.CTkLabel(sidebar, text="Attendance System", font=("Helvetica", 10), text_color="#7F8C8D").pack(anchor="w", padx=16)

# Buttons on sidebar
def show_dashboard():
    dashboard_frame.tkraise()
    refresh_dashboard()

def show_recognition():
    recognition_frame.tkraise()

ctk.CTkButton(sidebar, text="  🏠  Dashboard", command=show_dashboard, anchor="w").pack(fill="x", padx=12, pady=6)
ctk.CTkButton(sidebar, text="  🎥  Recognition", command=show_recognition, anchor="w").pack(fill="x", padx=12, pady=6)
ctk.CTkButton(sidebar, text="  📋  View Logs", command=lambda: open_logs_window(), anchor="w").pack(fill="x", padx=12, pady=6)
ctk.CTkButton(sidebar, text="  ⎋  Exit", command=root.destroy, fg_color="transparent", text_color="#E74C3C").pack(side="bottom", fill="x", padx=12, pady=12)

# Header
header = ctk.CTkFrame(root, height=60, fg_color="white")
header.grid(row=0, column=1, sticky="nwe", padx=16, pady=5)
ctk.CTkLabel(header, text="Face Recognition Attendance", font=("Helvetica", 16, "bold")).pack(side="left", padx=12)
date_label = ctk.CTkLabel(header, text=datetime.now().strftime("%A, %d %B %Y"), text_color="#7F8C8D")
date_label.pack(side="left", padx=8)

# Main area (stacked frames)
container = ctk.CTkFrame(root, fg_color="transparent")
container.grid(row=1, column=1, sticky="nsew", padx=16, pady=(0,16))
container.grid_rowconfigure(1, weight=1)
container.grid_columnconfigure(0, weight=1)

# Recognition Frame
recognition_frame = ctk.CTkFrame(container, corner_radius=12)
recognition_frame.grid(row=0, column=0, sticky="nsew")
ctk.CTkLabel(recognition_frame, text="Recognition", font=("Arial", 16, "bold")).pack(pady=(10,6), anchor="w", padx=12)

btns = ctk.CTkFrame(recognition_frame, fg_color="transparent")
btns.pack(padx=12, pady=6, anchor="w")
ctk.CTkButton(btns, text="Start Recognition", command=start_recognition, fg_color="#2ECC71", width=160).grid(row=0, column=0, padx=8)
ctk.CTkButton(btns, text="Stop Recognition", command=stop_recognition, fg_color="#E74C3C", width=160).grid(row=0, column=1, padx=8)

recognition_status_var = tk.StringVar(value="Stopped")
ctk.CTkLabel(recognition_frame, textvariable=recognition_status_var, font=("Arial", 12)).pack(pady=(6,12), anchor="w", padx=12)

ctk.CTkLabel(recognition_frame, text="Camera feed will open in a separate window.", font=("Arial", 10), text_color="#7F8C8D").pack(anchor="w", padx=12)

# Dashboard Frame (stacked on top of recognition_frame)
dashboard_frame = ctk.CTkFrame(container, corner_radius=12)
dashboard_frame.grid(row=0, column=0, sticky="nsew")  # same grid as recognition_frame; tkraise to show
dashboard_frame.grid_rowconfigure(2, weight=1)
dashboard_frame.grid_columnconfigure(0, weight=1)

# Dashboard top cards (timesheet, stats, activity)
cards = ctk.CTkFrame(dashboard_frame, fg_color="transparent")
cards.pack(fill="x", pady=(8,10), padx=8)
cards.grid_columnconfigure((0,1,2), weight=1)

# Timesheet card
timesheet = ctk.CTkFrame(cards, corner_radius=12, height=220)
timesheet.grid(row=0, column=0, padx=8, pady=5, sticky="nsew")
ctk.CTkLabel(timesheet, text="TIMESHEET", font=("Arial", 12, "bold"), text_color="#7F8C8D").grid(row=0, column=0, padx=15, pady=10, sticky="w")

# circular progress (small canvas with white background)
canvas_frame = ctk.CTkFrame(timesheet, fg_color="transparent")
canvas_frame.grid(row=1, column=0, pady=(0,10))
canvas_ring = tk.Canvas(canvas_frame, width=120, height=120, bg="white", highlightthickness=0)
canvas_ring.pack()
canvas_ring.create_oval(10, 10, 110, 110, outline="#EBEDEF", width=12)
# a placeholder percent derived from total checkins vs expected (for visual only)
stats_data = fetch_stats()
# avoid division by zero
percent = (stats_data["checks_today"] / 8.0) * 100 if stats_data["checks_today"] > 0 else 43.0
if percent > 100: percent = 100
# draw arc (extent based on percent)
canvas_ring.create_arc(10, 10, 110, 110, start=90, extent=- (360 * percent/100), style="arc", outline="#2ECC71", width=12)
ctk.CTkLabel(canvas_frame, text=f"{stats_data['checks_today']} checks", font=("Helvetica", 16, "bold"), text_color="#27AE60").pack(pady=(8,0))
ctk.CTkLabel(timesheet, text=f"Last check: {datetime.now().strftime('%d %b %Y %I:%M %p')}", font=("Arial", 10), text_color="#7F8C8D").grid(row=2, column=0, pady=(0,10))
ctk.CTkButton(timesheet, text="Punch Out", fg_color="#2ECC71", hover_color="#27AE60", height=35, font=("Arial", 12, "bold")).grid(row=3, column=0, padx=20, pady=10, sticky="ew")

# Stats card
stats_card = ctk.CTkFrame(cards, corner_radius=12)
stats_card.grid(row=0, column=1, padx=8, pady=6, sticky="nsew")
ctk.CTkLabel(stats_card, text="STATISTICS", font=("Arial", 11, "bold"), text_color="#7F8C8D").pack(anchor="w", padx=12, pady=(8,4))
stats_container = ctk.CTkFrame(stats_card, fg_color="transparent")
stats_container.pack(fill="x", padx=6, pady=6)

# Activity card
activity_card = ctk.CTkFrame(cards, corner_radius=12)
activity_card.grid(row=0, column=2, padx=8, pady=6, sticky="nsew")
ctk.CTkLabel(activity_card, text="TODAY'S ACTIVITY", font=("Arial", 11, "bold"), text_color="#7F8C8D").pack(anchor="w", padx=12, pady=(8,4))
activity_container = ctk.CTkFrame(activity_card, fg_color="transparent")
activity_container.pack(fill="both", expand=True, padx=6, pady=6)

# Table area
table_card = ctk.CTkFrame(dashboard_frame, corner_radius=12)
table_card.pack(fill="both", expand=True, padx=8, pady=(0,10))
table_card.grid_rowconfigure(0, weight=1)
table_card.grid_columnconfigure(0, weight=1)

header_frame = ctk.CTkFrame(table_card, fg_color="transparent")
header_frame.pack(fill="x", padx=8, pady=8)
ctk.CTkLabel(header_frame, text="ATTENDANCE RECORDS", font=("Arial", 12, "bold"), text_color="#7F8C8D").pack(side="left", padx=6)

refresh_btn = ctk.CTkButton(header_frame, text="Refresh", width=100, command=lambda: refresh_dashboard())
refresh_btn.pack(side="right", padx=8)

cols = ["ID", "Name", "Date", "Time"]
tree = ttk.Treeview(table_card, columns=cols, show="headings", height=10)
for i,c in enumerate(cols):
    tree.heading(c, text=c)
    tree.column(c, anchor="center", width=(60 if i==0 else 200))
vsb = ttk.Scrollbar(table_card, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=vsb.set)
vsb.pack(side="right", fill="y")
tree.pack(fill="both", expand=True, padx=8, pady=(0,8))

# Chart area
chart_card = ctk.CTkFrame(dashboard_frame, corner_radius=12, height=200)
chart_card.pack(fill="x", padx=8, pady=(0,8))
ctk.CTkLabel(chart_card, text="WEEKLY CHECK-INS (last 7 days)", font=("Arial", 11, "bold"), text_color="#7F8C8D").pack(anchor="w", padx=12, pady=(8,4))
fig, ax = plt.subplots(figsize=(8,2.5), dpi=90)
chart_canvas = FigureCanvasTkAgg(fig, master=chart_card)
chart_canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=(0,12))

# ---------------- Dashboard refresh logic ----------------
def clear_container(container):
    for w in container.winfo_children():
        w.destroy()

def refresh_dashboard():
    try:
        stats_d = fetch_stats()
        weekly = fetch_weekly_counts(7)
        rows = fetch_attendance(limit=500)
        today_acts = fetch_today_activities(10)

        # update timesheet arc + labels
        canvas_ring.delete("arc")
        checks = stats_d.get("checks_today", 0)
        percent = min(100, (checks / 8.0) * 100 if checks>0 else 43)
        canvas_ring.create_arc(10,10,110,110, start=90, extent=-(360*percent/100), style="arc", outline="#2ECC71", width=12, tags="arc")
        # update timesheet text labels (first two children are canvas anqd text labels)
        for child in timesheet.winfo_children():
            # we set the second label text to checks and third to last check
            pass
        # update stat rows
        clear_container(stats_container)
        fill_stat_rows = [
            ("Unique today", stats_d.get("unique_today",0)),
            ("Checks today", stats_d.get("checks_today",0)),
            ("This week", stats_d.get("checks_week",0)),
            ("This month", stats_d.get("checks_month",0)),
        ]
        for k,v in fill_stat_rows:
            row = ctk.CTkFrame(stats_container, fg_color="transparent")
            row.pack(fill="x", padx=6, pady=5)
            ctk.CTkLabel(row, text=k, font=("Arial",11), text_color="#2C3E50").pack(side="left")
            ctk.CTkLabel(row, text=str(v), font=("Arial",11,"bold"), text_color="#2C3E50").pack(side="right")

        # update activity list
        clear_container(activity_container)
        if not today_acts:
            ctk.CTkLabel(activity_container, text="No check-ins today", font=("Arial",11), text_color="#7F8C8D").pack(anchor="w", padx=6, pady=4)
        else:
            for (nm, tm) in today_acts:
                row = ctk.CTkFrame(activity_container, fg_color="transparent")
                row.pack(fill="x", padx=6, pady=4)
                dot = tk.Canvas(row, width=10, height=10, bg="white", highlightthickness=0)
                dot.create_oval(2,2,8,8, fill="#2ECC71")
                dot.pack(side="left", padx=(4,8))
                ctk.CTkLabel(row, text=nm, font=("Arial",11), text_color="#2C3E50").pack(side="left")
                ctk.CTkLabel(row, text=str(tm), font=("Arial",11), text_color="#7F8C8D").pack(side="right", padx=8)

        # update table
        for r in tree.get_children():
            tree.delete(r)
        for r in rows:
            tree.insert("", tk.END, values=r)

        # update chart
        ax.clear()
        days = [d for d,_ in weekly]
        labels = [datetime.strptime(d, "%Y-%m-%d").strftime("%a") for d,_ in weekly]
        counts = [c for _,c in weekly]
        bars = ax.bar(labels, counts, color="#3498DB")
        ax.set_ylim(0, max(counts)+1 if counts else 1)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{int(h)}", xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        chart_canvas.draw()

    except Exception as e:
        print("refresh_dashboard error:", e)
        messagebox.showerror("Error", f"Failed to load dashboard data:\n{e}")

# ---------------- Logs window ----------------
def open_logs_window():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT id, name, date, time FROM {TABLE_NAME} ORDER BY date DESC, time DESC LIMIT 1000")
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        print("open_logs_window DB error:", e)
        messagebox.showerror("DB Error", str(e))
        return

    win = tk.Toplevel(root)
    win.title("Attendance Logs")
    win.geometry("900x600")
    frame = ttk.Frame(win)
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    tree2 = ttk.Treeview(frame, columns=("ID","Name","Date","Time"), show="headings")
    for c in ("ID","Name","Date","Time"):
        tree2.heading(c, text=c)
        tree2.column(c, width=150 if c!="ID" else 60, anchor="center")
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree2.yview)
    tree2.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    tree2.pack(fill="both", expand=True)
    for r in rows:
        tree2.insert("", tk.END, values=r)
    ttk.Button(win, text="Export CSV", command=lambda: export_logs(rows)).pack(pady=8)

def export_logs(rows):
    import csv
    fn = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ID","Name","Date","Time"])
            w.writerows(rows)
        messagebox.showinfo("Export", f"Exported to {fn}")
    except Exception as e:
        messagebox.showerror("Export error", str(e))

# ---------------- Startup ----------------
# Show recognition by default
recognition_frame.tkraise()

# Run initial dashboard load in background so UI appears quickly
Thread(target=refresh_dashboard, daemon=True).start()

root.mainloop()
