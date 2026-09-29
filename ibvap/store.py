"""Event log (SQLite + snapshots) and command-and-control webhook integration."""
import csv, json, os, sqlite3, threading, urllib.request
from datetime import datetime
import cv2


class EventStore:
    def __init__(self, db="ibvap_events.db", snap_dir="snapshots"):
        self.snap_dir = snap_dir
        os.makedirs(snap_dir, exist_ok=True)
        self.db = sqlite3.connect(db, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, wall TEXT,
            media_t REAL, type TEXT, severity TEXT, cls TEXT, track_id INT, detail TEXT, snapshot TEXT)""")
        self.db.commit()

    def add(self, ev, frame=None, camera="CAM-01", webhook=""):
        wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snap = ""
        if frame is not None and ev.severity != "INFO":
            snap = os.path.join(self.snap_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{ev.type}.jpg")
            cv2.imwrite(snap, frame)
        with self.lock:
            self.db.execute("INSERT INTO events(wall,media_t,type,severity,cls,track_id,detail,snapshot) VALUES(?,?,?,?,?,?,?,?)",
                            (wall, ev.ts, ev.type, ev.severity, ev.cls, ev.track_id, ev.detail, snap))
            self.db.commit()
        if webhook and ev.severity != "INFO":
            payload = {"camera": camera, "time": wall, "type": ev.type, "severity": ev.severity,
                       "object": ev.cls, "detail": ev.detail, "snapshot": snap}
            threading.Thread(target=self._post, args=(webhook, payload), daemon=True).start()

    @staticmethod
    def _post(url, payload):
        try:
            req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass

    def recent(self, n=15):
        with self.lock:
            return self.db.execute("SELECT wall,severity,type,cls,detail,snapshot FROM events ORDER BY id DESC LIMIT ?", (n,)).fetchall()

    def count(self):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM events WHERE severity!='INFO'").fetchone()[0]

    def export_csv(self, path="ibvap_events.csv"):
        with self.lock:
            rows = self.db.execute("SELECT wall,media_t,type,severity,cls,track_id,detail,snapshot FROM events").fetchall()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "media_t", "type", "severity", "object", "track_id", "detail", "snapshot"])
            w.writerows(rows)
        return path
