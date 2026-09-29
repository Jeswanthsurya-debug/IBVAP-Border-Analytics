"""IBVAP pipeline: detect -> track -> rules/ANPR/face/night -> annotate. One call per frame."""
import copy
import cv2, numpy as np
from .tracker import IoUTracker, iou
from .rules import RuleEngine, Event
from .night import brightness, enhance, MotionDetector
from . import anpr

COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PLATE_VEHICLES = {"car", "motorcycle", "bus", "truck"}

DEFAULT_CFG = {
    "weights": "yolov8n.pt", "device": "cpu", "conf": 0.35, "imgsz": 640,
    # all coordinates normalised 0..1 so they work at any resolution
    "zones": [{"name": "Restricted Zone", "poly": [[0.55, 0.30], [0.98, 0.30], [0.98, 0.98], [0.55, 0.98]]}],
    "tripwires": [{"name": "Border Line", "a": [0.55, 0.20], "b": [0.55, 1.0]}],
    "intrusion_frames": 3, "cooldown_s": 10.0,
    "loiter_s": 8.0, "loiter_radius": 0.04, "run_speed": 0.30, "group_n": 3,
    "night_brightness": 60, "min_motion_area": 0.002,
    "enable_anpr": True, "enable_face": True, "enable_night": True,
    "watchlist_plates": ["TN09AB1234"],
}

_DET = {}


class YoloDetector:
    def __init__(self, weights, device):
        from ultralytics import YOLO
        self.model, self.device, self.conf, self.imgsz = YOLO(weights), device, 0.35, 640

    def __call__(self, frame):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, classes=list(COCO),
                               device=self.device, verbose=False)[0]
        return [(tuple(b.xyxy[0].tolist()), COCO[int(b.cls)], float(b.conf)) for b in r.boxes]


def get_detector(cfg):
    key = (cfg["weights"], cfg["device"])
    if key not in _DET:
        _DET[key] = YoloDetector(*key)
    d = _DET[key]
    d.conf, d.imgsz = cfg["conf"], cfg["imgsz"]
    return d


class Pipeline:
    def __init__(self, cfg=None, detector=None):
        self.cfg = {**copy.deepcopy(DEFAULT_CFG), **(cfg or {})}
        self.det = detector or get_detector(self.cfg)
        self.tracker, self.rules = IoUTracker(), RuleEngine(self.cfg)
        self.motion = MotionDetector(self.cfg["min_motion_area"])
        self.face_cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.n, self.night, self.stats = 0, False, {"person": 0, "vehicle": 0}
        self.plate_tries = {}

    # ---------------------------------------------------------------- per frame
    def process(self, frame, t):
        c, ev = self.cfg, []
        H, W = frame.shape[:2]
        self.n += 1
        self.night = brightness(frame) < c["night_brightness"]
        work = enhance(frame) if self.night else frame
        tracks = self.tracker.update(self.det(work), t)
        ev += self.rules.evaluate(tracks, t, (W, H))
        motion = self.motion.detect(frame)

        for tr in tracks:
            if tr.hits < 3:
                continue
            if c["enable_face"] and tr.cls == "person":
                ev += self._face(tr, work, t)
            if c["enable_anpr"] and tr.cls in PLATE_VEHICLES:
                ev += self._plate(tr, work, t)
        if c["enable_night"] and self.night:
            ev += self._night_motion(motion, tracks, t, (W, H))

        self.stats = {"person": sum(t_.cls == "person" for t_ in tracks),
                      "vehicle": sum(t_.cls != "person" for t_ in tracks)}
        return self._draw(frame, tracks, motion if self.night else [], (W, H)), ev

    # ---------------------------------------------------------------- modules
    def _face(self, tr, img, t):
        x1, y1, x2, y2 = map(int, tr.box)
        if y2 - y1 < 90 or self.n % 5 or tr.meta.get("face"):
            return []
        head = img[max(0, y1):y1 + int((y2 - y1) * 0.45), max(0, x1):x2]
        if head.size == 0:
            return []
        head = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
        faces = self.face_cc.detectMultiScale(head, 1.1, 5, minSize=(24, 24))
        if len(faces):
            fx, fy, fw, fh = faces[0]
            tr.meta["face"] = (x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh)
            return [Event(t, "FACE_DETECTED", "INFO", "face detected on tracked person", tr.id, tr.cls, tr.box)]
        return []

    def _plate(self, tr, img, t):
        x1, y1, x2, y2 = map(int, tr.box)
        if tr.meta.get("plate") or x2 - x1 < 100 or self.n % 10 or self.plate_tries.get(tr.id, 0) >= 5:
            return []
        self.plate_tries[tr.id] = self.plate_tries.get(tr.id, 0) + 1
        try:
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                return []
            plate = anpr.read_plate(crop)
        except ImportError:
            self.cfg["enable_anpr"] = False  # easyocr not installed
            return []
        if not plate:
            return []
        tr.meta["plate"] = plate
        out = [Event(t, "PLATE_READ", "INFO", f"plate {plate}", tr.id, tr.cls, tr.box)]
        if plate in {p.upper() for p in self.cfg["watchlist_plates"]}:
            out.append(Event(t, "WATCHLIST_VEHICLE", "CRITICAL", f"watch-listed plate {plate}", tr.id, tr.cls, tr.box))
        return out

    def _night_motion(self, motion, tracks, t, size):
        W, H = size
        out = []
        for b in motion:
            if any(iou(b, tr.box) > 0.1 for tr in tracks):
                continue  # already explained by a detected object
            cx, cy = (b[0] + b[2]) / 2, b[3]
            in_zone = self.rules.in_any_zone((cx, cy), size)
            if self.rules.cooldown_ok(("NIGHT", int(cx / W * 8), int(cy / H * 6)), t):
                out.append(Event(t, "NIGHT_MOTION", "HIGH" if in_zone else "MEDIUM",
                                 "unidentified movement in low light" + (" inside restricted zone" if in_zone else ""),
                                 box=b))
        return out

    # ---------------------------------------------------------------- drawing
    def _draw(self, frame, tracks, motion, size):
        W, H = size
        out = frame.copy()
        ov = out.copy()
        for _, poly in self.rules.zones_px(size):
            cv2.fillPoly(ov, [poly.astype(np.int32)], (0, 0, 255))
        out = cv2.addWeighted(ov, 0.18, out, 0.82, 0)
        for _, poly in self.rules.zones_px(size):
            cv2.polylines(out, [poly.astype(np.int32)], True, (0, 0, 255), 2)
        for w in self.cfg["tripwires"]:
            cv2.line(out, (int(w["a"][0] * W), int(w["a"][1] * H)), (int(w["b"][0] * W), int(w["b"][1] * H)), (0, 255, 255), 2)
        for tr in tracks:
            col = (0, 200, 255) if tr.cls == "person" else (80, 220, 80)
            x1, y1, x2, y2 = map(int, tr.box)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            label = f"#{tr.id} {tr.cls}" + (f" {tr.meta['plate']}" if "plate" in tr.meta else "")
            cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            if tr.meta.get("face"):
                fx1, fy1, fx2, fy2 = tr.meta["face"]
                cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (255, 120, 0), 2)
        for b in motion:
            cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
        mode = "NIGHT (enhanced)" if self.night else "DAY"
        cv2.putText(out, f"IBVAP | {mode} | persons {self.stats['person']} vehicles {self.stats['vehicle']}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out
