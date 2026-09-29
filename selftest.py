"""Self-test with a fake detector: proves rules, night motion, ANPR logic work without a camera/GPU.
Run:  python selftest.py          (add --yolo to also test the real YOLO model, --ocr for EasyOCR)"""
import sys
import numpy as np, cv2
from ibvap import Pipeline
from ibvap.anpr import extract_plate

W, H, FPS = 640, 360, 10


class Fake:
    """Scripted detections: A walks across the fence, B stands still, C sprints."""
    def __call__(self, frame):
        t = self.i / FPS
        d = []
        ax = 100 + 60 * t                                  # slow walker, crosses tripwire at x=352
        d.append(((ax - 20, 200, ax + 20, 300), "person", .9))
        d.append(((100, 100, 140, 200), "person", .9))     # stationary loiterer
        if 2 <= t <= 4:
            cx = 20 + 300 * (t - 2)                        # fast runner (~300px/s = 0.47 widths/s)
            d.append(((cx - 20, 50, cx + 20, 150), "person", .9))
        self.i += 1
        return d


def run_day():
    p = Pipeline(detector=Fake()); p.det.i = 0
    types = set()
    for i in range(130):
        frame = np.full((H, W, 3), 120, np.uint8)
        _, ev = p.process(frame, i / FPS)
        types |= {e.type for e in ev}
    return types


def run_night():
    class Nothing:
        def __call__(self, f): return []
    p = Pipeline(detector=Nothing())
    types = set()
    for i in range(80):
        frame = np.full((H, W, 3), 15, np.uint8)
        if i > 40:                                          # something moves in the dark, inside zone
            x = 380 + (i - 40) * 4
            cv2.rectangle(frame, (x, 200), (x + 40, 280), (90, 90, 90), -1)
        _, ev = p.process(frame, i / FPS)
        types |= {e.type for e in ev}
    return types


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    return cond


if __name__ == "__main__":
    ok = True
    d = run_day()
    for k in ("INTRUSION", "TRIPWIRE", "LOITERING", "RUNNING"):
        ok &= check(f"day rule: {k}", k in d)
    ok &= check("night motion alert", "NIGHT_MOTION" in run_night())
    for raw, want in (("TNO9ABI234", "TN09AB1234"), ("xx KA01MG5 678 yy", "KA01MG5678"), ("MH12DE1433", "MH12DE1433"), ("DL8CAF5O3O", "DL8CAF5030"), ("HELLO WORLD", None)):
        ok &= check(f"ANPR text fix {raw!r} -> {want}", extract_plate(raw) == want)
    if "--ocr" in sys.argv:
        from ibvap.anpr import read_plate
        img = np.full((200, 400, 3), 60, np.uint8)
        cv2.rectangle(img, (60, 110), (340, 170), (255, 255, 255), -1)
        cv2.putText(img, "TN09AB1234", (70, 155), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3)
        r = read_plate(img); print("OCR result:", r); ok &= check("EasyOCR reads synthetic plate", r == "TN09AB1234")
    if "--yolo" in sys.argv:
        p = Pipeline(); out, ev = p.process(np.zeros((H, W, 3), np.uint8), 0.0)
        ok &= check("real YOLO model loads and runs", out.shape[:2] == (H, W))
    print("\nALL PASSED" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
