"""Night-time handling: brightness check, low-light enhancement, classical motion detection."""
import cv2, numpy as np


def brightness(frame):
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def enhance(frame):
    """CLAHE on luminance + gamma lift so the detector can see in low light."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    lut = np.array([255 * (i / 255) ** 0.6 for i in range(256)], np.uint8)
    return cv2.LUT(out, lut)


class MotionDetector:
    """Background subtraction (MOG2). Catches movement even when the detector sees nothing."""
    def __init__(self, min_area_frac=0.002, warmup=30):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
        self.min_area_frac, self.warmup, self.n = min_area_frac, warmup, 0

    def detect(self, frame):
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mask = self.bg.apply(g)
        self.n += 1
        if self.n < self.warmup:
            return []
        mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_frac * frame.shape[0] * frame.shape[1]
        boxes = []
        for c in cnts:
            if cv2.contourArea(c) >= min_area:
                x, y, w, h = cv2.boundingRect(c)
                boxes.append((x, y, x + w, y + h))
        return boxes
