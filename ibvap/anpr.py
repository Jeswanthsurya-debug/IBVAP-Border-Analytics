"""ANPR: classical plate localisation (edges + contours) -> OCR -> Indian plate format validation."""
import re, cv2, numpy as np

PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")  # e.g. TN09AB1234, DL8CAF5030
TO_ALPHA = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "4": "A"}
TO_DIGIT = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2", "Q": "0", "D": "0", "L": "1", "A": "4", "G": "6"}
STATE_CODES = set("AN AP AR AS BR CH CG DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY RJ SK TN TR TS UK UA UP WB".split())
_reader = None


def _fix(s, dd=2):
    """Position-aware OCR correction: SS + dd digits + letters + NNNN."""
    n = len(s)
    out = []
    for i, ch in enumerate(s):
        want_alpha = i < 2 or (2 + dd <= i < n - 4)
        if want_alpha and ch.isdigit():
            ch = TO_ALPHA.get(ch, ch)
        elif not want_alpha and ch.isalpha():
            ch = TO_DIGIT.get(ch, ch)
        out.append(ch)
    return "".join(out)


def extract_plate(text):
    """Best valid plate in the OCR text = the candidate needing the fewest character corrections."""
    s = re.sub(r"[^A-Z0-9]", "", text.upper())
    best = None
    for n in (9, 10, 11):
        for i in range(0, len(s) - n + 1):
            raw = s[i:i + n]
            for dd in (2, 1):
                cand = _fix(raw, dd)
                if PLATE_RE.match(cand) and cand[:2] in STATE_CODES:
                    changes = sum(a != b for a, b in zip(raw, cand))
                    if best is None or changes < best[0]:
                        best = (changes, cand)
    return best[1] if best else None


def find_plate_regions(crop):
    g = cv2.bilateralFilter(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 9, 75, 75)
    e = cv2.dilate(cv2.Canny(g, 60, 180), np.ones((3, 3), np.uint8), 1)
    cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = g.shape
    regs = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:25]:
        x, y, cw, ch = cv2.boundingRect(c)
        if 2.0 <= cw / max(ch, 1) <= 6.5 and 0.15 * w < cw < 0.95 * w and ch > 0.04 * h:
            p = 4
            regs.append(crop[max(0, y - p):y + ch + p, max(0, x - p):x + cw + p])
        if len(regs) >= 3:
            break
    regs.append(crop[int(h * 0.5):, :])  # fallback: lower half of vehicle
    return regs


def _ocr(img):
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    if img.shape[0] < 60:
        f = 60 / img.shape[0]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    res = _reader.readtext(img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return " ".join(r[1] for r in res)


def read_plate(vehicle_crop):
    """Returns a valid plate string or None."""
    for reg in find_plate_regions(vehicle_crop):
        if reg.size == 0:
            continue
        p = extract_plate(_ocr(reg))
        if p:
            return p
    return None
