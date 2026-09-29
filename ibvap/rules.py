"""Rule-based behaviour analytics on top of tracked objects (pure geometry + timers)."""
from dataclasses import dataclass
import cv2, numpy as np

MIN_HITS = 3


@dataclass
class Event:
    ts: float
    type: str
    severity: str  # INFO / MEDIUM / HIGH / CRITICAL
    detail: str
    track_id: int = -1
    cls: str = ""
    box: tuple = ()


def _ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(A, B, C, D):
    return _ccw(A, C, D) != _ccw(B, C, D) and _ccw(A, B, C) != _ccw(A, B, D)


class RuleEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.inside, self.last = {}, {}

    def cooldown_ok(self, key, t):
        if t - self.last.get(key, -1e9) < self.cfg["cooldown_s"]:
            return False
        self.last[key] = t
        return True

    def zones_px(self, size):
        W, H = size
        return [(z["name"], np.array([[x * W, y * H] for x, y in z["poly"]], np.float32)) for z in self.cfg["zones"]]

    def in_any_zone(self, pt, size):
        return any(cv2.pointPolygonTest(p, (float(pt[0]), float(pt[1])), False) >= 0 for _, p in self.zones_px(size))

    def evaluate(self, tracks, t, size):
        c, W, H, ev = self.cfg, size[0], size[1], []
        zones = self.zones_px(size)
        wires = [(w["name"], (w["a"][0] * W, w["a"][1] * H), (w["b"][0] * W, w["b"][1] * H)) for w in c["tripwires"]]
        persons_in = {name: [] for name, _ in zones}

        for tr in tracks:
            if tr.hits < MIN_HITS:
                continue
            ax, ay = tr.anchor
            is_person = tr.cls == "person"
            # 1) Virtual fence / restricted zone intrusion
            for name, poly in zones:
                inside = cv2.pointPolygonTest(poly, (float(ax), float(ay)), False) >= 0
                k = (tr.id, name)
                self.inside[k] = self.inside.get(k, 0) + 1 if inside else 0
                if inside and is_person:
                    persons_in[name].append(tr)
                if self.inside[k] == c["intrusion_frames"]:
                    ev.append(Event(t, "INTRUSION", "HIGH" if is_person else "MEDIUM",
                                    f"{tr.cls} entered '{name}'", tr.id, tr.cls, tr.box))
            # 2) Tripwire (border line) crossing
            if len(tr.trail) >= 2:
                p0, p1 = tr.trail[-2][1:], tr.trail[-1][1:]
                for name, a, b in wires:
                    if segments_intersect(p0, p1, a, b) and self.cooldown_ok(("TW", tr.id, name), t):
                        s0 = (b[0] - a[0]) * (p0[1] - a[1]) - (b[1] - a[1]) * (p0[0] - a[0])
                        d = "side 1 -> side 2" if s0 < 0 else "side 2 -> side 1"
                        ev.append(Event(t, "TRIPWIRE", "HIGH", f"{tr.cls} crossed '{name}' ({d})", tr.id, tr.cls, tr.box))
            if not is_person:
                continue
            # 3) Running / abnormal speed (fraction of frame width per second)
            pts = [p for p in tr.trail if p[0] >= t - 1.0]
            if tr.hits >= 8 and len(pts) >= 2 and pts[-1][0] - pts[0][0] >= 0.5:
                dt = pts[-1][0] - pts[0][0]
                sp = np.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]) / dt / W
                if sp > c["run_speed"] and self.cooldown_ok(("RUN", tr.id), t):
                    ev.append(Event(t, "RUNNING", "MEDIUM", f"person moving fast ({sp:.2f} widths/s)", tr.id, tr.cls, tr.box))
            # 4) Loitering: stays inside a small radius for loiter_s seconds
            if t - tr.born >= c["loiter_s"]:
                pts = np.array([[p[1], p[2]] for p in tr.trail if p[0] >= t - c["loiter_s"]])
                if len(pts) > 5 and np.max(np.linalg.norm(pts - pts.mean(0), axis=1)) < c["loiter_radius"] * W \
                        and self.cooldown_ok(("LOI", tr.id), t):
                    ev.append(Event(t, "LOITERING", "MEDIUM", f"person stationary > {c['loiter_s']}s", tr.id, tr.cls, tr.box))

        # 5) Group intrusion
        for name, lst in persons_in.items():
            if len(lst) >= c["group_n"] and self.cooldown_ok(("GRP", name), t):
                ev.append(Event(t, "GROUP_INTRUSION", "HIGH", f"{len(lst)} persons inside '{name}'"))
        return ev
