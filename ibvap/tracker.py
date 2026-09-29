"""Lightweight IoU multi-object tracker (no ML, no extra deps)."""
from collections import deque
from dataclasses import dataclass, field

GROUP = {"person": "p", "bicycle": "2w", "motorcycle": "2w", "car": "v", "bus": "v", "truck": "v"}


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    id: int
    cls: str
    box: tuple
    conf: float
    born: float
    hits: int = 1
    missed: int = 0
    trail: deque = field(default_factory=lambda: deque(maxlen=900))  # (t, x, y) of ground point
    meta: dict = field(default_factory=dict)

    @property
    def anchor(self):  # ground contact point = bottom-centre of box
        return ((self.box[0] + self.box[2]) / 2, self.box[3])


class IoUTracker:
    def __init__(self, iou_thr=0.25, max_missed=20):
        self.iou_thr, self.max_missed = iou_thr, max_missed
        self.tracks, self._next = [], 1

    def update(self, dets, t):
        """dets: list of (box, cls, conf). Returns tracks matched in this frame."""
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, (box, cls, _) in enumerate(dets):
                if GROUP[cls] != GROUP[tr.cls]:
                    continue
                v = iou(tr.box, box)
                if v >= self.iou_thr:
                    pairs.append((v, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for v, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            tr = self.tracks[ti]
            tr.box, tr.cls, tr.conf = dets[di][0], dets[di][1], dets[di][2]
            tr.hits += 1; tr.missed = 0
            tr.trail.append((t, *tr.anchor))
        # fallback for fast movers / low FPS: match leftovers by centre distance
        for ti, tr in enumerate(self.tracks):
            if ti in used_t:
                continue
            tw = max(tr.box[2] - tr.box[0], tr.box[3] - tr.box[1])
            tc = ((tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2)
            best = None
            for di, (box, cls, _) in enumerate(dets):
                if di in used_d or GROUP[cls] != GROUP[tr.cls]:
                    continue
                d = ((tc[0] - (box[0] + box[2]) / 2) ** 2 + (tc[1] - (box[1] + box[3]) / 2) ** 2) ** 0.5
                if d < 0.9 * tw and (best is None or d < best[0]):
                    best = (d, di)
            if best:
                di = best[1]; used_t.add(ti); used_d.add(di)
                tr.box, tr.cls, tr.conf = dets[di][0], dets[di][1], dets[di][2]
                tr.hits += 1; tr.missed = 0
                tr.trail.append((t, *tr.anchor))
        for di, (box, cls, conf) in enumerate(dets):
            if di in used_d:
                continue
            tr = Track(self._next, cls, box, conf, t)
            tr.trail.append((t, *tr.anchor))
            self._next += 1
            self.tracks.append(tr); used_t.add(len(self.tracks) - 1)
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.missed += 1
        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]
        return [tr for tr in self.tracks if tr.missed == 0]
