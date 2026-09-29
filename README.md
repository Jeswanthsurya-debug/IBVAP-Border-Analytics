# IBVAP – Intelligent Border Video Analytics Platform (SIH prototype)

Turns any standard CCTV / IP-camera stream into an alerting surveillance feed. **Hybrid design:** a pretrained
detector (YOLOv8n) finds people/vehicles; everything that makes it *security* logic is transparent, tunable,
rule-based code (geometry, timers, speed, background subtraction). Easy to explain, easy to tune per site.

| Requirement | How it is done |
|---|---|
| Human detection & tracking | YOLOv8n + own IoU tracker (`tracker.py`) |
| Vehicle detection & classification | YOLOv8n (car / bus / truck / motorcycle / bicycle) |
| Face detection | OpenCV Haar cascade on the head region of tracked persons |
| ANPR | contour-based plate localisation -> EasyOCR -> Indian plate regex + state-code check + OCR correction |
| Virtual fence intrusion | polygon zones + tripwire lines (`rules.py`) |
| Suspicious activity | loitering, running, group intrusion, watch-listed plate |
| Night-time movement | brightness check -> CLAHE/gamma enhancement + MOG2 motion detection (catches movers the detector misses) |
| Alerts & event log | SQLite + snapshot images + CSV export |
| C2 integration | JSON webhook per alert (`store.py`) |

## Run (Mac)
```bash
brew install python@3.12                # torch wheels are safest on 3.11/3.12
cd ibvap
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python selftest.py                      # logic tests, no camera needed
streamlit run app.py                    # dashboard -> choose Webcam or upload a video
```
First run downloads `yolov8n.pt` (~6 MB) and, on first plate read, the EasyOCR models.

## Demo tips
* Webcam: stand in the red zone -> INTRUSION; cross the yellow line -> TRIPWIRE; stay still 8 s -> LOITERING.
* Night: dim the room / use a dark clip -> "NIGHT" mode; wave something -> NIGHT_MOTION.
* Plates: play a road clip with clear Indian plates, or hold a printed plate to the webcam. Add the number to the watch-list.
* Edit the fence in the sidebar JSON (coordinates 0-1) to match any camera view.
* `--yolo` / `--ocr` flags on `selftest.py` test the real model and OCR.

## Known limits (be upfront with judges)
* Haar face detection is frontal-only; recognition (matching against a watch-list) is the planned next module.
* ANPR needs a reasonably large, sharp plate; night ANPR needs IR/plate-illuminated cameras.
* Speed thresholds are in image space; production would calibrate per camera (homography).
* CPU prototype; scale-out = one pipeline worker per camera + GPU/edge box, central event DB.
