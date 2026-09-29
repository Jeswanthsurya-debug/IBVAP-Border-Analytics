"""IBVAP dashboard.  Run:  streamlit run app.py"""
import html, json, os, tempfile, time
import cv2, pandas as pd, streamlit as st
from ibvap import Pipeline, EventStore, DEFAULT_CFG

st.set_page_config(page_title="IBVAP", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ theme
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#0f1720; --panel:#16212d; --panel2:#1b2836; --line:#263646;
  --text:#e7edf3; --muted:#8798aa; --ice:#62c6f2;
  --high:#ff6b6b; --med:#f2b04a; --info:#62c6f2; --ok:#4fd1a1;
}
html, body, [class*="css"], .stApp, button, input, textarea { font-family:'Manrope',system-ui,sans-serif !important; }
.stApp { background:var(--bg); color:var(--text); }
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.block-container { padding:1.4rem 2.2rem 2rem; max-width:1500px; }

/* sidebar */
[data-testid="stSidebar"] { background:#0b121a; border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarContent"] { padding-top:1.2rem; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color:var(--muted); font-size:.82rem; font-weight:500; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:8px; background:var(--panel); }
[data-testid="stExpander"] summary { font-weight:600; font-size:.9rem; color:var(--text); }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { background:#0f1720 !important; border:1px solid var(--line) !important; border-radius:6px !important; color:var(--text) !important; }
[data-testid="stSidebar"] textarea { font-size:.75rem; }
.side-brand { font-weight:800; font-size:1.05rem; letter-spacing:.01em; margin-bottom:.9rem; }
.side-brand span { color:var(--muted); font-weight:500; }

/* top bar */
.topbar { display:flex; align-items:center; justify-content:space-between; padding-bottom:1rem; margin-bottom:1.1rem; border-bottom:1px solid var(--line); }
.topbar h1 { font-size:1.35rem; font-weight:800; margin:0; letter-spacing:-.01em; }
.topbar p { margin:.15rem 0 0; color:var(--muted); font-size:.85rem; }
.pill { display:inline-flex; align-items:center; gap:.5rem; padding:.4rem .85rem; border:1px solid var(--line); border-radius:999px; background:var(--panel); font-size:.82rem; font-weight:600; }
.dot { width:8px; height:8px; border-radius:50%; background:var(--muted); }
.live .dot { background:var(--ok); box-shadow:0 0 0 0 rgba(79,209,161,.6); animation:pulse 1.8s infinite; }
@keyframes pulse { 70%{box-shadow:0 0 0 8px rgba(79,209,161,0);} 100%{box-shadow:0 0 0 0 rgba(79,209,161,0);} }
@media (prefers-reduced-motion:reduce){ .live .dot{animation:none;} }

/* toggle */
[data-testid="stToggle"] label p { font-weight:700 !important; color:var(--text) !important; font-size:.95rem !important; }

/* video */
[data-testid="stImage"] img { border-radius:8px; border:1px solid var(--line); }
.idle { aspect-ratio:16/9; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:.4rem; text-align:center;
  border:1px dashed #33485c; border-radius:8px; background:repeating-linear-gradient(135deg,#121c27,#121c27 14px,#141f2b 14px,#141f2b 28px); color:var(--muted); padding:1rem; }
.idle b { color:var(--text); font-size:1.05rem; }
.idle.err b { color:var(--high); }

/* kpis */
.kpis { display:grid; grid-template-columns:repeat(4,1fr); border:1px solid var(--line); border-radius:8px; background:var(--panel); margin-bottom:1.3rem; }
.kpis div { padding:.85rem 1rem; border-right:1px solid var(--line); }
.kpis div:last-child { border-right:0; }
.kpis b { display:block; font-size:1.7rem; font-weight:800; line-height:1.1; font-variant-numeric:tabular-nums; }
.kpis span { color:var(--muted); font-size:.78rem; font-weight:500; }
.kpis .hot b { color:var(--med); }

/* alerts */
.sec { display:flex; align-items:baseline; justify-content:space-between; margin:0 0 .6rem; }
.sec h3 { font-size:1rem; font-weight:700; margin:0; padding:0; }
.sec small { color:var(--muted); font-size:.78rem; }
.feed { border:1px solid var(--line); border-radius:8px; background:var(--panel); overflow:hidden; }
.row { display:grid; grid-template-columns:5px 62px 1fr auto; align-items:center; gap:.8rem; padding:.62rem .9rem .62rem 0; border-bottom:1px solid var(--line); }
.row:last-child { border-bottom:0; }
.row .bar { align-self:stretch; background:var(--info); }
.row.high .bar { background:var(--high); } .row.med .bar { background:var(--med); }
.row .t { color:var(--muted); font-size:.78rem; font-variant-numeric:tabular-nums; margin-left:.2rem; }
.row .m b { display:block; font-size:.88rem; font-weight:700; }
.row .m small { color:var(--muted); font-size:.75rem; }
.row .n { font-size:.75rem; color:var(--muted); border:1px solid var(--line); border-radius:5px; padding:.1rem .45rem; }
.empty { padding:1.4rem 1rem; color:var(--muted); font-size:.85rem; text-align:center; }
.snap-gap { height:1.3rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_store():
    return EventStore()


store = get_store()
geo_default = json.dumps({"zones": DEFAULT_CFG["zones"], "tripwires": DEFAULT_CFG["tripwires"]}, indent=1)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="side-brand">IBVAP <span>settings</span></div>', unsafe_allow_html=True)

    with st.expander("Camera", expanded=True):
        src_type = st.radio("Source", ["Upload video", "Webcam", "RTSP / file path / URL"], key="src_type")
        source, is_file = None, True
        if src_type == "Upload video":
            up = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"], key="upload")
            if up:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1])
                tmp.write(up.read()); tmp.close(); source = tmp.name
        elif src_type == "Webcam":
            source, is_file = 0, False
        else:
            source = st.text_input("Stream address", placeholder="rtsp://user:pass@ip:554/stream or path/to/video.mp4", key="url")
            is_file = not str(source).lower().startswith(("rtsp", "http"))
        camera = st.text_input("Camera name", "BOP-ALPHA-CAM-01", key="cam")

    with st.expander("Detection rules", expanded=True):
        conf = st.slider("Detection confidence", 0.2, 0.8, 0.35, 0.05, key="conf")
        skip = st.slider("Process every Nth frame", 1, 6, 2, key="skip")
        loiter_s = st.slider("Loitering time (s)", 3, 30, 8, key="loiter")
        run_speed = st.slider("Running speed (frame widths/s)", 0.1, 1.0, 0.3, 0.05, key="run")
        group_n = st.slider("Group alert (people in zone)", 2, 8, 3, key="grp")
        night_thr = st.slider("Night mode below brightness", 20, 120, 60, key="night")
        c1, c2, c3 = st.columns(3)
        en_anpr = c1.checkbox("ANPR", True, key="anpr")
        en_face = c2.checkbox("Face", True, key="face")
        en_night = c3.checkbox("Night", True, key="nightmode")

    with st.expander("Zones and integrations", expanded=False):
        watch = st.text_input("Watch-list plates", "TN09AB1234", help="Comma separated", key="watch")
        geo_txt = st.text_area("Virtual fence and tripwires (0-1 coordinates)", geo_default, height=220, key="geo")
        webhook = st.text_input("Command and control webhook URL", "", placeholder="Optional", key="hook")

    st.download_button("Download event log (CSV)", open(store.export_csv(), "rb").read(), "ibvap_events.csv", use_container_width=True)

# ------------------------------------------------------------------ layout
header_ph = st.empty()


def header(live: bool):
    cls, label = ("live", "Monitoring live") if live else ("", "Standby")
    header_ph.markdown(
        f'<div class="topbar"><div><h1>IBVAP</h1>'
        f'<p>Intelligent Border Video Analytics Platform. Turns any CCTV or IP camera into an alerting feed.</p></div>'
        f'<div class="pill {cls}"><span class="dot"></span>{label}<span style="color:var(--muted);font-weight:500">'
        f'{html.escape(camera)}</span></div></div>', unsafe_allow_html=True)


left, right = st.columns([3, 2], gap="large")
with left:
    run = st.toggle("Start monitoring", key="run_toggle")
    video_ph = st.empty()
with right:
    kpi_ph = st.empty()
    alerts_ph = st.empty()
    st.markdown('<div class="snap-gap"></div><div class="sec"><h3>Latest snapshot</h3></div>', unsafe_allow_html=True)
    snap_ph = st.empty()

header(run)


def kpis(persons=0, vehicles=0, fps=0.0):
    n = store.count()
    kpi_ph.markdown(
        f'<div class="kpis"><div><b>{persons}</b><span>People</span></div>'
        f'<div><b>{vehicles}</b><span>Vehicles</span></div>'
        f'<div class="hot"><b>{n}</b><span>Alerts logged</span></div>'
        f'<div><b>{fps:.1f}</b><span>Frames per second</span></div></div>', unsafe_allow_html=True)


def sev_class(s):
    s = str(s).upper()
    return "high" if s in ("HIGH", "CRITICAL") else "med" if s == "MEDIUM" else ""


def render_alerts():
    rows = store.recent(12)
    if not rows:
        alerts_ph.markdown('<div class="sec"><h3>Live alerts</h3></div><div class="feed"><div class="empty">No alerts yet. Events appear here as soon as a rule triggers.</div></div>', unsafe_allow_html=True)
        return
    # collapse consecutive repeats (e.g. continuous loitering) into one row with a count
    groups = []
    for r in rows:
        if groups and groups[-1]["type"] == r[2] and groups[-1]["obj"] == r[3]:
            groups[-1]["n"] += 1
        else:
            groups.append({"time": str(r[0]), "sev": r[1], "type": r[2], "obj": r[3], "detail": r[4], "n": 1})
    items = ""
    for g in groups:
        title = html.escape(str(g["type"]).replace("_", " ").capitalize())
        sub = html.escape(str(g["detail"] or g["obj"]))
        tag = f'x{g["n"]}' if g["n"] > 1 else html.escape(str(g["obj"]))
        items += (f'<div class="row {sev_class(g["sev"])}"><div class="bar"></div><div class="t">{html.escape(g["time"][-8:])}</div>'
                  f'<div class="m"><b>{title}</b><small>{sub}</small></div><div class="n">{tag}</div></div>')
    alerts_ph.markdown(f'<div class="sec"><h3>Live alerts</h3><small>Latest {len(rows)} events</small></div><div class="feed">{items}</div>', unsafe_allow_html=True)
    snaps = [r[5] for r in rows if r[5] and os.path.exists(r[5])]
    if snaps:
        snap_ph.image(snaps[0], use_container_width=True)


def notice(title, body, err=False):
    video_ph.markdown(f'<div class="idle {"err" if err else ""}"><b>{title}</b><span>{body}</span></div>', unsafe_allow_html=True)


kpis()
render_alerts()

if not run:
    notice("Ready when you are", "Choose a camera in the sidebar, then switch on Start monitoring. A webcam is the quickest way to try the virtual fence.")
    st.stop()
if source in (None, ""):
    notice("No camera selected", "Upload a video, pick the webcam, or enter a stream address in the sidebar.", err=True); st.stop()

try:
    geo = json.loads(geo_txt)
    cfg = {"conf": conf, "loiter_s": loiter_s, "run_speed": run_speed, "group_n": group_n,
           "night_brightness": night_thr, "enable_anpr": en_anpr, "enable_face": en_face, "enable_night": en_night,
           "watchlist_plates": [p.strip().upper() for p in watch.split(",") if p.strip()],
           "zones": geo["zones"], "tripwires": geo["tripwires"]}
    pipe = Pipeline(cfg)
except ImportError as e:
    notice("Missing dependency", f"{html.escape(str(e))}. Run: pip install -r requirements.txt", err=True); st.stop()
except Exception as e:
    notice("Could not start", html.escape(str(e)), err=True); st.stop()

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    notice("Could not open the camera", "Check the address or permissions and try again.", err=True); st.stop()
src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
last, fps, frames = time.time(), 0.0, 0

while True:
    for _ in range(skip - 1):
        cap.grab()
    ok, frame = cap.read()
    if not ok:
        notice("Stream ended", "The video finished or the camera disconnected."); break
    if frame.shape[1] > 960:
        frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
    t = cap.get(cv2.CAP_PROP_POS_FRAMES) / src_fps if is_file else time.time()
    out, events = pipe.process(frame, t)
    for e in events:
        store.add(e, out, camera=camera, webhook=webhook)
    frames += 1
    fps = 0.9 * fps + 0.1 / max(time.time() - last, 1e-3); last = time.time()
    video_ph.image(out[:, :, ::-1], channels="RGB", use_container_width=True)
    if frames % 3 == 0 or events:
        kpis(pipe.stats["person"], pipe.stats["vehicle"], fps)
    if events or frames % 20 == 0:
        render_alerts()
cap.release()
