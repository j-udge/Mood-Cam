import streamlit as st, cv2, numpy as np, os, threading
from deepface import DeepFace
from collections import deque
from ui_styles import CSS, CAM_LABEL, STATUS_ON, STATUS_OFF, IDLE_CARD

st.set_page_config(page_title="Emotion Tracker", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

EMOJI_DIR, SMOOTH, MIN_CONF, SKIP, DETECTOR = r"C:/Face_to_emojii/src/emojis", 6, 40, 4, 'opencv'
TC, TXT, DIM = (74,158,255), (240,244,255), (60,120,140)
_lock, _latest, _pending = threading.Lock(), {"emo":"neutral","scores":{},"region":None}, {"frame":None,"running":False}

ctrl_col, _, feed_col = st.columns([1, 0.08, 3])
with ctrl_col:
    st.markdown(CAM_LABEL, unsafe_allow_html=True)
    run_camera = st.toggle("Enable Live Feed")
    st.markdown(STATUS_ON if run_camera else STATUS_OFF, unsafe_allow_html=True)
with feed_col:
    ui_placeholder = st.empty()
    if not run_camera: ui_placeholder.markdown(IDLE_CARD, unsafe_allow_html=True)

def preprocess(f):
    l,a,b = cv2.split(cv2.cvtColor(f, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge([cv2.createCLAHE(3.5,(6,6)).apply(l),a,b]), cv2.COLOR_LAB2BGR)

def smooth(hist, scores):
    hist.append(scores); t = {}
    for s in hist:
        for e,v in s.items(): t[e] = t.get(e,0)+v
    return max(t, key=t.get)

def draw_face(frame, x, y, w, h, emo):
    for cx,cy,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame,(cx,cy),(cx+dx*14,cy),TC,2); cv2.line(frame,(cx,cy),(cx,cy+dy*14),TC,2)
    (tw,th),_ = cv2.getTextSize(emo.upper(), cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    cv2.rectangle(frame,(x,y-th-14),(x+tw+16,y-2),TC,-1)
    cv2.putText(frame, emo.upper(), (x+8,y-6), cv2.FONT_HERSHEY_DUPLEX, 0.65, (8,12,16), 1)

def build_panel(frame, emo, scores):
    p = np.full_like(frame,(18,14,12))
    cv2.line(p,(0,0),(p.shape[1],0),TC,2); cv2.line(p,(0,0),(0,p.shape[0]),TC,1)
    cv2.putText(p,"DETECTED",(30,50),cv2.FONT_HERSHEY_DUPLEX,0.5,DIM,1)
    cv2.putText(p,emo.capitalize(),(30,90),cv2.FONT_HERSHEY_DUPLEX,1.1,TXT,1)
    by = 130
    for e,v in sorted(scores.items(), key=lambda x:-x[1]):
        col = TC if e==emo else (50,60,70)
        cv2.rectangle(p,(30,by),(30+int(v*1.8),by+12),col,-1)
        cv2.putText(p,f"{e[:3].upper()} {int(v)}%",(30,by-4),cv2.FONT_HERSHEY_DUPLEX,0.38,TXT if e==emo else DIM,1)
        by += 28
    path = next((os.path.join(EMOJI_DIR,f"{emo.lower()}{ext}") for ext in ('.jpeg','.jpg','.png') if os.path.exists(os.path.join(EMOJI_DIR,f"{emo.lower()}{ext}"))),None)
    if path:
        raw = cv2.imread(path); h,w = raw.shape[:2]; s = min(h,w); raw = raw[(h-s)//2:(h+s)//2,(w-s)//2:(w+s)//2]
        img = cv2.resize(raw,(160,160)); hp,wp = p.shape[:2]; yo,xo = max(0,hp-180),(wp-160)//2
        mask = np.zeros((160,160),dtype=np.uint8); cv2.rectangle(mask,(0,10),(160,150),255,-1); cv2.rectangle(mask,(10,0),(150,160),255,-1)
        for pt in [(10,10),(150,10),(10,150),(150,150)]: cv2.circle(mask,pt,10,255,-1)
        p[yo:yo+160,xo:xo+160] = cv2.bitwise_and(img,img,mask=mask)
    return p

def _worker():
    with _lock: frame = _pending["frame"]
    h,w = frame.shape[:2]; scale = min(1.0, 320/w); small = cv2.resize(frame,(0,0),fx=scale,fy=scale) if scale<1 else frame
    try:
        f0 = DeepFace.analyze(preprocess(small),actions=['emotion'],detector_backend=DETECTOR,enforce_detection=True,silent=True)[0]
        if f0['emotion'][f0['dominant_emotion']] >= MIN_CONF:
            r,inv = f0['region'], 1/scale
            with _lock: _latest.update(emo=f0['dominant_emotion'], scores=f0['emotion'],
                                       region=(int(r['x']*inv),int(r['y']*inv),int(r['w']*inv),int(r['h']*inv)))
    except: pass
    with _lock: _pending["running"] = False

def submit(frame):
    with _lock:
        if _pending["running"]: return
        _pending.update(frame=frame.copy(), running=True)
    threading.Thread(target=_worker, daemon=True).start()

if run_camera:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    for prop,val in [(cv2.CAP_PROP_FRAME_WIDTH,640),(cv2.CAP_PROP_FRAME_HEIGHT,480),(cv2.CAP_PROP_FPS,30),(cv2.CAP_PROP_BUFFERSIZE,1)]: cap.set(prop,val)
    hist = deque(maxlen=SMOOTH); idx = 0
    while run_camera:
        ret, frame = cap.read()
        if not ret: break
        if idx % SKIP == 0: submit(frame)
        idx += 1
        with _lock: emo,scores,region = _latest["emo"],dict(_latest["scores"]),_latest["region"]
        if region: draw_face(frame,*region,emo)
        panel = build_panel(frame,emo,scores) if scores else np.full_like(frame,(18,14,12))
        if not scores: cv2.putText(panel,"No face",(30,90),cv2.FONT_HERSHEY_DUPLEX,0.8,(60,70,80),1)
        if scores: smooth(hist, scores)
        _, buf = cv2.imencode('.jpg', cv2.hconcat([frame,panel]), [cv2.IMWRITE_JPEG_QUALITY,85])
        ui_placeholder.image(buf.tobytes(), use_container_width=True)
    cap.release()