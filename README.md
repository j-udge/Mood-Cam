# Emotion Tracker

A real-time facial emotion detection app built with Streamlit, OpenCV, and DeepFace. It reads your webcam feed, detects your face, classifies your emotion, and displays a matching emoji alongside a live confidence bar chart.

---

## Features

- Live webcam feed with face bounding box overlay
- Real-time emotion classification via DeepFace
- Smoothed predictions across a rolling history window to reduce flicker
- Side panel with per-emotion confidence bars and a matching emoji
- Threaded inference — UI never blocks waiting for the model
- Dark, minimal UI styled with custom CSS

---

## Project Structure

```
Face_to_emojii/
├── src/
│   ├── main.py          # App entry point — camera loop, UI, inference
│   ├── ui_styles.py     # CSS, HTML snippets for Streamlit
│   └── emojis/          # Emoji images named by emotion label
│       ├── happy.jpeg
│       ├── sad.jpeg
│       ├── angry.jpeg
│       ├── neutral.jpeg
│       ├── surprise.jpeg
│       ├── fear.jpeg
│       └── disgust.jpeg
```

---

## Requirements

- Python 3.9+
- Webcam

Install dependencies:

```bash
pip install streamlit deepface opencv-python-headless numpy tf-keras
```

> DeepFace will auto-download the model weights on first run.

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/yourname/face_to_emojii.git
cd face_to_emojii
```

**2. Add emoji images**

Place `.jpeg` files in `src/emojis/`, named exactly after DeepFace emotion labels:

```
angry.jpeg  disgust.jpeg  fear.jpeg  happy.jpeg
neutral.jpeg  sad.jpeg  surprise.jpeg
```

**3. Update the emoji path**

In `main.py`, set `EMOJI_DIR` to your local path:

```python
EMOJI_DIR = r"C:/Face_to_emojii/src/emojis"
```

**4. Run the app**

```bash
streamlit run src/main.py
```

---

## Configuration

All tunable constants are at the top of `main.py`:

| Constant | Default | Description |
|---|---|---|
| `SMOOTH` | `6` | Rolling window size for prediction smoothing |
| `MIN_CONF` | `40` | Minimum confidence % to accept a detection |
| `SKIP` | `4` | Run inference every N frames |
| `DETECTOR` | `'opencv'` | DeepFace detector backend |

---

## How It Works

1. Each frame is captured from the webcam via OpenCV
2. Every 4th frame is sent to a background thread for DeepFace inference
3. The frame is preprocessed with CLAHE (contrast enhancement) before analysis
4. Detected emotion and region are written to a shared state under a lock
5. The main loop reads that state, draws the face box, builds the side panel, and streams the combined image to Streamlit
6. Predictions are smoothed over the last 6 frames to reduce jitter

---

## Troubleshooting

**Emoji not showing** — Check that `EMOJI_DIR` points to the correct folder and that files are named exactly as listed above (lowercase, `.jpeg` extension).

**No face detected** — Ensure lighting is adequate. Try lowering `MIN_CONF` if detections are being filtered out.

**Slow FPS** — Increase `SKIP` to run inference less frequently, or switch `DETECTOR` to `'ssd'` for faster (less accurate) detection.

**Camera not opening** — Make sure no other app is using the webcam. The app uses `cv2.CAP_DSHOW` (Windows DirectShow); remove that flag on Mac/Linux.

---

## License

MIT

