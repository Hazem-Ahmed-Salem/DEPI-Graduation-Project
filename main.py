# main.py
import os
import pickle
import time
from typing import List, Tuple
import cv2 as cv
import numpy as np
import streamlit as st
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from monitoring_module import check_model_performance

# ---------------- Config ----------------
MODEL_PICKLE = "hand_gesture_norm_model.pkl" # Model pickle (must contain model and le)
SCALER_PICKLE = "scaler.pkl"
CAMERA_ID = 0

QUEUE_LENGTH = 5
CONF_THRESH = 0.60 # threshold to show label (0..1)

# Preprocessing
ORIGIN_IDX = 0
SCALE_METHOD = "max_dist"
APPLY_ROTATION = True
MIRROR_LEFT = False # set True only if you mirrored left-hand training data

# UI appearance
CARD_BG = (15, 20, 25, 200)
TEXT_COLOR = (255, 255, 255)
SECONDARY_TEXT = (200, 200, 200)
TOP3_COLOR = (255, 210, 0)

# ---------------- Cached resource loader ----------------
@st.cache_resource
def load_resources(model_path: str = MODEL_PICKLE, scaler_path: str = SCALER_PICKLE):
    """
    Load model, label-encoder, and scaler (if present).
    - Accepts common pickle shapes: (model, le) tuple OR {'model':..., 'le':...}
    - Requires that `le` is present in the pickle. If not, raises FileNotFoundError with instructions.
    This function is cached and will run only once per Streamlit process.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Place it in the working directory.")

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    model = None
    le = None

    # Flexible detection, but require `le` be present in the same pickle
    if isinstance(obj, tuple) and len(obj) >= 2:
        model, le = obj[0], obj[1]
    elif isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        # try to find label encoder key
        if "le" in obj:
            le = obj["le"]
        elif "label_encoder" in obj:
            le = obj["label_encoder"]
    else:
        model = obj

    if le is None:
        raise FileNotFoundError(
            "Label encoder not found inside the model pickle. Please pack the label encoder with the model "
            "(e.g. `pickle.dump((model, le), f)` or `pickle.dump({'model':model,'le':le}, f)`)."
        )

    scaler = None
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            scaler = None

    return model, le, scaler

# ---------------- Helpers ----------------
def normalize_landmarks_flat(pts_flat, origin_idx=ORIGIN_IDX, scale_method=SCALE_METHOD, rotate=APPLY_ROTATION, eps=1e-8):
    pts = np.asarray(pts_flat, dtype=np.float32).reshape(-1)
    F = pts.size
    if F == 63:
        C = 3
    elif F == 42:
        C = 2
    else:
        return pts  # unknown format

    pts = pts.reshape(21, C)
    origin = pts[origin_idx].copy()
    pts = pts - origin

    if scale_method == "max_dist":
        d = np.linalg.norm(pts, axis=1)
        s = d.max()
    elif scale_method == "wrist_to_middle":
        s = np.linalg.norm(pts[9])
    elif scale_method == "bbox":
        s = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    else:
        s = 1.0
    s = max(s, eps)
    pts = pts / s

    if rotate and C >= 2:
        v = pts[9, :2]
        angle = np.arctan2(v[0], v[1])
        c = np.cos(-angle); sn = np.sin(-angle)
        R = np.array([[c, -sn], [sn, c]], dtype=np.float32)
        pts[:, :2] = pts[:, :2] @ R.T

    return pts.reshape(-1)

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

def smooth_prediction(pred_vector, hand_id, history, queue_len=QUEUE_LENGTH):
    if hand_id not in history:
        history[hand_id] = []
    history[hand_id].append(pred_vector)
    if len(history[hand_id]) > queue_len:
        history[hand_id].pop(0)
    return np.mean(history[hand_id], axis=0)

def draw_label_card(frame_rgb: np.ndarray, title: str, subtitle: str, top3: List[Tuple[str, float]], position: str = "right") -> np.ndarray:
    pil = Image.fromarray(frame_rgb.astype("uint8"))
    draw = ImageDraw.Draw(pil, "RGBA")

    W, H = pil.size
    card_w, card_h = int(W * 0.33), int(120 + 22 * max(len(top3), 1))
    padding = 12
    margin = 20

    if position == "right":
        x0, y0 = W - card_w - margin, margin
    else:
        x0, y0 = margin, margin
    x1, y1 = x0 + card_w, y0 + card_h

    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=12, fill=CARD_BG)

    # fonts
    try:
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        font_regular = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font_bold = ImageFont.load_default()
        font_regular = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((x0 + padding, y0 + padding), title, font=font_bold, fill=TEXT_COLOR)
    draw.text((x0 + padding, y0 + 28), subtitle, font=font_regular, fill=SECONDARY_TEXT)

    y = y0 + 55
    for label, conf in top3:
        draw.ellipse([(x0 + padding, y + 6), (x0 + padding + 10, y + 16)], fill=TOP3_COLOR)
        draw.text((x0 + padding + 16, y), f"{label} - {conf:.1f}%", font=font_small, fill=TEXT_COLOR)
        y += 22

    return np.array(pil)

# ---------------- UI ----------------
st.set_page_config(page_title="Hand Gesture (cached loader)", layout="wide")
st.markdown(
    """
    <style>
    .title {
        font-size:40px;
        font-weight:700;
        background: linear-gradient(90deg,#1DA1F2,#34D399);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        margin-bottom:6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div style="text-align:center"><div class="title">Hand Gesture Recognition</div></div>', unsafe_allow_html=True)

col_vid, col_info = st.columns([2.5, 1])
with col_vid:
    video_placeholder = st.empty()
    b1, b2, b3 = st.columns([1,1,1])
    with b1:
        start = st.button("Start Camera")
    with b2:
        stop = st.button("Stop Camera")
    with b3:
        clear_history_btn = st.button("Clear History")

with col_info:
    st.markdown("### Predictions")
    right_slot = st.empty()
    right_top3 = st.empty()
    left_slot = st.empty()
    left_top3 = st.empty()

# ---------------- Load resources (cached) ----------------
try:
    model, le, scaler = load_resources(MODEL_PICKLE, SCALER_PICKLE)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Mediapipe init
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.6, min_tracking_confidence=0.5)

history = {}

# camera state
if "running" not in st.session_state:
    st.session_state.running = False
if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False
if clear_history_btn:
    history.clear()
    st.success("Cleared history")

cap = None
if st.session_state.running:
    cap = cv.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        st.error("Cannot open camera.")
        st.session_state.running = False
        cap = None

# helper to update info column
def update_info(model_name, right_info, left_info):
    if right_info:
        lbl, conf, top3 = right_info
        right_slot.markdown(f"**Right:** {lbl} ({conf:.1f}%)")
        right_top3.markdown("\n".join([f"- **{a}** — {b:.1f}%" for a,b in top3]))
    else:
        right_slot.markdown("**Right:** —")
        right_top3.markdown("")
    if left_info:
        lbl, conf, top3 = left_info
        left_slot.markdown(f"**Left:** {lbl} ({conf:.1f}%)")
        left_top3.markdown("\n".join([f"- **{a}** — {b:.1f}%" for a,b in top3]))
    else:
        left_slot.markdown("**Left:** —")
        left_top3.markdown("")

# ---------------- Main loop ----------------
try:
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera read failed.")
            break

        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        right_display = None
        left_display = None

        if results.multi_hand_landmarks:
            for i, (lm_list, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                side = handedness.classification[0].label # 'Left' or 'Right'
                hand_id = f"{side}_{i}"

                # flatten landmarks x,y,z
                pts = []
                for lm in lm_list.landmark:
                    pts.extend([lm.x, lm.y, lm.z])
                pts = np.array(pts, dtype=np.float32)

                # normalize
                pts_norm = normalize_landmarks_flat(pts)

                # optional mirroring if used in training
                if MIRROR_LEFT and side.lower() == "left":
                    C = 3 if pts_norm.size == 63 else 2
                    tmp = pts_norm.reshape(21, C)
                    tmp[:, 0] *= -1.0
                    pts_norm = tmp.reshape(-1)

                # apply scaler if available
                inp = pts_norm.reshape(1, -1)
                if scaler is not None:
                    try:
                        inp = scaler.transform(inp)
                    except Exception:
                        pass

                # predict
                try:
                    # Start Time
                    t_start = time.time()
                    raw = model.predict(inp, verbose=0)
                    # End Time with ms
                    inference_time_ms = (time.time() - t_start) * 1000
                except Exception as e:
                    st.warning(f"Prediction error: {e}")
                    inference_time_ms = 0.0
                    continue

                pv = raw[0]
                if not np.isclose(pv.sum(), 1.0, atol=1e-3):
                    pv = softmax(pv)

                smooth = smooth_prediction(pv, hand_id, history)
                idx = int(np.argmax(smooth))
                conf_ratio = float(smooth[idx])
                conf = conf_ratio * 100.0

                # decode label using the label encoder that came from the model pickle
                try:
                    label_name = le.inverse_transform([idx])[0]
                except Exception:
                    label_name = str(idx)

                check_model_performance(
                    conf_ratio,           # Confidence Value
                    label_name,           # Gesture Name
                    side,                 # Which hand (right/left)
                    MODEL_PICKLE,         # Model Name
                    inference_time_ms     # Prediction Time
                )

                # top-3
                sorted_idx = np.argsort(smooth)[-3:][::-1]
                top3 = []
                for si in sorted_idx:
                    try:
                        nm = le.inverse_transform([si])[0]
                    except Exception:
                        nm = str(si)
                    top3.append((nm, float(smooth[si] * 100.0)))

                if side.lower() == "right" and right_display is None:
                    right_display = (f"{side}: {label_name}", conf, top3)
                if side.lower() == "left" and left_display is None:
                    left_display = (f"{side}: {label_name}", conf, top3)

        # draw cards
        frame_out = frame_rgb.copy()
        if right_display:
            t, c, top3 = right_display
            frame_out = draw_label_card(frame_out, t, f"{c:.1f}%", top3, position="right")
        if left_display:
            t, c, top3 = left_display
            frame_out = draw_label_card(frame_out, t, f"{c:.1f}%", top3, position="left")

        # update info column
        update_info(MODEL_PICKLE, right_display, left_display)

        video_placeholder.image(frame_out, channels="RGB")
        time.sleep(0.01)

        if stop:
            st.session_state.running = False
            break

finally:
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    cv.destroyAllWindows()