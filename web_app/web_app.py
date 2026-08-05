from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import tempfile

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.detector import default_device, find_model, load_model as create_model, predict

st.set_page_config(page_title="YOLO26 Polyp Detector", layout="wide")

APP_DIR = Path(__file__).resolve().parent
try:
    MODEL_PATH = find_model(APP_DIR, ROOT_DIR)
except FileNotFoundError:
    MODEL_PATH = ROOT_DIR / "models" / "best.pt"
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return create_model(model_path)


def detect(model: YOLO, image: Image.Image, confidence: float):
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    device = default_device()
    result = predict(model, bgr, confidence, device)
    annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    return result, annotated_rgb, device


def detection_table(result) -> pd.DataFrame:
    rows = []
    if result.boxes is None:
        return pd.DataFrame(columns=["Class", "Confidence", "x1", "y1", "x2", "y2"])
    classes = result.boxes.cls.int().cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    boxes = result.boxes.xyxy.cpu().tolist()
    for class_id, score, box in zip(classes, scores, boxes):
        rows.append(
            {
                "Class": result.names[class_id],
                "Confidence": round(score, 4),
                "x1": round(box[0], 1),
                "y1": round(box[1], 1),
                "x2": round(box[2], 1),
                "y2": round(box[3], 1),
            }
        )
    return pd.DataFrame(rows)


def as_jpeg(image_rgb: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(image_rgb).save(output, format="JPEG", quality=92)
    return output.getvalue()


def process_video(model: YOLO, uploaded_file, confidence: float, progress) -> tuple[bytes, int]:
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        output_path = Path(temp_dir) / "polyp_detection.mp4"
        input_path.write_bytes(uploaded_file.getbuffer())

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Could not open the uploaded video.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError("Could not create the processed MP4 video.")

        frames = 0
        device = default_device()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = predict(model, frame, confidence, device)
                writer.write(result.plot())
                frames += 1
                if total:
                    progress.progress(
                        min(frames / total, 1.0),
                        text=f"Processing frame {frames}/{total}",
                    )
        finally:
            capture.release()
            writer.release()

        if frames == 0:
            raise ValueError("The uploaded video contained no readable frames.")
        return output_path.read_bytes(), frames


st.title("YOLO26 Polyp Detector")
st.caption("Detect polyps in an image, an uploaded video, or a live webcam stream.")

if not MODEL_PATH.is_file():
    st.error("best.pt was not found. Place best.pt in the same folder as web_app.py.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load best.pt: {error}")
    st.stop()

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    source_type = st.radio("Input source", ["Upload image", "Upload video", "Webcam"])

source = None
if source_type == "Upload image":
    source = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if source_type == "Webcam":
    st.info("Click START and allow camera access. Click STOP to end live detection.")
    webcam_confidence = float(confidence)
    webcam_device = default_device()

    def process_webcam_frame(frame: av.VideoFrame) -> av.VideoFrame:
        frame_bgr = frame.to_ndarray(format="bgr24")
        result = predict(model, frame_bgr, webcam_confidence, webcam_device)
        return av.VideoFrame.from_ndarray(result.plot(), format="bgr24")

    webrtc_streamer(
        key="polyp-realtime-webcam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=process_webcam_frame,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif source_type == "Upload video":
    video_source = st.file_uploader(
        "Choose a video", type=["mp4", "avi", "mov", "mkv", "webm"]
    )
    if video_source is None:
        st.info("Upload a video to start detection.")
    elif st.button("Process video", type="primary"):
        try:
            progress = st.progress(0.0, text="Preparing video...")
            video_bytes, frame_count = process_video(model, video_source, confidence, progress)
            progress.empty()
            st.success(f"Processed {frame_count} video frames.")
            st.video(video_bytes, format="video/mp4")
            st.download_button(
                "Download annotated video",
                data=video_bytes,
                file_name="polyp_detection_result.mp4",
                mime="video/mp4",
            )
        except Exception as error:
            st.error(f"Video detection failed: {error}")

elif source is not None:
    try:
        original = Image.open(source).convert("RGB")
        with st.spinner("Detecting polyps..."):
            result, annotated, device = detect(model, original, confidence)

        table = detection_table(result)
        left, right = st.columns(2)
        with left:
            st.subheader("Original")
            st.image(original, use_container_width=True)
        with right:
            st.subheader("Detection result")
            st.image(annotated, use_container_width=True)

        count = len(table)
        st.success(f"Detection complete: {count} object{'s' if count != 1 else ''} found on {device}.")
        if count:
            st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.info("No polyp was detected at the selected confidence threshold.")

        st.download_button(
            "Download annotated image",
            data=as_jpeg(annotated),
            file_name="polyp_detection_result.jpg",
            mime="image/jpeg",
        )
    except Exception as error:
        st.error(f"Detection failed: {error}")
elif source_type == "Upload image":
    st.info("Provide an image to start detection.")
