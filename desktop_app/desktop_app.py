from __future__ import annotations

import sys
import threading
import tkinter as tk
import shutil
import tempfile
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.detector import default_device, find_model, load_model, predict


APP_TITLE = "YOLO26 Polyp Detector"
IMAGE_TYPES = [("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
VIDEO_TYPES = [("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")]


class PolypDetectorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1100x760")
        self.root.minsize(820, 620)

        self.device = default_device()
        self.model: YOLO | None = None
        self.source_bgr = None
        self.result_bgr = None
        self.photo = None
        self.capture: cv2.VideoCapture | None = None
        self.camera_running = False
        self.camera_busy = False
        self.video_running = False
        self.video_writer: cv2.VideoWriter | None = None
        self.video_result_path: Path | None = None
        self.video_frame_number = 0
        self.video_frame_total = 0

        self.confidence = tk.DoubleVar(value=0.25)
        self.status = tk.StringVar(value="Loading best.pt...")
        self.summary = tk.StringVar(value="No image selected")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(fill=tk.X)

        self.open_button = ttk.Button(toolbar, text="Open image", command=self.open_image, state=tk.DISABLED)
        self.open_button.pack(side=tk.LEFT, padx=(0, 6))
        self.video_button = ttk.Button(toolbar, text="Open video", command=self.open_video, state=tk.DISABLED)
        self.video_button.pack(side=tk.LEFT, padx=6)
        self.detect_button = ttk.Button(toolbar, text="Detect", command=self.detect_image, state=tk.DISABLED)
        self.detect_button.pack(side=tk.LEFT, padx=6)
        self.camera_button = ttk.Button(toolbar, text="Start webcam", command=self.toggle_camera, state=tk.DISABLED)
        self.camera_button.pack(side=tk.LEFT, padx=6)
        self.save_button = ttk.Button(toolbar, text="Save result", command=self.save_result, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=6)

        ttk.Label(toolbar, text="Confidence").pack(side=tk.LEFT, padx=(22, 4))
        ttk.Scale(toolbar, from_=0.05, to=0.95, variable=self.confidence, length=180).pack(side=tk.LEFT)
        ttk.Label(toolbar, textvariable=self.confidence, width=5).pack(side=tk.LEFT, padx=4)

        self.image_label = ttk.Label(self.root, anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        footer = ttk.Frame(self.root, padding=10)
        footer.pack(fill=tk.X)
        ttk.Label(footer, textvariable=self.summary, font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(footer, textvariable=self.status).pack(anchor=tk.W, pady=(4, 0))

    def _load_model(self) -> None:
        try:
            search_dirs = [ROOT_DIR, Path.cwd(), Path(sys.executable).resolve().parent]
            bundle_dir = getattr(sys, "_MEIPASS", None)
            if bundle_dir:
                search_dirs.insert(0, Path(bundle_dir))
            model_path = find_model(*search_dirs)
            self.model = load_model(model_path)
            self.root.after(0, lambda: self._model_ready(model_path))
        except Exception as error:
            self.root.after(0, lambda: self._show_error("Model loading failed", error))

    def _model_ready(self, model_path: Path) -> None:
        self.open_button.config(state=tk.NORMAL)
        self.video_button.config(state=tk.NORMAL)
        self.camera_button.config(state=tk.NORMAL)
        self.status.set(f"Ready | Model: {model_path.name} | Device: {self.device}")

    def open_image(self) -> None:
        self.stop_capture()
        self.video_result_path = None
        path = filedialog.askopenfilename(title="Select an image", filetypes=IMAGE_TYPES)
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror(APP_TITLE, "The selected image could not be opened.")
            return
        self.source_bgr = frame
        self.result_bgr = None
        self.detect_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        self.summary.set(Path(path).name)
        self.status.set("Image loaded. Select Detect.")
        self._show_frame(frame)

    def open_video(self) -> None:
        self.stop_capture()
        path = filedialog.askopenfilename(title="Select a video", filetypes=VIDEO_TYPES)
        if not path:
            return
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            messagebox.showerror(APP_TITLE, "The selected video could not be opened.")
            return

        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result_path = Path(tempfile.gettempdir()) / "polyp_detection_result.mp4"
        writer = cv2.VideoWriter(
            str(result_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            messagebox.showerror(APP_TITLE, "Could not create the processed MP4 video.")
            return

        self.capture = capture
        self.video_writer = writer
        self.video_result_path = result_path
        self.video_frame_number = 0
        self.video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_running = True
        self.result_bgr = None
        self._set_controls(False)
        self.summary.set(Path(path).name)
        self.status.set("Video detection running...")
        self._next_capture_frame()

    def detect_image(self) -> None:
        if self.source_bgr is None or self.model is None:
            return
        self._set_controls(False)
        self.status.set("Running detection...")
        frame = self.source_bgr.copy()
        threading.Thread(target=self._infer_still, args=(frame,), daemon=True).start()

    def _infer_still(self, frame) -> None:
        try:
            result = self._predict(frame)
            annotated = result.plot()
            self.root.after(0, lambda: self._finish_still(result, annotated))
        except Exception as error:
            self.root.after(0, lambda: self._show_error("Detection failed", error))

    def _finish_still(self, result, annotated) -> None:
        self.result_bgr = annotated
        self._show_frame(annotated)
        self._show_summary(result)
        self.status.set("Detection complete.")
        self._set_controls(True)
        self.save_button.config(state=tk.NORMAL)

    def _predict(self, frame):
        return predict(self.model, frame, float(self.confidence.get()), self.device)

    def toggle_camera(self) -> None:
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self) -> None:
        self.stop_capture()
        self.video_result_path = None
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            messagebox.showerror(APP_TITLE, "Could not open webcam 0.")
            return
        self.camera_running = True
        self.camera_button.config(text="Stop webcam")
        self.open_button.config(state=tk.DISABLED)
        self.video_button.config(state=tk.DISABLED)
        self.detect_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.status.set("Live webcam detection running.")
        self._next_capture_frame()

    def _next_capture_frame(self) -> None:
        if not (self.camera_running or self.video_running) or self.capture is None:
            return
        if self.video_running and self.camera_busy:
            return
        ok, frame = self.capture.read()
        if not ok:
            if self.video_running:
                self._finish_video()
            else:
                self.stop_camera()
                messagebox.showerror(APP_TITLE, "Could not read a webcam frame.")
            return
        if not self.camera_busy:
            self.camera_busy = True
            threading.Thread(target=self._infer_camera, args=(frame,), daemon=True).start()
        if self.camera_running:
            self.root.after(15, self._next_capture_frame)

    def _infer_camera(self, frame) -> None:
        try:
            result = self._predict(frame)
            annotated = result.plot()
            self.root.after(0, lambda: self._finish_camera(result, annotated))
        except Exception as error:
            self.root.after(0, lambda: self._show_error("Webcam detection failed", error))

    def _finish_camera(self, result, annotated) -> None:
        self.camera_busy = False
        if not (self.camera_running or self.video_running):
            return
        self.result_bgr = annotated
        if self.video_running and self.video_writer is not None:
            self.video_writer.write(annotated)
            self.video_frame_number += 1
            total = f"/{self.video_frame_total}" if self.video_frame_total else ""
            self.status.set(f"Processing video frame {self.video_frame_number}{total}...")
        self._show_frame(annotated)
        self._show_summary(result)
        if self.video_running:
            self.root.after(1, self._next_capture_frame)

    def _finish_video(self) -> None:
        self.video_running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.camera_busy = False
        self._set_controls(True)
        self.save_button.config(state=tk.NORMAL if self.video_result_path else tk.DISABLED)
        self.status.set(f"Video complete: {self.video_frame_number} frames. Select Save result.")

    def stop_capture(self) -> None:
        was_video = self.video_running
        self.video_running = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.stop_camera()
        if was_video:
            self.status.set("Video processing stopped.")

    def stop_camera(self) -> None:
        self.camera_running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if hasattr(self, "camera_button"):
            self.camera_button.config(text="Start webcam")
            if self.model is not None:
                self.open_button.config(state=tk.NORMAL)
                self.video_button.config(state=tk.NORMAL)
                self.camera_button.config(state=tk.NORMAL)
                self.save_button.config(state=tk.NORMAL if self.result_bgr is not None else tk.DISABLED)
        self.status.set("Webcam stopped.")

    def _show_summary(self, result) -> None:
        counts: dict[str, int] = {}
        if result.boxes is not None:
            for class_id in result.boxes.cls.int().cpu().tolist():
                name = result.names[class_id]
                counts[name] = counts.get(name, 0) + 1
        total = sum(counts.values())
        details = ", ".join(f"{name}: {count}" for name, count in counts.items()) or "none"
        self.summary.set(f"Detections: {total} ({details})")

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_width = max(self.image_label.winfo_width() - 20, 640)
        max_height = max(self.image_label.winfo_height() - 20, 480)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)

    def save_result(self) -> None:
        if self.video_result_path is not None and self.video_result_path.is_file():
            path = filedialog.asksaveasfilename(
                title="Save annotated video", defaultextension=".mp4", filetypes=VIDEO_TYPES
            )
            if path:
                try:
                    shutil.copyfile(self.video_result_path, path)
                    self.status.set(f"Saved: {path}")
                except OSError as error:
                    messagebox.showerror(APP_TITLE, f"The video could not be saved: {error}")
            return
        if self.result_bgr is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save annotated result",
            defaultextension=".jpg",
            filetypes=IMAGE_TYPES,
        )
        if path:
            if not cv2.imwrite(path, self.result_bgr):
                messagebox.showerror(APP_TITLE, "The result could not be saved.")
            else:
                self.status.set(f"Saved: {path}")

    def _set_controls(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.open_button.config(state=state)
        self.video_button.config(state=state)
        self.camera_button.config(state=state)
        self.detect_button.config(state=state if self.source_bgr is not None else tk.DISABLED)

    def _show_error(self, title: str, error: Exception) -> None:
        self.camera_busy = False
        self.stop_camera()
        self._set_controls(self.model is not None)
        self.status.set(str(error))
        messagebox.showerror(title, str(error))

    def close(self) -> None:
        self.stop_capture()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PolypDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
