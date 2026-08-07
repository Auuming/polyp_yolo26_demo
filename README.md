# YOLO26 Polyp Detection Demo

A personal learning project exploring the end-to-end process of training, evaluating, and deploying a YOLO-based polyp detection model. The project fine-tunes YOLO26 Nano to locate polyps in endoscopy images and provides both a Streamlit web app and a Windows desktop app for testing the trained models on images, videos, and webcam streams.

## Model variants

| App label | File | Purpose |
| --- | --- | --- |
| V1 – positives only (baseline) | `models/best.pt` | Baseline trained from positive polyp examples |
| V2 – positives + 400 negatives | `models/best_v2.pt` | Adds negative images to help study and reduce false positives |

The included weights are YOLO26 Nano models using an inference image size of 640 pixels. Results depend on the training data, image quality, domain shift, and selected confidence threshold.

## Dataset sources

- Positive polyp examples: [EDF-YOLO for polyp detection](https://github.com/noushin94/EDF-YOLO-for-polyp-detection)
- Negative examples: [PolypGen2021](https://www.kaggle.com/datasets/kokoroou/polypgen2021)

## Project structure

```text
polyp_yolo26_demo/
├── desktop_app/           # Tkinter desktop interface and PyInstaller spec
├── models/                # Trained V1 and V2 model weights
├── notebooks/             # Colab training, evaluation, and inference notebooks
│   └── tests/             # Example image and video inputs
├── shared/                # Model discovery and shared inference functions
├── web_app/               # Streamlit application, requirements, and Dockerfile
├── railway.toml           # Railway deployment configuration
└── README.md
```

## Web app

deployed: https://polypyolo26demo-production.up.railway.app

Run local from the repository root:

```powershell
pip install -r web_app\requirements.txt
streamlit run web_app\web_app.py
```

Select a model, confidence threshold, and input source from the sidebar.

## Desktop app

From the repository root:

```powershell
pip install -r desktop_app\requirements.txt
python desktop_app\desktop_app.py
```

Can build EXE on window:

```powershell
pyinstaller --noconfirm desktop_app\PolypDetector.spec
./dist/PolypDetector.exe
```

## Learn and train with the notebooks

The notebooks are designed for Google Colab:

- `notebooks/Polyp_Yolo26.ipynb` — baseline training workflow
- `notebooks/Polyp_Yolo26_v2.ipynb` — training workflow with added negative examples

Training defaults in the current notebook include 65 epochs, image size 640, batch size 4, and an initial learning rate of 0.01.

## Docker

Build from the repository root:

```powershell
docker build -f web_app/Dockerfile -t polyp-yolo26-demo .
docker run --rm -p 8501:8501 polyp-yolo26-demo
```

Then visit `http://localhost:8501`.

## Deployment
https://polypyolo26demo-production.up.railway.app

`railway.toml` configures Railway to build `web_app/Dockerfile`, check Streamlit's health endpoint, and restart the service on failure.

## Tech Stack

Built with [Ultralytics YOLO](https://docs.ultralytics.com/), [Roboflow](https://roboflow.com/), [Streamlit](https://streamlit.io/), OpenCV, PyTorch, and Tkinter.
