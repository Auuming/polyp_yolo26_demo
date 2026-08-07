FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# OpenCV video I/O and common headless runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY web_app/requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY web_app/web_app.py ./
COPY shared/ ./shared/
COPY models/best.pt models/best_v2.pt ./models/

EXPOSE 8501

CMD ["sh", "-c", "streamlit run web_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
