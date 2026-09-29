# Cephalometric Landmark API
# Recommended weights: saved_models/best_hrnet_w48.pth  (MRE 1.10 mm)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHECKPOINT_PATH=saved_models/best_hrnet_w48.pth \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY config.py predict.py api_server.py evaluate.py train.py ./
COPY configs ./configs
COPY models ./models
COPY data_loader ./data_loader
COPY utils ./utils
COPY saved_models/config_used.json saved_models/best_hrnet_w48.pth ./saved_models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "api_server.py"]
