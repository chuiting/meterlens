# Container for hosting the web app (Google Cloud Run or any Docker host).
# The CNN weights are not in the image: src/fetch_models.py downloads them at
# start-up from the URLs in config.yaml (set HF_TOKEN for a private repo).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# The extra index makes pip pick the CPU-only torch build for the detector
# instead of the default one, which bundles ~2 GB of CUDA libraries.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .
ENV METERLENS_OPEN_BROWSER=0 PORT=8080
CMD python src/fetch_models.py && uvicorn api.main:app --host 0.0.0.0 --port $PORT
