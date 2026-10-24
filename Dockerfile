FROM python:3.11-slim

# ffmpeg decodes iTunes previews for live (out-of-catalog) song lookups
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image ~1.5GB smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bake the embedding model into the image, then run offline so startup never
# waits on (or fails because of) the Hugging Face Hub
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
# CLAP (audio + text in one space) for song <-> film matching and live songs
RUN python -c "from transformers import ClapModel, ClapProcessor; ClapModel.from_pretrained('laion/clap-htsat-unfused'); ClapProcessor.from_pretrained('laion/clap-htsat-unfused')"
ENV HF_HUB_OFFLINE=1

COPY src/ src/
COPY api/ api/
COPY web/ web/
COPY models/ models/

EXPOSE 8000
# hosting platforms (Render, Railway, Fly, Cloud Run) pass the port in $PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
