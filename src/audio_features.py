"""
Extracts our own audio features (tempo, energy, brightness, etc.) from
real preview clips, replacing Spotify's now-deprecated audio-features
endpoint. Requires ffmpeg on your system PATH.
"""

import os
import subprocess
import requests
import librosa
import numpy as np

TEMP_DIR = "data/raw/previews"
os.makedirs(TEMP_DIR, exist_ok=True)


def download_preview(preview_url, track_id):
    """Save a 30s preview clip. Apple previews are AAC/.m4a, not .mp3 --
    matching the real extension matters for decoding later."""
    ext = ".m4a" if preview_url.lower().endswith(".m4a") else ".mp3"
    raw_path = os.path.join(TEMP_DIR, f"{track_id}{ext}")
    if not os.path.exists(raw_path):
        resp = requests.get(preview_url, timeout=15)
        resp.raise_for_status()
        with open(raw_path, "wb") as f:
            f.write(resp.content)
    return raw_path


def convert_to_wav(raw_path, track_id):
    """Convert to mono 22050Hz WAV via ffmpeg directly -- more reliable
    on Windows than librosa's built-in AAC decoding fallback."""
    wav_path = os.path.join(TEMP_DIR, f"{track_id}.wav")
    if os.path.exists(wav_path):
        return wav_path
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path, "-ar", "22050", "-ac", "1", wav_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-500:]}")
    return wav_path


def extract_features(file_path):
    """Our version of Spotify's old audio-features: tempo (BPM), energy
    (RMS loudness), brightness (spectral centroid), zcr (noisiness proxy),
    and a 12-dim chroma vector (rough key/pitch profile)."""
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    return {
        "tempo": float(tempo) if np.isscalar(tempo) else float(tempo[0]),
        "energy": float(np.mean(rms)),
        "brightness": float(np.mean(spectral_centroid)),
        "zcr": float(np.mean(zcr)),
        "chroma_mean": chroma.mean(axis=1).tolist(),
    }


def get_features_for_track(track):
    """Full pipeline: download -> convert -> extract. Returns None on
    failure so callers can skip a bad track without crashing a batch run."""
    try:
        raw_path = download_preview(track["preview_url"], track["track_id"])
        wav_path = convert_to_wav(raw_path, track["track_id"])
        return {**track, **extract_features(wav_path)}
    except Exception as e:
        print(f"Skipping {track.get('track_name', '?')}: {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":
    from itunes_client import search_tracks

    results = search_tracks("Blinding Lights", limit=1)
    if results:
        enriched = get_features_for_track(results[0])
        if enriched:
            print("Connected OK. Extracted features:")
            for k, v in enriched.items():
                if k != "chroma_mean":
                    print(f"  {k}: {v}")