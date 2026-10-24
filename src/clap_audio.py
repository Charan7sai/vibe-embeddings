"""
CLAP (contrastive language-audio pretraining): audio and text in one
embedding space, so a song's preview can be compared directly with a
description like "dark, tense music for a horror film".

Uses LAION's clap-htsat-unfused checkpoint. (larger_clap_music loads without
errors but its audio and text towers come out unaligned -- every audio-text
cosine ~0 -- so it is not used.) This replaces hand-written rules
(minor key + low brightness -> "dark") with a model that learned what
music described in words actually sounds like.
"""

import numpy as np

MODEL_NAME = "laion/clap-htsat-unfused"
SR = 48000          # CLAP expects 48 kHz
WINDOW_S = 10       # and 10-second clips

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import ClapModel, ClapProcessor
        _model = ClapModel.from_pretrained(MODEL_NAME).eval()
        _processor = ClapProcessor.from_pretrained(MODEL_NAME)
        torch.set_grad_enabled(False)
    return _model, _processor


def _features(out):
    # transformers 5 returns an output object here; 4.x returned the tensor
    tensor = out if hasattr(out, "detach") else out.pooler_output
    return tensor.detach().numpy()


def _unit(x):
    return (x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)).astype(np.float32)


def _window(path):
    """The middle 10 seconds of a preview: past any intro, before the fade."""
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    n = WINDOW_S * SR
    if len(y) > n:
        start = (len(y) - n) // 2
        y = y[start:start + n]
    return y


def embed_audio(paths, batch_size=8):
    """(n, 512) unit vectors for audio files."""
    model, processor = _load()
    out = []
    for i in range(0, len(paths), batch_size):
        clips = [_window(p) for p in paths[i:i + batch_size]]
        inputs = processor(audio=clips, sampling_rate=SR, return_tensors="pt")
        out.append(_features(model.get_audio_features(**inputs)))
    return _unit(np.vstack(out))


def embed_text(texts, batch_size=64):
    """(n, 512) unit vectors for text, in the same space as embed_audio."""
    model, processor = _load()
    out = []
    for i in range(0, len(texts), batch_size):
        inputs = processor(text=list(texts[i:i + batch_size]), padding=True, return_tensors="pt")
        out.append(_features(model.get_text_features(**inputs)))
    return _unit(np.vstack(out))
