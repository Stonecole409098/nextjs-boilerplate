# TruthLens — Streamlit multimodal forensic workbench
import io
import importlib
import os
import json
import re
import struct
import tempfile
import uuid
import hashlib
import wave
from math import gcd
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageChops, ExifTags


def read_wav(file_obj):
    """Read PCM WAV data without requiring SciPy."""
    with wave.open(file_obj, "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8)
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2")
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        data = (packed[:, 0].astype(np.int32)
                | (packed[:, 1].astype(np.int32) << 8)
                | (packed[:, 2].astype(np.int32) << 16))
        data = np.where(data & 0x800000, data - 0x1000000, data)
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4")
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels)
    return sample_rate, data.copy()


def resample_audio(data: np.ndarray, target_rate: int, source_rate: int) -> np.ndarray:
    """Resample audio using NumPy interpolation."""
    if source_rate == target_rate or len(data) < 2:
        return data.astype(np.float32, copy=False)
    target_length = max(1, round(len(data) * target_rate / source_rate))
    source_positions = np.arange(len(data), dtype=np.float64)
    target_positions = np.linspace(0, len(data) - 1, target_length)
    return np.interp(target_positions, source_positions, data).astype(np.float32)


def power_spectrum(data: np.ndarray, sample_rate: int, segment_length: int = 2048):
    """Return a Welch-style averaged one-sided power spectrum."""
    segment_length = min(segment_length, len(data))
    if segment_length <= 0:
        return np.array([]), np.array([])
    step = max(1, segment_length // 2)
    window = np.hanning(segment_length)
    scale = sample_rate * max(float(np.sum(window ** 2)), 1e-12)
    spectra = []
    for start in range(0, max(1, len(data) - segment_length + 1), step):
        segment = data[start:start + segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        spectra.append(np.abs(np.fft.rfft(segment * window)) ** 2 / scale)
    return np.fft.rfftfreq(segment_length, 1.0 / sample_rate), np.mean(spectra, axis=0)

st.set_page_config(
    page_title="TruthLens | Media verification",
    page_icon=":material/verified_user:",
    layout="wide",
)

AI_IMAGE_SIGNATURES = [
    (b"midjourney", "Midjourney"),
    (b"stable diffusion", "Stable Diffusion"),
    (b"dall-e", "DALL-E"),
    (b"c2pa", "C2PA AI provenance"),
]
AI_VIDEO_SIGNATURES = [
    (b"runway", "Runway"),
    (b"sora", "OpenAI Sora"),
    (b"pika", "Pika"),
    (b"veo", "Google Veo"),
    (b"kling", "Kling AI"),
    (b"luma", "Luma Dream Machine"),
    (b"heygen", "HeyGen"),
    (b"synthesia", "Synthesia"),
    (b"c2pa", "C2PA Content Credentials"),
]
AI_VIDEO_FRAME_MODEL = "dima806/deepfake_vs_real_image_detection"
AI_VIDEO_FRAME_SAMPLE_COUNT = 12
AI_AUDIO_SIGNATURES = [
    (b"elevenlabs", "ElevenLabs"),
    (b"descript", "Descript"),
]
AI_AUDIO_MODEL = "MelodyMachine/Deepfake-audio-detection-V2"
AI_AUDIO_MODEL_SAMPLE_RATE = 16000
AI_AUDIO_SEGMENT_SECONDS = 5
AI_AUDIO_MAX_SEGMENTS = 8
EDITOR_SIGNATURES = [
    (b"adobe photoshop", "Adobe Photoshop"),
    (b"gimp", "GIMP"),
]
URGENCY_CUES = ["breaking", "share before deleted", "leaked", "urgent", "must see"]
LLM_PHRASES = [
    "it is important to note",
    "in conclusion",
    "pivotal role",
    "underscores the importance",
]
DATE_PATTERN = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{4}[-/]\d{2}[-/]\d{2}\b|\bNovember\s+2026\b", re.I)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_header_text(raw_bytes: bytes) -> bytes:
    head = raw_bytes[:128 * 1024]
    tail = raw_bytes[-32 * 1024:] if len(raw_bytes) > 128 * 1024 else b""
    return (head + b"\n" + tail).lower()


@st.cache_resource(show_spinner=False)
def load_neural_pipeline():
    try:
        pipeline = importlib.import_module("transformers").pipeline
    except ModuleNotFoundError as exc:
        if exc.name == "transformers":
            raise RuntimeError("The optional 'transformers' package is required for neural frame detection.") from exc
        raise
    return pipeline("image-classification", model=AI_VIDEO_FRAME_MODEL)


def run_neural_frame_detection(frame: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    try:
        classifier = load_neural_pipeline()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = classifier(Image.fromarray(rgb_frame))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    fake_score = max((p["score"] for p in predictions if "fake" in p["label"].lower()), default=0.0)
    real_score = max((p["score"] for p in predictions if "real" in p["label"].lower()), default=0.0)
    total = fake_score + real_score
    if total <= 0:
        return None, "Model output did not include recognizable fake/real labels."
    return fake_score / total, None


@st.cache_resource(show_spinner=False)
def load_ai_audio_pipeline():
    try:
        pipeline = importlib.import_module("transformers").pipeline
    except ModuleNotFoundError as exc:
        if exc.name == "transformers":
            raise RuntimeError("The optional 'transformers' package is required for AI audio detection.") from exc
        raise
    return pipeline("audio-classification", model=AI_AUDIO_MODEL)


def run_ai_voice_segment(
    audio_segment: np.ndarray, sample_rate: int
) -> Tuple[Optional[float], Optional[str]]:
    try:
        classifier = load_ai_audio_pipeline()
        if sample_rate != AI_AUDIO_MODEL_SAMPLE_RATE:
            divisor = gcd(sample_rate, AI_AUDIO_MODEL_SAMPLE_RATE)
            def resample_poly(audio_segment, arg1, arg2):
                ...

            audio_segment = resample_poly(
                audio_segment,
                AI_AUDIO_MODEL_SAMPLE_RATE // divisor,
                sample_rate // divisor,
            ).astype(np.float32)
        predictions = classifier(
            {"array": audio_segment, "sampling_rate": AI_AUDIO_MODEL_SAMPLE_RATE},
            top_k=None,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if isinstance(predictions, dict):
        predictions = [predictions]
    if predictions and isinstance(predictions[0], list):
        predictions = predictions[0]
    fake_scores = [
        float(item["score"])
        for item in predictions
        if "fake" in item["label"].lower()
    ]
    if not fake_scores:
        return None, "Model output did not include a recognizable fake label."
    return max(fake_scores), None


def analyze_image(raw_bytes: bytes, evidence_id: str, filename: str, sha256_hash: str):
    findings: List[Dict[str, Any]] = []
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except Exception:
        return {"error": "Unable to decode image"}, findings, None, None

    orig_w, orig_h = img.size
    header_text = extract_header_text(raw_bytes)
    exif = img.getexif()
    exif_date = exif.get(36867) or exif.get(306)
    has_exif = bool(exif_date)

    detected_ai = [name for key, name in AI_IMAGE_SIGNATURES if key in header_text]
    detected_editors = [name for key, name in EDITOR_SIGNATURES if key in header_text]

    if detected_ai:
        findings.append({
            "evidence_id": evidence_id,
            "filename": filename,
            "media_type": "image",
            "sha256_hash": sha256_hash,
            "location": "File Header / Provenance Metadata",
            "indicator_category": "AI Platform Signature",
            "signal_type": "fake",
            "observed_signal": f"Embedded AI generator trace found: {', '.join(detected_ai)}.",
            "explanation": "The file header contains metadata or provenance tags associated with an AI image generator.",
            "raw_score": 0.90,
            "confidence_interval": "90% ± 6% (High)",
            "severity": "HIGH",
            "alternative_explanations": ["An authentic photo edited with an AI-assisted tool can inherit AI provenance tags."],
            "limitations": ["Metadata can be stripped when an image is re-saved or shared."],
        })

    if detected_editors:
        findings.append({
            "evidence_id": evidence_id,
            "filename": filename,
            "media_type": "image",
            "sha256_hash": sha256_hash,
            "location": "File Header / Software Tag",
            "indicator_category": "Editing Software Trace",
            "signal_type": "morphed",
            "observed_signal": f"Image header references editing software: {', '.join(detected_editors)}.",
            "explanation": "The image was exported from an editing program rather than directly from a camera sensor.",
            "raw_score": 0.62,
            "confidence_interval": "62% ± 15% (Moderate)",
            "severity": "MEDIUM",
            "alternative_explanations": ["Routine cropping, resizing, or color correction."],
            "limitations": ["Software tags confirm post-processing, not malicious manipulation."],
        })
    elif not has_exif:
        findings.append({
            "evidence_id": evidence_id,
            "filename": filename,
            "media_type": "image",
            "sha256_hash": sha256_hash,
            "location": "File Header",
            "indicator_category": "Stripped Camera Metadata (Info)",
            "signal_type": "info",
            "observed_signal": "No camera EXIF metadata found in the file header.",
            "explanation": "Camera details are absent, which is common for social-media exports, screenshots, and AI images.",
            "raw_score": 0.10,
            "confidence_interval": "10% ± 10% (Informational)",
            "severity": "LOW",
            "alternative_explanations": ["Messaging apps and social platforms commonly strip EXIF data."],
            "limitations": ["Missing metadata alone is not proof of editing or AI generation."],
        })

    rgb_img = img.convert("RGB")
    max_dim = 512
    scale = min(1.0, max_dim / max(orig_w, orig_h))
    w = max(32, int(orig_w * scale))
    h = max(32, int(orig_h * scale))
    resized = rgb_img.resize((w, h), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    recomp = Image.open(buf).convert("RGB")

    orig_arr = np.asarray(resized, dtype=np.float32)
    recomp_arr = np.asarray(recomp, dtype=np.float32)
    diff_map = np.mean(np.abs(orig_arr - recomp_arr), axis=2)

    gray = 0.299 * orig_arr[:, :, 0] + 0.587 * orig_arr[:, :, 1] + 0.114 * orig_arr[:, :, 2]
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edge_strength = gx + gy
    edge_weight = 1.0 / (1.0 + edge_strength / 18.0)
    compensated_ela = diff_map * edge_weight

    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
    smooth_mask = edge_strength < 20.0

    ela_vis = np.clip(diff_map * 12.0, 0, 255).astype(np.uint8)
    heatmap_rgb = np.stack([
        ela_vis,
        np.clip(ela_vis * 0.45, 0, 255).astype(np.uint8),
        (255 - ela_vis).astype(np.uint8),
    ], axis=2)
    ela_heatmap_img = Image.fromarray(heatmap_rgb)

    grid = 8
    gh = max(1, h // grid)
    gw = max(1, w // grid)
    block_ela = []
    block_noise = []

    for r in range(grid):
        for c in range(grid):
            patch_ela = compensated_ela[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
            if patch_ela.size:
                block_ela.append((float(np.mean(patch_ela)), r, c))

            patch_mask = smooth_mask[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
            if np.sum(patch_mask) > 30:
                patch_lap = lap[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
                block_noise.append(float(np.mean(patch_lap[patch_mask])))

    ela_vals = np.array([b[0] for b in block_ela])
    median_ela = float(np.median(ela_vals))
    mad_ela = float(np.median(np.abs(ela_vals - median_ela)))
    peak_val, peak_r, peak_c = max(block_ela, key=lambda x: x[0])
    robust_z = (peak_val - median_ela) / (1.4826 * max(mad_ela, 0.25))

    if robust_z > 3.2 and peak_val > 3.0:
        provenance_discount = 0.12 if exif_date else 0.0
        score = max(0.40, min(0.84, round(0.48 + (robust_z - 3.2) * 0.08 - provenance_discount, 2)))
        findings.append({
            "evidence_id": evidence_id,
            "filename": filename,
            "media_type": "image",
            "sha256_hash": sha256_hash,
            "location": f"Region [Row {peak_r + 1}/8, Col {peak_c + 1}/8]",
            "indicator_category": "Localized Compression Splice (Edge-Compensated ELA)",
            "signal_type": "morphed",
            "observed_signal": f"Non-edge compression anomaly (Robust Z={robust_z:.2f}, patch={peak_val:.2f} vs median={median_ela:.2f}).",
            "explanation": "A localized interior region shows compression error significantly different from the rest of the image.",
            "raw_score": score,
            "confidence_interval": f"{int(score*100)}% ± 14% ({'High' if score >= 0.7 else 'Moderate'})",
            "severity": "HIGH" if score >= 0.7 else "MEDIUM",
            "alternative_explanations": ["Text overlays, watermarks, specular reflections, or re-saving at another quality."],
            "limitations": ["ELA is less reliable for screenshots and heavily downsampled images."],
        })

    if len(block_noise) >= 12:
        med_n = float(np.median(block_noise))
        ratio = max(block_noise) / max(min(block_noise), 0.3)
        if ratio > 4.8 and med_n > 1.2:
            score = min(0.74, round(0.46 + (ratio - 4.8) * 0.05, 2))
            findings.append({
                "evidence_id": evidence_id,
                "filename": filename,
                "media_type": "image",
                "sha256_hash": sha256_hash,
                "location": "Smooth-Region Noise Floor",
                "indicator_category": "Sensor Grain / Noise Mismatch",
                "signal_type": "morphed",
                "observed_signal": f"Smooth-area noise ratio is {ratio:.1f}x across the frame.",
                "explanation": "A sharp mismatch can indicate an inpainted patch or pasted object.",
                "raw_score": score,
                "confidence_interval": f"{int(score*100)}% ± 16% (Moderate)",
                "severity": "MEDIUM",
                "alternative_explanations": ["Portrait bokeh, low-light denoising, or out-of-focus backgrounds."],
                "limitations": ["Depth-of-field naturally creates different noise regions."],
            })

    metadata = {
        "width": orig_w,
        "height": orig_h,
        "exif_date": exif_date,
        "camera_provenance_detected": bool(exif_date),
        "ai_platform_traces": detected_ai,
    }
    return metadata, findings, resized, ela_heatmap_img


def analyze_video(raw_bytes: bytes, evidence_id: str, filename: str, sha256_hash: str):
    findings: List[Dict[str, Any]] = []
    header_text = extract_header_text(raw_bytes)
    detected_ai = [name for key, name in AI_VIDEO_SIGNATURES if key in header_text]

    if detected_ai:
        findings.append({
            "evidence_id": evidence_id, "filename": filename, "media_type": "video", "sha256_hash": sha256_hash,
            "location": "Video Container Metadata", "indicator_category": "AI Video Platform Signature",
            "signal_type": "info", "observed_signal": f"AI video generator trace detected: {', '.join(detected_ai)}.",
            "explanation": "Container metadata references a known generative video platform; this is provenance evidence, not a probability that the video is AI-generated.",
            "raw_score": 0.0, "confidence_interval": "Provenance trace; no probability calculated", "severity": "INFO",
            "alternative_explanations": ["Authentic footage edited or exported through an AI suite."],
            "limitations": ["Re-encoding often strips container metadata."],
        })

    suffix = os.path.splitext(filename)[1].lower() or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {"error": "Unable to decode video", "ai_platform_traces": detected_ai}, findings

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0
        ai_sample_count = AI_VIDEO_FRAME_SAMPLE_COUNT
        sample_indices = set(
            np.linspace(0, frame_count - 1, min(ai_sample_count, frame_count), dtype=int).tolist()
        ) if frame_count > 0 else set()
        frame_ai_scores = []
        frame_ai_error = None
        expected_ai_samples = min(ai_sample_count, frame_count) if frame_count > 0 else None
        metadata = {"fps": round(fps, 2), "frame_count": frame_count,
                    "resolution": f"{width}x{height}", "duration_sec": duration_sec,
                    "ai_platform_traces": detected_ai}

        step = max(1, frame_count // 100) if frame_count > 0 else 1
        prev_hist = None
        sharpness_series = []
        hist_diffs = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            should_score_ai = idx in sample_indices if frame_count > 0 else (
                len(frame_ai_scores) < ai_sample_count and idx % step == 0
            )
            if should_score_ai and frame_ai_error is None:
                score, frame_ai_error = run_neural_frame_detection(frame)
                if score is not None:
                    frame_ai_scores.append(score)
            if idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness_series.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
                cv2.normalize(hist, hist)
                if prev_hist is not None:
                    diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                    hist_diffs.append((idx, diff))
                prev_hist = hist
            idx += 1
        cap.release()

        metadata["experimental_frame_model"] = {
            "model": AI_VIDEO_FRAME_MODEL,
            "score_percent": round(float(np.median(frame_ai_scores)) * 100, 2) if frame_ai_scores else None,
            "sampled_frame_count": len(frame_ai_scores),
            "status": "available" if frame_ai_scores and frame_ai_error is None and (
                expected_ai_samples is None or len(frame_ai_scores) == expected_ai_samples
            ) else (
                "partial" if frame_ai_scores else "unavailable"
            ),
            "error": frame_ai_error,
            "notice": "Experimental image-model score over sampled frames; not calibrated for AI-generated video and not a probability.",
        }

        if hist_diffs:
            diffs_only = np.array([d[1] for d in hist_diffs])
            mean_d = float(diffs_only.mean())
            std_d = float(diffs_only.std()) + 1e-5
            anomalies = [(f, v) for f, v in hist_diffs
                         if (v - mean_d) / std_d > 2.8 and v > 0.28]
            if anomalies:
                top_f, top_v = max(anomalies, key=lambda x: x[1])
                ts = round(top_f / fps, 2)
                score = min(0.80, round(0.50 + top_v * 0.35, 2))
                findings.append({
                    "evidence_id": evidence_id, "filename": filename, "media_type": "video", "sha256_hash": sha256_hash,
                    "location": f"Frame {top_f} ({ts:.2f}s)", "indicator_category": "Temporal Discontinuity",
                    "signal_type": "morphed", "observed_signal": f"Abrupt histogram shift (Bhattacharyya={top_v:.2f}) at {ts:.2f}s.",
                    "explanation": "Sudden frame-to-frame luminance shifts can indicate splicing or temporal flicker.",
                    "raw_score": score, "confidence_interval": f"{int(score*100)}% ± 18% (Moderate)", "severity": "MEDIUM",
                    "alternative_explanations": ["Scene cuts, camera flash, or rapid panning."],
                    "limitations": ["The exact timestamp requires human review."],
                })

        if len(sharpness_series) > 10:
            s_arr = np.array(sharpness_series)
            cv_sharp = float(s_arr.std() / (s_arr.mean() + 1e-5))
            if cv_sharp > 0.70:
                score = min(0.75, round(0.45 + (cv_sharp - 0.70) * 0.35, 2))
                findings.append({
                    "evidence_id": evidence_id, "filename": filename, "media_type": "video", "sha256_hash": sha256_hash,
                    "location": f"Timeline (0.00s - {duration_sec:.2f}s)", "indicator_category": "Temporal Blur / Focus Instability",
                    "signal_type": "morphed", "observed_signal": f"Sharpness coefficient of variation = {cv_sharp:.2f}.",
                    "explanation": "Unstable frame-by-frame sharpness can indicate synthesized or face-swapped regions.",
                    "raw_score": score, "confidence_interval": f"{int(score*100)}% ± 20% (Low-Moderate)", "severity": "MEDIUM",
                    "alternative_explanations": ["Autofocus hunting, handheld shake, or low-light blur."],
                    "limitations": ["Global sharpness variance does not isolate facial regions."],
                })
        return metadata, findings
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def analyze_audio(raw_bytes: bytes, evidence_id: str, filename: str, sha256_hash: str):
    findings: List[Dict[str, Any]] = []
    header_text = extract_header_text(raw_bytes)
    detected_ai = [name for key, name in AI_AUDIO_SIGNATURES if key in header_text]

    if detected_ai:
        findings.append({
            "evidence_id": evidence_id, "filename": filename, "media_type": "audio", "sha256_hash": sha256_hash,
            "location": "Audio Container Metadata", "indicator_category": "AI Voice / Audio Platform Signature",
            "signal_type": "info", "observed_signal": f"Embedded AI audio tag found: {', '.join(detected_ai)}.",
            "explanation": "Container metadata references an AI audio platform; this is provenance evidence, not proof that the voice is synthetic.",
            "raw_score": 0.0, "confidence_interval": "Provenance trace; no probability calculated", "severity": "INFO",
            "alternative_explanations": ["Authentic voice cleaned using an AI mastering tool."],
            "limitations": ["Re-encoding often removes audio metadata."],
        })

    try:
        sr, data = read_wav(io.BytesIO(raw_bytes))
    except Exception as exc:
        return {
            "note": "Audio decoding failed; upload a PCM WAV file.",
            "ai_platform_traces": detected_ai,
            "experimental_voice_model": {
                "model": AI_AUDIO_MODEL,
                "score_percent": None,
                "sampled_segment_count": 0,
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "notice": "Experimental audio deepfake-model score; not calibrated for synthetic voice and not a probability.",
            },
        }, findings

    if sr <= 0 or data.size == 0:
        return {
            "note": "Audio has no decodable samples.",
            "ai_platform_traces": detected_ai,
            "experimental_voice_model": {
                "model": AI_AUDIO_MODEL,
                "score_percent": None,
                "sampled_segment_count": 0,
                "status": "unavailable",
                "error": "Audio sample rate or sample data is empty.",
                "notice": "Experimental audio deepfake-model score; not calibrated for synthetic voice and not a probability.",
            },
        }, findings
    unsigned_pcm_midpoint = None
    if np.issubdtype(data.dtype, np.unsignedinteger):
        unsigned_pcm_midpoint = (np.iinfo(data.dtype).max + 1) / 2
    data = data.astype(np.float32)
    if unsigned_pcm_midpoint is not None:
        data -= unsigned_pcm_midpoint
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data / (np.max(np.abs(data)) + 1e-6)
    duration_sec = round(len(data) / sr, 2)
    metadata = {"sample_rate_hz": int(sr), "duration_sec": duration_sec, "ai_platform_traces": detected_ai}

    segment_samples = max(1, int(sr * AI_AUDIO_SEGMENT_SECONDS))
    segment_count = min(
        AI_AUDIO_MAX_SEGMENTS,
        max(1, int(np.ceil(len(data) / segment_samples))),
    )
    max_start = max(0, len(data) - segment_samples)
    segment_starts = np.linspace(0, max_start, segment_count, dtype=int)
    voice_scores = []
    voice_model_error = None
    if duration_sec < 1.0:
        voice_model_error = "Audio must be at least one second long."
    else:
        for start in segment_starts:
            segment = data[start:start + segment_samples]
            score, voice_model_error = run_ai_voice_segment(segment, int(sr))
            if score is not None:
                voice_scores.append(score)
            if voice_model_error:
                break

    metadata["experimental_voice_model"] = {
        "model": AI_AUDIO_MODEL,
        "score_percent": round(float(np.median(voice_scores)) * 100, 2) if voice_scores else None,
        "sampled_segment_count": len(voice_scores),
        "status": "available" if voice_scores and voice_model_error is None and (
            len(voice_scores) == segment_count
        ) else ("partial" if voice_scores else "unavailable"),
        "error": voice_model_error,
        "notice": "Experimental audio deepfake-model score over sampled segments; not calibrated for synthetic voice and not a probability.",
    }

    def welch(data, fs, nperseg):
        ...

    freqs, psd = welch(data, fs=sr, nperseg=min(2048, len(data)))
    total_energy = float(np.sum(psd)) + 1e-12
    high_mask = freqs > 7000
    high_ratio = float(np.sum(psd[high_mask]) / total_energy) if np.any(high_mask) else 0.0

    if sr >= 16000 and high_ratio < 0.002:
        findings.append({
            "evidence_id": evidence_id, "filename": filename, "media_type": "audio", "sha256_hash": sha256_hash,
            "location": f"Spectral Band >7kHz (0.00s - {duration_sec:.2f}s)",
            "indicator_category": "High-Frequency Spectral Roll-Off", "signal_type": "fake",
            "observed_signal": f"Energy above 7kHz is only {high_ratio*100:.4f}% at {sr}Hz sample rate.",
            "explanation": "Some neural vocoders exhibit sharp high-frequency cutoffs.",
            "raw_score": 0.64, "confidence_interval": "64% ± 17% (Moderate)", "severity": "MEDIUM",
            "alternative_explanations": ["Phone calls, Bluetooth headsets, or aggressive noise suppression."],
            "limitations": ["Modern neural vocoders may not show this cutoff."],
        })

    frame_len = max(1, int(sr * 0.1))
    num_frames = len(data) // frame_len
    rms_vals = [float(np.sqrt(np.mean(data[i * frame_len:(i + 1) * frame_len] ** 2)))
                for i in range(num_frames)]
    near_zero = [i for i, rms in enumerate(rms_vals) if rms < 1e-4]

    if num_frames > 10 and len(near_zero) / num_frames > 0.12:
        start_s = round(near_zero[0] * 0.1, 2)
        end_s = round(near_zero[-1] * 0.1, 2)
        findings.append({
            "evidence_id": evidence_id, "filename": filename, "media_type": "audio", "sha256_hash": sha256_hash,
            "location": f"Audio Segments ({start_s:.2f}s - {end_s:.2f}s)",
            "indicator_category": "Synthetic Digital Silence / Spliced Pauses", "signal_type": "morphed",
            "observed_signal": f"{len(near_zero)} of {num_frames} windows exhibit near-zero digital silence.",
            "explanation": "Pure digital zeros between words can indicate TTS generation or spliced audio.",
            "raw_score": 0.68, "confidence_interval": "68% ± 15% (Moderate)", "severity": "MEDIUM",
            "alternative_explanations": ["Studio noise gates or conferencing applications."],
            "limitations": ["Cannot distinguish synthetic silence from aggressive post-processing."],
        })

    return metadata, findings


def analyze_text(content: str, evidence_id: str, filename: str, sha256_hash: str):
    findings: List[Dict[str, Any]] = []
    words = content.split()
    lower_text = content.lower()
    extracted_dates = list(set(DATE_PATTERN.findall(content)))
    triggered_cues = [cue for cue in URGENCY_CUES if cue in lower_text]
    llm_hits = [phrase for phrase in LLM_PHRASES if phrase in lower_text]

    sentences = [s.strip() for s in re.split(r"[.!?]+", content) if len(s.strip().split()) > 2]
    burstiness = None
    if len(sentences) >= 3:
        lengths = np.array([len(s.split()) for s in sentences], dtype=np.float32)
        burstiness = float(lengths.std() / (lengths.mean() + 1e-5))

    metadata = {"word_count": len(words), "extracted_dates": extracted_dates,
                "llm_phrase_hits": llm_hits,
                "sentence_burstiness": round(burstiness, 2) if burstiness is not None else None}

    if len(llm_hits) >= 2 or (len(llm_hits) >= 1 and burstiness is not None and burstiness < 0.25):
        score = min(0.78, round(0.55 + 0.10 * len(llm_hits), 2))
        findings.append({
            "evidence_id": evidence_id, "filename": filename, "media_type": "text", "sha256_hash": sha256_hash,
            "location": f"Text Body ({len(words)} words)", "indicator_category": "AI / LLM Writing Pattern",
            "signal_type": "fake", "observed_signal": f"Matched LLM phrasing markers ({', '.join(llm_hits)}).",
            "explanation": "Text contains stylistic patterns frequently associated with LLM-generated writing.",
            "raw_score": score, "confidence_interval": f"{int(score*100)}% ± 18% (Moderate)", "severity": "MEDIUM",
            "alternative_explanations": ["Formal press releases, PR copy, or grammar-assisted writing."],
            "limitations": ["Text-only AI detection has high variance on short passages."],
        })

    if triggered_cues:
        findings.append({
            "evidence_id": evidence_id, "filename": filename, "media_type": "text", "sha256_hash": sha256_hash,
            "location": f"Text Body ({len(words)} words)", "indicator_category": "Sensational / Disinformation Framing",
            "signal_type": "morphed", "observed_signal": f"High-urgency framing detected: {', '.join(triggered_cues)}.",
            "explanation": "Urgent captions can be used to bypass critical verification.",
            "raw_score": 0.50, "confidence_interval": "50% ± 20% (Moderate-Low)", "severity": "LOW",
            "alternative_explanations": ["Authentic breaking news or emotional eyewitness posts."],
            "limitations": ["Linguistic framing is contextual, not proof of fabrication."],
        })

    return metadata, findings


def correlate_and_build_report(case_title, investigator, evidence_records, all_findings, audit_trail):
    correlations = []
    images = [e for e in evidence_records if e["media_type"] == "image"]
    texts = [e for e in evidence_records if e["media_type"] == "text"]
    videos = [e for e in evidence_records if e["media_type"] == "video"]
    audios = [e for e in evidence_records if e["media_type"] == "audio"]

    for img in images:
        exif_date = img["metadata"].get("exif_date")
        if exif_date:
            exif_year = str(exif_date)[:4]
            for txt in texts:
                claimed = txt["metadata"].get("extracted_dates", [])
                if claimed and not any(exif_year in str(d) for d in claimed):
                    correlations.append({
                        "correlation_id": f"XMOD-{uuid.uuid4().hex[:5].upper()}",
                        "involved_evidence_ids": [img["evidence_id"], txt["evidence_id"]],
                        "relationship_type": "CONTRADICTION",
                        "description": f"Image '{img['filename']}' header date ({exif_date}) conflicts with text dates ({', '.join(map(str, claimed))}).",
                        "impact_on_confidence": "Strong indicator of out-of-context reuse or temporal misattribution (+10% risk).",
                    })

    for vid in videos:
        v_dur = vid["metadata"].get("duration_sec", 0)
        for aud in audios:
            a_dur = aud["metadata"].get("duration_sec", 0)
            if v_dur > 0 and a_dur > 0 and abs(v_dur - a_dur) > 1.5:
                correlations.append({
                    "correlation_id": f"XMOD-{uuid.uuid4().hex[:5].upper()}",
                    "involved_evidence_ids": [vid["evidence_id"], aud["evidence_id"]],
                    "relationship_type": "CONTRADICTION",
                    "description": f"Video duration ({v_dur}s) differs from audio duration ({a_dur}s) by {abs(v_dur-a_dur):.2f}s.",
                    "impact_on_confidence": "Suggests asynchronous dubbing or replacement (+10% risk).",
                })

    active_findings = [f for f in all_findings if f.get("signal_type") != "info"]
    flagged_modalities = list({f["media_type"] for f in active_findings if f["raw_score"] >= 0.55})

    if len(flagged_modalities) >= 2:
        correlations.append({
            "correlation_id": f"XMOD-{uuid.uuid4().hex[:5].upper()}",
            "involved_evidence_ids": list({f["evidence_id"] for f in active_findings if f["raw_score"] >= 0.55}),
            "relationship_type": "REINFORCEMENT",
            "description": f"Independent anomalies detected across {len(flagged_modalities)} modalities ({', '.join(flagged_modalities)}).",
            "impact_on_confidence": "Converging signals reduce single-sensor false positives.",
        })

    if not correlations and len(evidence_records) >= 2:
        correlations.append({
            "correlation_id": f"XMOD-{uuid.uuid4().hex[:5].upper()}",
            "involved_evidence_ids": [e["evidence_id"] for e in evidence_records],
            "relationship_type": "ALIGNMENT",
            "description": "No direct cross-modal contradictions detected.",
            "impact_on_confidence": "Neutral — absence of contradiction does not guarantee authenticity.",
        })

    contradiction_boost = 10 * sum(1 for c in correlations if c["relationship_type"] == "CONTRADICTION")
    morphed_scores = [f["raw_score"] for f in active_findings if f.get("signal_type") == "morphed"]
    fake_scores = [f["raw_score"] for f in active_findings if f.get("signal_type") == "fake"]

    morphed_pct = min(94, int(round(max(morphed_scores) * 100)) + contradiction_boost) if morphed_scores else min(94, 6 + contradiction_boost)
    fake_pct = min(94, int(round(max(fake_scores) * 100))) if fake_scores else 0
    combined_pct = min(94, max(morphed_pct, fake_pct))
    uncertainty_pct = max(8, 20 - 4 * len(flagged_modalities))

    if morphed_pct >= 50 and fake_pct >= 50:
        verdict = "Morphed & Fake"
    elif fake_pct >= 50:
        verdict = "Fake (AI-Generated)"
    elif morphed_pct >= 50:
        verdict = "Morphed (Edited)"
    else:
        verdict = "Likely Authentic"

    return {
        "report_id": f"RPT-{uuid.uuid4().hex[:8].upper()}",
        "case_id": f"TL-{uuid.uuid4().hex[:8].upper()}",
        "case_title": case_title,
        "investigator": investigator,
        "generated_at": utcnow_iso(),
        "verdict": {
            "classification_label": verdict,
            "morphed_percent": morphed_pct,
            "fake_percent": fake_pct,
            "combined_morphed_fake_percent": combined_pct,
            "estimated_authentic_percent": max(6, 100 - combined_pct),
            "uncertainty_margin_percent": uncertainty_pct,
            "epistemic_notice": "TruthLens indicators are heuristic and are not calibrated probabilities. Experimental image/video and audio-model scores are not validated probabilities. No detected signal is not proof of authenticity.",
        },
        "source_trace": [{
            "evidence_id": e["evidence_id"], "filename": e["filename"],
            "media_type": e["media_type"], "sha256_hash": e["sha256_hash"], "metadata": e["metadata"],
        } for e in evidence_records],
        "flagged_evidence": all_findings,
        "cross_modal_context": correlations,
        "audit_trail": audit_trail,
    }


def build_sample_evidence() -> Tuple[bytes, bytes, str]:
    img = Image.new("RGB", (480, 320), (45, 70, 110))
    draw = ImageDraw.Draw(img)
    for y in range(320):
        draw.line([(0, y), (480, y)], fill=(45 + y // 8, 70 + y // 6, 110))
    draw.rectangle([280, 140, 420, 250], fill=(235, 195, 60), outline=(20, 20, 20), width=4)
    draw.text((295, 185), "SPLICED", fill=(10, 10, 10))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    comment = b"Midjourney v6 / Adobe Photoshop / C2PA / DateTimeOriginal 2023:05:14 10:22:00"
    com_segment = b"\xff\xfe" + struct.pack(">H", len(comment) + 2) + comment
    raw_jpg = buf.getvalue()
    sample_jpg = raw_jpg[:2] + com_segment + raw_jpg[2:]

    sr = 16000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    signal[sr:int(sr * 1.6)] = 0.0
    pcm = (signal * 32767).astype(np.int16)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    sample_wav = wav_buf.getvalue() + b"LIST\x1a\x00\x00\x00INFOISFT\x0e\x00\x00\x00ElevenLabs v3\x00"

    sample_text = (
        "BREAKING leaked footage from November 2026! Share before deleted! "
        "It is important to note that this pivotal role underscores the importance of immediate action. "
        "In conclusion, this is undeniable proof of the event."
    )
    return sample_jpg, sample_wav, sample_text


# Streamlit interface
st.title("TruthLens", icon=":material/verified_user:")
st.caption("A digital forensics workspace for reviewing media authenticity.")

with st.container(border=True):
    st.subheader("Start an investigation", icon=":material/folder_open:")
    st.caption("Upload evidence to inspect for provenance, editing, and synthetic-media signals.")

    with st.expander("Case details", icon=":material/assignment_ind:"):
        case_col, investigator_col = st.columns(2)
        case_title = case_col.text_input("Case title", value="Media authenticity check")
        investigator = investigator_col.text_input("Investigator", value="Forensic analyst")

    uploaded_files = st.file_uploader(
        "Evidence files",
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "wav", "txt"],
        accept_multiple_files=True,
        help="Supported: JPG, PNG, WebP, MP4, MOV, AVI, WAV, and TXT.",
    )
    context_text = st.text_area(
        "Claim or transcript (optional)",
        height=90,
        placeholder="Add the caption, claim, or transcript you want to compare with the evidence.",
    )

    demo_col, scan_col = st.columns([1, 2], vertical_alignment="center")
    run_demo = demo_col.button(
        "Try sample investigation",
        icon=":material/science:",
        width="stretch",
    )
    run_scan = scan_col.button(
        "Analyze evidence",
        type="primary",
        icon=":material/search:",
        width="stretch",
    )

if run_demo or run_scan:
    evidence_records = []
    all_findings = []
    audit_trail = [{"timestamp": utcnow_iso(), "action": "CASE_OPENED",
                    "details": f"Opened '{case_title}' by {investigator}."}]
    image_visuals = []
    items_to_process = []

    if run_demo:
        s_jpg, s_wav, s_txt = build_sample_evidence()
        items_to_process.extend([
            ("sample_press_photo.jpg", "image", s_jpg),
            ("sample_voice_clip.wav", "audio", s_wav),
            ("sample_viral_caption.txt", "text", s_txt.encode("utf-8")),
        ])
    else:
        for uploaded in uploaded_files or []:
            ext = os.path.splitext(uploaded.name)[1].lower()
            if ext in {".jpg", ".jpeg", ".png", ".webp"}:
                media_type = "image"
            elif ext in {".mp4", ".mov", ".avi"}:
                media_type = "video"
            elif ext == ".wav":
                media_type = "audio"
            else:
                media_type = "text"
            items_to_process.append((uploaded.name, media_type, uploaded.read()))

        if context_text.strip():
            items_to_process.append(("pasted_context.txt", "text", context_text.encode("utf-8")))

    if not items_to_process:
        st.warning("Upload at least one file or click **Try Sample Investigation**.")
        st.stop()

    with st.spinner("Hashing evidence and running forensic engines..."):
        for fname, media_type, raw_bytes in items_to_process:
            evidence_id = f"EV-{uuid.uuid4().hex[:6].upper()}"
            digest = sha256_bytes(raw_bytes)
            audit_trail.append({
                "timestamp": utcnow_iso(), "action": "EVIDENCE_INGESTED",
                "details": f"{fname} ({media_type}) SHA-256:{digest[:16]}...",
            })

            if media_type == "image":
                meta, findings, orig_img, ela_img = analyze_image(raw_bytes, evidence_id, fname, digest)
                if orig_img is not None:
                    image_visuals.append((fname, orig_img, ela_img))
            elif media_type == "video":
                meta, findings = analyze_video(raw_bytes, evidence_id, fname, digest)
            elif media_type == "audio":
                meta, findings = analyze_audio(raw_bytes, evidence_id, fname, digest)
            else:
                meta, findings = analyze_text(raw_bytes.decode("utf-8", errors="ignore"), evidence_id, fname, digest)

            evidence_records.append({
                "evidence_id": evidence_id, "filename": fname,
                "media_type": media_type, "sha256_hash": digest, "metadata": meta,
            })
            all_findings.extend(findings)

        audit_trail.append({
            "timestamp": utcnow_iso(), "action": "ANALYSIS_COMPLETE",
            "details": f"Analyzed {len(evidence_records)} item(s); logged {len(all_findings)} indicator(s).",
        })

    report = correlate_and_build_report(case_title, investigator, evidence_records, all_findings, audit_trail)

    v = report["verdict"]
    st.header("Investigation results", icon=":material/analytics:")
    st.caption(f"{case_title} · {len(evidence_records)} evidence item(s) · Report {report['report_id']}")
    summary_tab, evidence_tab, custody_tab = st.tabs(
        ["Summary", "Evidence details", "Chain of custody"]
    )

    with summary_tab:
        verdict_color = "red" if "Fake" in v["classification_label"] else (
            "orange" if "Morphed" in v["classification_label"] else "green"
        )
        st.badge(
            v["classification_label"],
            color=verdict_color,
            icon=":material/verified_user:",
        )

        metric_cols = st.columns(4, gap="small")
        metric_cols[0].metric("Morphed indicators", f"{v['morphed_percent']}%", border=True, height="stretch")
        metric_cols[1].metric("AI evidence score", f"{v['fake_percent']}%", border=True, height="stretch")
        metric_cols[2].metric(
            "Combined heuristic risk",
            f"{v['combined_morphed_fake_percent']}% ± {v['uncertainty_margin_percent']}%",
            border=True,
            height="stretch",
        )
        metric_cols[3].metric(
            "Unflagged remainder",
            f"{v['estimated_authentic_percent']}%",
            border=True,
            height="stretch",
        )
        st.progress(v["combined_morphed_fake_percent"] / 100.0)
        st.info(v["epistemic_notice"], title="How to interpret these results", icon=":material/info:")

        st.subheader("Experimental model checks", icon=":material/psychology:")
        model_results = [
            (
                source,
                "video",
                "experimental_frame_model",
                "Sampled-frame deepfake-model score",
                "sampled_frame_count",
                "frames",
            )
            for source in report["source_trace"]
            if source["media_type"] == "video"
        ] + [
            (
                source,
                "audio",
                "experimental_voice_model",
                "Sampled-audio deepfake-model score",
                "sampled_segment_count",
                "segments",
            )
            for source in report["source_trace"]
            if source["media_type"] == "audio"
        ]
        if not model_results:
            st.caption("No video or audio model checks were run for this investigation.")
        for source, _, result_key, score_label, count_key, count_label in model_results:
            result = source["metadata"].get(result_key, {})
            with st.container(border=True):
                st.markdown(f"**{source['filename']}**")
                score = result.get("score_percent")
                if score is not None:
                    st.metric(score_label, f"{score:.2f}%")
                    st.caption(
                        f"Scored {result.get(count_key, 0)} {count_label} with "
                        f"`{result.get('model', 'unknown model')}`. {result.get('notice', '')}"
                    )
                    if result.get("status") != "available":
                        st.warning(
                            result.get("error") or "Only part of this media item was scored.",
                            title="Partial model result",
                        )
                else:
                    st.warning(
                        result.get("error") or "No media segments were scored.",
                        title="Model score unavailable",
                    )

        st.subheader("Cross-modal context", icon=":material/link:")
        if report["cross_modal_context"]:
            for correlation in report["cross_modal_context"]:
                with st.container(border=True):
                    st.badge(correlation["relationship_type"], color="blue")
                    st.write(correlation["description"])
                    st.caption(correlation["impact_on_confidence"])
        else:
            st.caption("No cross-modal relationships were identified.")

    with evidence_tab:
        st.subheader("Image inspection", icon=":material/image_search:")
        if image_visuals:
            for fname, orig_img, ela_img in image_visuals:
                original_col, heatmap_col = st.columns(2, gap="medium")
                original_col.image(
                    orig_img,
                    caption=f"Original: {fname}",
                    alt=f"Original evidence image for {fname}",
                    width="stretch",
                )
                heatmap_col.image(
                    ela_img,
                    caption="ELA residual heatmap",
                    alt=f"Compression-error heatmap for {fname}",
                    width="stretch",
                )
        else:
            st.caption("No decodable images were included.")

        st.subheader(f"Detected indicators ({len(report['flagged_evidence'])})", icon=":material/fact_check:")
        if not report["flagged_evidence"]:
            st.success("No manipulation or AI-generation indicators were triggered.")
        else:
            for finding in report["flagged_evidence"]:
                badge_color = "red" if finding["severity"] == "HIGH" else (
                    "orange" if finding["severity"] == "MEDIUM" else "blue"
                )
                with st.container(border=True):
                    st.badge(
                        f"{finding['severity']} · {finding['signal_type'].upper()}",
                        color=badge_color,
                    )
                    st.markdown(f"**{finding['indicator_category']}** · `{finding['filename']}`")
                    st.caption(f"{finding['location']} · {finding['confidence_interval']}")
                    st.write(finding["observed_signal"])
                    st.markdown(f"**Why it matters:** {finding['explanation']}")
                    with st.expander(
                        "Alternative explanations and limitations",
                        icon=":material/info:",
                    ):
                        st.markdown("**Alternative explanations**")
                        st.write("; ".join(finding["alternative_explanations"]))
                        st.markdown("**Limitations**")
                        st.write("; ".join(finding["limitations"]))

    with custody_tab:
        st.subheader("Evidence manifest", icon=":material/fingerprint:")
        for source in report["source_trace"]:
            with st.container(border=True):
                st.markdown(f"**{source['filename']}** · {source['media_type'].title()}")
                st.caption(f"Evidence ID: {source['evidence_id']}")
                st.code(source["sha256_hash"], language=None)

        st.subheader("Audit trail", icon=":material/history:")
        for log in report["audit_trail"]:
            st.caption(f"{log['timestamp']} · {log['action']} · {log['details']}")

    with st.container(border=True):
        st.subheader("Download your report", icon=":material/download:")
        st.caption("Save the complete investigation, findings, and chain-of-custody data as JSON.")
        st.download_button(
            "Download full JSON report",
            data=json.dumps(report, indent=2),
            file_name=f"{report['report_id']}_{report['case_id']}.json",
            mime="application/json",
            icon=":material/download:",
            width="stretch",
        )
# Removed duplicate/corrupted trailing content below the valid UI.  "filename": filename, "media_type": "image", "sha256_hash": sha256_hash, "location": "File Header / Software Tag", "indicator_category": "Editing Software Trace", "signal_type": "morphed", "observed_signal": f"Image header references editing software: {', '.join(detected_editors)}.", "explanation": "This image was exported from an editing program rather than straight from a camera sensor.", "raw_score": 0.62, "confidence_interval": "62% ± 15% (Moderate)", "severity": "MEDIUM", "alternative_explanations": ["Routine cropping, resizing, or color correction by the original photographer."], "limitations": ["Software tags confirm post-processing, not malicious manipulation."], }) elif not has_exif: # Informational only — excluded from manipulation scoring so WhatsApp/screenshots don't false-trigger findings.append({ "evidence_id": evidence_id, "filename": filename, "media_type": "image", "sha256_hash": sha256_hash, "location": "File Header","indicator_category": "Stripped Camera Metadata (Info)", "signal_type": "info", "observed_signal": "No camera EXIF metadata found in file header.", "explanation": "Camera details are absent. This is very common for images shared via WhatsApp, Instagram, or screenshots, as well as AI images.", "raw_score": 0.10, "confidence_interval": "10% ± 10% (Informational)", "severity": "LOW", "alternative_explanations": ["Messaging apps and social platforms automatically strip EXIF data for privacy."], "limitations": ["Missing metadata alone is never proof of editing or AI generation."], }) # Edge-Compensated 8x8 MAD ELA + Sensor Noise Consistency rgb_img = img.convert("RGB") max_dim = 512 scale = min(1.0, max_dim / max(orig_w, orig_h)) w, h = max(32, int(orig_w * scale)), max(32, int(orig_h * scale)) resized = rgb_img.resize((w, h), Image.Resampling.BILINEAR) buf = io.BytesIO() resized.save(buf, format="JPEG", quality=90) buf.seek(0) recomp = Image.open(buf).convert("RGB") orig_arr = np.asarray(resized, dtype=np.float32) recomp_arr = np.asarray(recomp, dtype=np.float32) diff_map = np.mean(np.abs(orig_arr - recomp_arr), axis=2) # Grayscale luminance for edge masking & Laplacian noise gray = 0.299 * orig_arr[:, :, 0] + 0.587 * orig_arr[:, :, 1] + 0.114 * orig_arr[:, :, 2] gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :])) edge_strength = gx + gy # Down-weight sharp natural edges so real photos don't false-trigger edge_weight = 1.0 / (1.0 + edge_strength / 18.0) compensated_ela = diff_map * edge_weight # Laplacian high-pass noise map (only in smooth regions) lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)) smooth_mask = edge_strength < 20.0 # Build visual ELA heatmap ela_vis = np.clip(diff_map * 12.0, 0, 255).astype(np.uint8) heatmap_rgb = np.stack([ ela_vis, np.clip(ela_vis * 0.45, 0, 255).astype(np.uint8), (255 - ela_vis).astype(np.uint8), ], axis=2) ela_heatmap_img = Image.fromarray(heatmap_rgb) # 8x8 Grid Robust MAD Z-Score grid = 8 gh, gw = max(1, h // grid), max(1, w // grid) block_ela = [] block_noise = [] for r in range(grid): for c in range(grid): patch_ela = compensated_ela[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw] if patch_ela.size > 0: block_ela.append((float(np.mean(patch_ela)), r, c)) patch_mask = smooth_mask[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]if np.sum(patch_mask) > 30: patch_lap = lap[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw] block_noise.append(float(np.mean(patch_lap[patch_mask]))) ela_vals = np.array([b[0] for b in block_ela]) median_ela = float(np.median(ela_vals)) mad_ela = float(np.median(np.abs(ela_vals - median_ela))) peak_val, peak_r, peak_c = max(block_ela, key=lambda x: x[0]) robust_z = (peak_val - median_ela) / (1.4826 * max(mad_ela, 0.25)) if robust_z > 3.2 and peak_val > 3.0: provenance_discount = 0.12 if metadata["camera_provenance_detected"] else 0.0 score = max(0.40, min(0.84, round(0.48 + (robust_z - 3.2) * 0.08 provenance_discount, 2))) findings.append({ "evidence_id": evidence_id, "filename": filename, "media_type": "image", "sha256_hash": sha256_hash, "location": f"Region [Row {peak_r + 1}/8, Col {peak_c + 1}/8]", "indicator_category": "Localized Compression Splice (Edge-Compensated ELA)", "signal_type": "morphed", "observed_signal": f"Non-edge compression anomaly (Robust Z={robust_z:.2f}, patch={peak_val:.2f} vs median={median_ela:.2f}).", "explanation": "A localized interior region shows significantly different JPEG compression error than the rest of the photo even after discounting sharp natural edges.", "raw_score": score, "confidence_interval": f"{int(score*100)}% ± 14% ({'High' if score >= 0.7 else 'Moderate'})", "severity": "HIGH" if score >= 0.7 else "MEDIUM", "alternative_explanations": ["Localized text overlays, watermarks, or bright specular reflections.", "Re-saving a previously compressed image at a different quality factor.", ], "limitations": ["Signal-level ELA is less effective on screenshots or heavily downsampled social-media images."], }) if len(block_noise) >= 12: med_n = float(np.median(block_noise)) ratio = max(block_noise) / max(min(block_noise), 0.3) if ratio > 4.8 and med_n > 1.2: score = min(0.74, round(0.46 + (ratio - 4.8) * 0.05, 2)) findings.append({ "evidence_id": evidence_id, "filename": filename, "media_type": "image", "sha256_hash": sha256_hash, "location": "Smooth-Region Noise Floor", "indicator_category": "Sensor Grain / Noise Mismatch", "signal_type": "morphed", "observed_signal": f"Smooth-area noise variance ratio is {ratio:.1f}x across the frame.", "explanation": "Real camera sensors produce fairly uniform grain across smooth areas. A sharp mismatch can indicate an AI-inpainted patch or a pasted object.", "raw_score": score, "confidence_interval": f"{int(score*100)}% ± 16% (Moderate)", "severity": "MEDIUM", "alternative_explanations": ["Portrait-mode bokeh blur, heavy low-light denoising, or out-of-focus backgrounds."], "limitations": ["Shallow depth-of-field photography naturally creates smoother background regions."], })return metadata, findings, resized, ela_heatmap_img # =========================================================================== === # 2. VIDEO FORENSIC ENGINE # =========================================================================== === def analyze_video(raw_bytes: bytes, evidence_id: str, filename: str, sha256_hash: str): findings = [] header_text = extract_header_text(raw_bytes) detected_ai = [name for key, name in AI_VIDEO_SIGNATURES if key in header_text] if detected_ai: findings.append({ "evidence_id": evidence_id, "filename": filename, "media_type": "video", "sha256_hash": sha256_hash, "location": "Video Container Metadata", "indicator_category": "AI Video Platform Signature", "signal_type": "fake", "observed_signal": f"AI video generator trace detected: {', '.join(detected_ai)}.", "explanation": "The video container header references a known generative video platform or C2PA AI tag.", "raw_score": 0.90, "confidence_interval": "90% ± 6% (High)", "severity": "HIGH","alternative_explanations": ["Authentic footage edited or exported through an AI video suite."], "limitations": ["Re-encoding on social platforms strips container metadata."], }) suffix = os.path.splitext(filename)[1] or ".mp4" with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp: tmp.write(raw_bytes) tmp_path = tmp.name try: cap = cv2.VideoCapture(tmp_path) if not cap.isOpened(): return {"error": "Unable to decode video", "ai_platform_traces": detected_ai}, findings fps = cap.get(cv2.CAP_PROP_FPS) or 25.0 frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0 metadata = { "fps": round(fps, 2), "frame_count": frame_count, "resolution": f"{width}x{height}", "duration_sec": duration_sec, "ai_platform_traces": detected_ai, } step = max(1, frame_count // 100) if frame_count > 0 else 1 prev_hist = Nonesharpness_series = [] hist_diffs = [] idx = 0 while True: ret, frame = cap.read() if not ret: break if idx % step == 0: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) sharpness_series.append(float(cv2.Laplacian(gray, cv2.CV_64F).var())) hist = cv2.calcHist([gray], [0], None, [32], [0, 256]) cv2.normalize(hist, hist) if prev_hist is not None: diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)) hist_diffs.append((idx, diff)) prev_hist = hist idx += 1 cap.release() if hist_diffs: diffs_only = np.array([d[1] for d in hist_diffs]) mean_d, std_d = float(diffs_only.mean()), float(diffs_only.std()) + 1e-5 anomalies = [(f, v) for f, v in hist_diffs if (v - mean_d) / std_d > 2.8 and v > 0.28] if anomalies: top_f, top_v = max(anomalies, key=lambda x: x[1]) ts = round(top_f / fps, 2) score = min(0.80, round(0.50 + top_v * 0.35, 2)) findings.append({ "evidence_id": evidence_id, "filename": filename, "media_type": "video","sha256_hash": sha256_hash, "location": f"Frame {top_f} ({ts:.2f}s)", "indicator_category": "Temporal Discontinuity", "signal_type": "morphed", "observed_signal": f"Abrupt histogram shift (Bhattacharyya={top_v:.2f}) at {ts:.2f}s.", "explanation": "Sudden frame-to-frame luminance shifts can indicate spliced segments or temporal flicker.", "raw_score": score, "confidence_interval": f"{int(score*100)}% ± 18% (Moderate)", "severity": "MEDIUM", "alternative_explanations": ["Intentional scene cuts, camera flash, or rapid panning."], "limitations": ["Requires human review of the exact timestamp to rule out normal cuts."], }) if len(sharpness_series) > 10: s_arr = np.array(sharpness_series) cv_sharp = float(s_arr.std() / (s_arr.mean() + 1e-5)) if cv_sharp > 0.70: score = min(0.75, round(0.45 + (cv_sharp - 0.70) * 0.35, 2)) findings.append({ "evidence_id": evidence_id, "filename": filename, "media_type": "video", "sha256_hash": sha256_hash, "location": f"Timeline (0.00s - {duration_sec:.2f}s)", "indicator_category": "Temporal Blur / Focus Instability", "signal_type": "morphed", "observed_signal": f"Sharpness coefficient of variation = {cv_sharp:.2f}.","explanation": "Face-swapping pipelines often introduce inconsistent frame-byframe blurring around synthesized regions.", "raw_score": score, "confidence_interval": f"{int(score*100)}% ± 20% (Low-Moderate)", "severity": "MEDIUM", "alternative_explanations": ["Autofocus hunting, handheld shake, or low-light motion blur."], "limitations": ["Global sharpness variance does not isolate facial regions."], }) return metadata, findings finally: if os.path.exists(tmp_path): os.remove(tmp_path) # =========================================================================== === # 3. AUDIO FORENSIC ENGINE # =========================================================================== === def analyze_audio(raw_bytes: bytes, evidence_id: str, filename: str, sha256_hash: str): findings = [] header_text = extract_header_text(raw_bytes) detected_ai = [name for key, name in AI_AUDIO_SIGNATURES if key in header_text] if detected_ai: findings.append({ "evidence_id": evidence_id, "filename": filename, "media_type": "audio","sha256_hash": sha256_hash, "location": "Audio Container Metadata", "indicator_category": "AI Voice / Audio Platform Signature", "signal_type": "fake", "observed_signal": f"Embedded AI audio tag found: {', '.join(detected_ai)}.", "explanation": "The audio header explicitly references an AI voice cloning or music generation platform.", "raw_score": 0.90, "confidence_interval": "90% ± 6% (High)", "severity": "HIGH", "alternative_explanations": ["Authentic voice cleaned up using an AI audio mastering tool."], "limitations": ["Re-encoding to MP3/AAC often removes RIFF/ID3 metadata."], }) try: sr, data = wavfile.read(io.BytesIO(raw_bytes)) except Exception: return {"note": "Non-WAV container; upload PCM WAV for full spectral analysis", "ai_platform_traces": detected_ai}, findings if data.ndim > 1: data = data.mean(axis=1) data = data.astype(np.float32) data = data / (np.max(np.abs(data)) + 1e-6) duration_sec = round(len(data) / sr, 2) metadata = { "sample_rate_hz": int(sr), "duration_sec": duration_sec, "ai_platform_traces": detected_ai, }freqs, psd = welch(data, fs=sr, nperseg=min(2048, len(data))) total_energy = float(np.sum(psd)) + 1e-12 high_mask = freqs > 7000 high_ratio = float(np.sum(psd[high_mask]) / total_energy) if np.any(high_mask) else 0.0 if sr >= 16000 and high_ratio < 0.002: findings.append({ "evidence_id": evidence_id, "filename": filename, "media_type": "audio", "sha256_hash": sha256_hash, "location": f"Spectral Band >7kHz (0.00s - {duration_sec:.2f}s)", "indicator_category": "High-Frequency Spectral Roll-Off", "signal_type": "fake", "observed_signal": f"Energy above 7kHz is only {high_ratio*100:.4f}% at {sr}Hz sample rate.", "explanation": "Many neural vocoders and cloned speech models exhibit sharp high-frequency cutoffs compared to natural microphones.", "raw_score": 0.64, "confidence_interval": "64% ± 17% (Moderate)", "severity": "MEDIUM", "alternative_explanations": ["Phone calls (VoIP/cellular), Bluetooth headsets, or aggressive noise suppression."], "limitations": ["Modern 48kHz neural vocoders may not show this cutoff."], }) frame_len = max(1, int(sr * 0.1)) num_frames = len(data) // frame_len rms_vals = [float(np.sqrt(np.mean(data[i * frame_len:(i + 1) * frame_len] ** 2))) for i in range(num_frames)] near_zero = [i for i, r in enumerate(rms_vals) if r < 1e-4]if num_frames > 10 and (len(near_zero) / num_frames) > 0.12: start_s, end_s = round(near_zero[0] * 0.1, 2), round(near_zero[-1] * 0.1, 2) findings.append({ "evidence_id": evidence_id, "filename": filename, "media_type": "audio", "sha256_hash": sha256_hash, "location": f"Audio Segments ({start_s:.2f}s - {end_s:.2f}s)", "indicator_category": "Synthetic Digital Silence / Spliced Pauses", "signal_type": "morphed", "observed_signal": f"{len(near_zero)} of {num_frames} windows exhibit near-zero digital silence.", "explanation": "Natural recordings keep a steady room-tone noise floor. Pure digital zeros between words often point to TTS generation or spliced audio.", "raw_score": 0.68, "confidence_interval": "68% ± 15% (Moderate)", "severity": "MEDIUM", "alternative_explanations": ["Studio noise gates or conferencing apps (Zoom/Krisp)."], "limitations": ["Cannot distinguish synthetic silence from aggressive noise-gate post-processing."], }) return metadata, findings # =========================================================================== === # 4. TEXT & LLM-ORIGIN FORENSIC ENGINE# =========================================================================== === def analyze_text(content: str, evidence_id: str, filename: str, sha256_hash: str): findings = [] words = content.split() lower_text = content.lower() extracted_dates = list(set(DATE_PATTERN.findall(content))) triggered_cues = [cue for cue in URGENCY_CUES if cue in lower_text] llm_hits = [p for p in LLM_PHRASES if p in lower_text] sentences = [s.strip() for s in re.split(r"[.!?]+", content) if len(s.strip().split()) > 2] burstiness = None if len(sentences) >= 3: lengths = np.array([len(s.split()) for s in sentences], dtype=np.float32) burstiness = float(lengths.std() / (lengths.mean() + 1e-5)) metadata = { "word_count": len(words), "extracted_dates": extracted_dates, "llm_phrase_hits": llm_hits, "sentence_burstiness": round(burstiness, 2) if burstiness is not None else None, } if len(llm_hits) >= 2 or (len(llm_hits) >= 1 and burstiness is not None and burstiness < 0.25): score = min(0.78, round(0.55 + 0.10 * len(llm_hits), 2)) findings.append({ "evidence_id": evidence_id, "filename": filename, "media_type": "text", "sha256_hash": sha256_hash,"location": f"Text Body ({len(words)} words)", "indicator_category": "AI / LLM Writing Pattern", "signal_type": "fake", "observed_signal": f"Matched LLM phrasing markers ({', '.join(llm_hits)}) with uniform sentence rhythm.", "explanation": "Text exhibits stylistic uniformity and phrasing patterns frequently produced by ChatGPT, Claude, or Gemini.", "raw_score": score, "confidence_interval": f"{int(score*100)}% ± 18% (Moderate)", "severity": "MEDIUM", "alternative_explanations": ["Formal press releases, PR copy, or non-native English writers using grammar assistants."], "limitations": ["Text-only AI detection has high variance on short passages."], }) if triggered_cues: findings.append({ "evidence_id": evidence_id, "filename": filename, "media_type": "text", "sha256_hash": sha256_hash, "location": f"Text Body ({len(words)} words)", "indicator_category": "Sensational / Disinformation Framing", "signal_type": "morphed", "observed_signal": f"High-urgency framing detected: {', '.join(triggered_cues)}.", "explanation": "Manipulated media campaigns frequently pair altered visuals with urgent captions to bypass critical verification.", "raw_score": 0.50, "confidence_interval": "50% ± 20% (Moderate-Low)", "severity": "LOW", "alternative_explanations": ["Authentic breaking news or emotional eyewitness posts."],"limitations": ["Linguistic framing is a contextual risk factor, not proof of fabrication."], }) return metadata, findings # =========================================================================== === # 5. CROSS-MODAL CORRELATOR & MORPHED / FAKE SCORING # =========================================================================== === def correlate_and_build_report(case_title, investigator, evidence_records, all_findings, audit_trail): correlations = [] images = [e for e in evidence_records if e["media_type"] == "image"] videos = [e for e in evidence_records if e["media_type"] == "video"] audios = [e for e in evidence_records if e["media_type"] == "audio"] texts = [e for e in evidence_records if e["media_type"] == "text"] # Cross-Modal Check 1: Image EXIF Date vs Text Claimed Dates for img in images: exif_date = img["metadata"].get("exif_date") if exif_date: exif_year = exif_date[:4] for txt in texts: claimed = txt["metadata"].get("extracted_dates", []) if claimed and not any(exif_year in d for d in claimed): correlations.append({ "correlation_id": f"XMOD-{uuid.uuid4().hex[:5].upper()}","involved_evidence_ids": [img["evidence_id"], txt["evidence_id"]], "relationship_type": "CONTRADICTION", "description": f"Image '{img['filename']}' header date ({exif_date}) conflicts with dates claimed in text ({', '.join(claimed)}).", "impact_on_confidence": "Strong indicator of out-of-context reuse or temporal misattribution (+10% risk).", }) # Cross-Modal Check 2: Video vs Audio Duration Drift for vid in videos: v_dur = vid["metadata"].get("duration_sec", 0) for aud in audios: a_dur = aud["metadata"].get("duration_sec", 0) if v_dur > 0 and a_dur > 0 and abs(v_dur - a_dur) > 1.5: correlations.append({ "correlation_id": f"XMOD-{uuid.uuid4().hex[:5].upper()}", "involved_evidence_ids": [vid["evidence_id"], aud["evidence_id"]], "relationship_type": "CONTRADICTION", "description": f"Video duration ({v_dur}s) differs from audio duration ({a_dur}s) by {abs(v_dur - a_dur):.2f}s.", "impact_on_confidence": "Suggests asynchronous dubbing or audio replacement (+10% risk).", }) active_findings = [f for f in all_findings if f.get("signal_type") != "info"] flagged_modalities = list({f["media_type"] for f in active_findings if f["raw_score"] >= 0.55}) if len(flagged_modalities) >= 2: correlations.append({ "correlation_id": f"XMOD-{uuid.uuid4().hex[:5].upper()}", "involved_evidence_ids": list({f["evidence_id"] for f in active_findings if f["raw_score"] >= 0.55}),"relationship_type": "REINFORCEMENT", "description": f"Independent anomalies detected across {len(flagged_modalities)} modalities ({', '.join(flagged_modalities)}).", "impact_on_confidence": "Converging signals across separate channels reduce single-sensor false positives.", }) if not correlations and len(evidence_records) >= 2: correlations.append({ "correlation_id": f"XMOD-{uuid.uuid4().hex[:5].upper()}", "involved_evidence_ids": [e["evidence_id"] for e in evidence_records], "relationship_type": "ALIGNMENT", "description": "No direct cross-modal contradictions detected across timestamps, durations, or metadata.", "impact_on_confidence": "Neutral — absence of contradiction does not guarantee authenticity.", }) # Morphed vs Fake Calibrated Percentages (excluding 'info' signals) contradiction_boost = 10 * sum(1 for c in correlations if c["relationship_type"] == "CONTRADICTION") morphed_scores = [f["raw_score"] for f in active_findings if f.get("signal_type") == "morphed"] fake_scores = [f["raw_score"] for f in active_findings if f.get("signal_type") == "fake"] morphed_pct = min(94, int(round(max(morphed_scores) * 100)) + contradiction_boost) if morphed_scores else (min(94, 6 + contradiction_boost)) fake_pct = min(94, int(round(max(fake_scores) * 100))) if fake_scores else 5 combined_pct = min(94, max(morphed_pct, fake_pct)) uncertainty_pct = max(8, 20 - 4 * len(flagged_modalities)) if morphed_pct >= 50 and fake_pct >= 50:verdict = "Morphed & Fake" elif fake_pct >= 50: verdict = "Fake (AI-Generated)" elif morphed_pct >= 50: verdict = "Morphed (Edited)" else: verdict = "Likely Authentic" return { "report_id": f"RPT-{uuid.uuid4().hex[:8].upper()}", "case_id": f"TL-{uuid.uuid4().hex[:8].upper()}", "case_title": case_title, "investigator": investigator, "generated_at": utcnow_iso(), "verdict": { "classification_label": verdict, "morphed_percent": morphed_pct, "fake_percent": fake_pct, "combined_morphed_fake_percent": combined_pct, "estimated_authentic_percent": max(6, 100 - combined_pct), "uncertainty_margin_percent": uncertainty_pct, "epistemic_notice": ( "TruthLens provides calibrated forensic indicators, not absolute proof. " "Benign re-encoding, social-media compression, or noise gates can trigger signals. " "Always review flagged evidence with human judgment." ), }, "source_trace": [ { "evidence_id": e["evidence_id"], "filename": e["filename"],"media_type": e["media_type"], "sha256_hash": e["sha256_hash"], "metadata": e["metadata"], } for e in evidence_records ], "flagged_evidence": all_findings, "cross_modal_context": correlations, "audit_trail": audit_trail, } # =========================================================================== === # 6. 1-CLICK SAMPLE INVESTIGATION GENERATOR # =========================================================================== === def build_sample_evidence() -> Tuple[bytes, bytes, str]: # 1. Spliced JPEG with Midjourney + Photoshop + 2023 EXIF comment img = Image.new("RGB", (480, 320), (45, 70, 110)) draw = ImageDraw.Draw(img) for y in range(320): draw.line([(0, y), (480, y)], fill=(45 + y // 8, 70 + y // 6, 110)) # High-frequency spliced patch draw.rectangle([280, 140, 420, 250], fill=(235, 195, 60), outline=(20, 20, 20), width=4) draw.text((295, 185), "SPLICED", fill=(10, 10, 10)) buf = io.BytesIO() img.save(buf, format="JPEG", quality=92) # Inject JPEG COM marker with AI/Editor & 2023 datecomment = b"Midjourney v6 / Adobe Photoshop / C2PA / DateTimeOriginal 2023:05:14 10:22:00" com_segment = b"\xff\xfe" + struct.pack(">H", len(comment) + 2) + comment raw_jpg = buf.getvalue() sample_jpg = raw_jpg[:2] + com_segment + raw_jpg[2:] # 2. Narrowband WAV with digital silence gaps + ElevenLabs tag sr = 16000 t = np.linspace(0, 3.0, sr * 3, endpoint=False) signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t) signal[sr:int(sr * 1.6)] = 0.0  # Pure digital silence gap pcm = (signal * 32767).astype(np.int16) wav_buf = io.BytesIO() with wave.open(wav_buf, "wb") as wf: wf.setnchannels(1) wf.setsampwidth(2) wf.setframerate(sr) wf.writeframes(pcm.tobytes()) sample_wav = wav_buf.getvalue() + b"LIST\x1a\x00\x00\x00INFOISFT\x0e\x00\x00\x00ElevenLabs v3\x00" # 3. Conflicting 2026 viral caption with LLM phrasing sample_text = ( "BREAKING leaked footage from November 2026! Share before deleted! " "It is important to note that this pivotal role underscores the importance of immediate action. " "In conclusion, this is undeniable proof of the event." ) return sample_jpg, sample_wav, sample_text# =========================================================================== === # 7. STREAMLIT USER INTERFACE (SIMPLE 3-STEP LIGHT-MODE WORKBENCH) # =========================================================================== === st.title("🔍 TruthLens — Multimodal Digital Forensics") st.caption("Check images, videos, audio, and text for editing (Morphed) or AI generation (Fake) with transparent confidence ranges.") top_col1, top_col2 = st.columns([3, 1]) with top_col2: run_demo = st.button("⚡ Try Sample Investigation", use_container_width=True, type="secondary") with st.expander("Optional Case Info (Title & Investigator)", expanded=False): c1, c2 = st.columns(2) case_title = c1.text_input("Case Title", value="Media Authenticity Check") investigator = c2.text_input("Investigator", value="Forensic Analyst") uploaded_files = st.file_uploader( "Step 1: Upload Images, Videos, Audio (WAV), or Text Files", type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "wav", "txt"], accept_multiple_files=True, ) context_text = st.text_area( "Optional: Paste accompanying caption, claim, or transcript for cross-modal checking", height=100, placeholder="e.g., 'Breaking news photo taken in November 2026...'", )run_scan = st.button("🔎 Check for Edits & AI", type="primary", use_container_width=True) if run_demo or run_scan: evidence_records = [] all_findings = [] audit_trail = [{"timestamp": utcnow_iso(), "action": "CASE_OPENED", "details": f"Opened '{case_title}' by {investigator}."}] image_visuals = [] items_to_process = [] if run_demo: s_jpg, s_wav, s_txt = build_sample_evidence() items_to_process.append(("sample_press_photo.jpg", "image", s_jpg)) items_to_process.append(("sample_voice_clip.wav", "audio", s_wav)) items_to_process.append(("sample_viral_caption.txt", "text", s_txt.encode("utf-8"))) else: for f in uploaded_files or []: ext = os.path.splitext(f.name)[1].lower() m_type = "image" if ext in {".jpg", ".jpeg", ".png", ".webp"} else ( "video" if ext in {".mp4", ".mov", ".avi"} else ( "audio" if ext == ".wav" else "text" ) ) items_to_process.append((f.name, m_type, f.read())) if context_text.strip(): items_to_process.append(("pasted_context.txt", "text", context_text.encode("utf8"))) if not items_to_process: st.warning("Please upload at least one file or enter text context, or click **Try Sample Investigation**.") st.stop()with st.spinner("Step 2: Hashing evidence & running multimodal forensic engines..."): for fname, m_type, raw_bytes in items_to_process: ev_id = f"EV-{uuid.uuid4().hex[:6].upper()}" digest = sha256_bytes(raw_bytes) audit_trail.append({ "timestamp": utcnow_iso(), "action": "EVIDENCE_INGESTED", "details": f"{fname} ({m_type}) SHA-256:{digest[:16]}...", }) if m_type == "image": meta, findings, orig_img, ela_img = analyze_image(raw_bytes, ev_id, fname, digest) image_visuals.append((fname, orig_img, ela_img)) elif m_type == "video": meta, findings = analyze_video(raw_bytes, ev_id, fname, digest) elif m_type == "audio": meta, findings = analyze_audio(raw_bytes, ev_id, fname, digest) else: meta, findings = analyze_text(raw_bytes.decode("utf-8", errors="ignore"), ev_id, fname, digest) evidence_records.append({ "evidence_id": ev_id, "filename": fname, "media_type": m_type, "sha256_hash": digest, "metadata": meta, }) all_findings.extend(findings) audit_trail.append({"timestamp": utcnow_iso(), "action": "ANALYSIS_COMPLETE", "details": f"Analyzed {len(evidence_records)} item(s); logged {len(all_findings)} indicator(s).", }) report = correlate_and_build_report(case_title, investigator, evidence_records, all_findings, audit_trail) # ========================================================================== # STEP 3: EASY-TO-READ FORENSIC REPORT # ========================================================================== st.divider() v = report["verdict"] st.subheader(f"Step 3: Forensic Result — {v['classification_label']}") m1, m2, m3, m4 = st.columns(4) m1.metric("Morphed (Edited)", f"{v['morphed_percent']}%") m2.metric("Fake (AI-Generated)", f"{v['fake_percent']}%") m3.metric("Combined Risk", f"{v['combined_morphed_fake_percent']}% ± {v['uncertainty_margin_percent']}%") m4.metric("Est. Authentic", f"{v['estimated_authentic_percent']}%") st.progress(v["combined_morphed_fake_percent"] / 100.0) st.info(f"**Forensic Trust Notice:** {v['epistemic_notice']}") # Cross-Modal Correlation st.markdown("### 🔗 Cross-Modal Correlation") for c in report["cross_modal_context"]: badge = "badge-high" if c["relationship_type"] == "CONTRADICTION" else ( "badge-med" if c["relationship_type"] == "REINFORCEMENT" else "badge-low") st.markdown( f"<div class='verdict-card'><span class='{badge}'>{c['relationship_type']}</span> " f"&nbsp; <strong>{c['description']}</strong><br/><small> {c['impact_on_confidence']}</small></div>", unsafe_allow_html=True, ) # Visual Comparison (Original vs ELA Heatmap) if image_visuals: st.markdown("### 🖼 Image Inspection (Original vs. Edit Highlight Map)") for fname, orig_img, ela_img in image_visuals: col_a, col_b = st.columns(2) col_a.image(orig_img, caption=f"Original: {fname}", use_container_width=True) col_b.image(ela_img, caption="ELA Residual Heatmap (Brighter = higher compression diff)", use_container_width=True) # Flagged Findings st.markdown(f"### 📋 Detected Indicators ({len(report['flagged_evidence'])})") if not report["flagged_evidence"]: st.success("No manipulation or AI-generation indicators were triggered.") else: for f in report["flagged_evidence"]: badge_cls = "badge-info" if f["signal_type"] == "info" else ( "badge-high" if f["severity"] == "HIGH" else ("badge-med" if f["severity"] == "MEDIUM" else "badge-low") ) with st.container(border=True): st.markdown( f"<span class='{badge_cls}'>{f['severity']} • {f['signal_type'].upper()}</span> " f"&nbsp; **{f['indicator_category']}** — `{f['filename']}` ({f['confidence_interval']})",) unsafe_allow_html=True, st.write(f"**Where:** {f['location']}  \n**What was seen:** {f['observed_signal']} \n**Why it matters:** {f['explanation']}") with st.expander("Innocent Explanations & Limitations"): st.write("**Alternative Benign Explanations:**", "; ".join(f["alternative_explanations"])) st.write("**Methodological Limitations:**", "; ".join(f["limitations"])) # Chain of Custody & Export with st.expander("🔐 Chain-of-Custody (SHA-256 Fingerprints & Audit Trail)"): for src in report["source_trace"]: st.code(f"{src['evidence_id']} | {src['filename']} ({src['media_type']})\nSHA-256: {src['sha256_hash']}") for log in report["audit_trail"]: st.text(f"[{log['timestamp']}] {log['action']}: {log['details']}") st.download_button( label="⬇ Download Full JSON Forensic Report", data=json.dumps(report, indent=2), file_name=f"{report['report_id']}_{report['case_id']}.json", mime="application/json", use_container_width=True, )