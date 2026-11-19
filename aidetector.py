import io
import os
import re
import json
import uuid
import wave
import struct
import hashlib
import tempfile
import importlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageChops, ImageFilter

# ===========================================================================
# PAGE CONFIG & LIGHT-MODE STYLING
# ===========================================================================
st.set_page_config(
    page_title="TruthLens — Digital Forensics & AI Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .verdict-card {
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        margin-bottom: 1rem;
    }
    .badge-high { background:#fee2e2; color:#991b1b; padding:4px 10px; border-radius:999px; font-weight:600; font-size:0.8rem; }
    .badge-med { background:#fef3c7; color:#92400e; padding:4px 10px; border-radius:999px; font-weight:600; font-size:0.8rem; }
    .badge-low { background:#dcfce7; color:#166534; padding:4px 10px; border-radius:999px; font-weight:600; font-size:0.8rem; }
    .badge-info { background:#e0f2fe; color:#075985; padding:4px 10px; border-radius:999px; font-weight:600; font-size:0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# SIGNATURE DATABASES
# ===========================================================================
AI_IMAGE_SIGNATURES = [
    ("midjourney", "Midjourney"),
    ("dall-e", "OpenAI DALL-E"),
    ("openai", "OpenAI Image Generator"),
    ("stable diffusion", "Stable Diffusion"),
    ("comfyui", "ComfyUI / Stable Diffusion"),
    ("adobe firefly", "Adobe Firefly"),
    ("imagen", "Google Imagen"),
    ("flux", "Black Forest Labs Flux"),
    ("leonardo", "Leonardo AI"),
    ("c2pa", "C2PA Content Credentials"),
    ("trainedalgorithmicmedia", "C2PA AI-Generated Tag"),
]

EDITING_SIGNATURES = [
    "photoshop", "gimp", "lightroom", "canva", "snapseed", "facetune", "picsart", "pixlr"
]

CAMERA_SIGNATURES = [
    "apple", "iphone", "samsung", "sm-", "google", "pixel", "canon", "nikon",
    "sony", "fujifilm", "xiaomi", "redmi", "oneplus", "vivo", "oppo", "motorola", "nothing"
]

AI_VIDEO_SIGNATURES = [
    ("sora", "OpenAI Sora"),
    ("veo", "Google Veo"),
    ("runway", "Runway Gen"),
    ("kling", "Kling AI"),
    ("pika", "Pika Labs"),
    ("luma", "Luma Dream Machine"),
    ("heygen", "HeyGen"),
    ("synthesia", "Synthesia"),
    ("c2pa", "C2PA Content Credentials"),
]
AI_VIDEO_FRAME_MODEL = "dima806/deepfake_vs_real_image_detection"
AI_VIDEO_FRAME_SAMPLE_COUNT = 12

AI_AUDIO_SIGNATURES = [
    ("elevenlabs", "ElevenLabs"),
    ("suno", "Suno AI"),
    ("udio", "Udio"),
    ("lyria", "Google Lyria"),
    ("openai", "OpenAI TTS"),
    ("playht", "PlayHT"),
    ("murf", "Murf AI"),
    ("stable audio", "Stable Audio"),
]

DATE_PATTERN = re.compile(
    r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+20\d{2}|20\d{2})\b",
    re.IGNORECASE,
)

URGENCY_CUES = [
    "breaking", "leaked", "confirmed 100%", "share before deleted", "urgent", "undeniable proof"
]

LLM_PHRASES = [
    "it is important to note", "in conclusion", "delve into", "tapestry of",
    "testament to", "underscores the importance", "navigate the complexities", "pivotal role"
]

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def extract_header_text(raw_bytes: bytes) -> str:
    head = raw_bytes[:131072]
    tail = raw_bytes[-32768:] if len(raw_bytes) > 131072 else b""
    return (head + b"\n" + tail).decode("latin-1", errors="ignore").lower()

# ===========================================================================
# 1. UPGRADED FORENSIC ENGINES (WITH CACHING & ROBUST FALLBACK)
# ===========================================================================

@st.cache_resource(show_spinner=False)
def load_neural_pipeline():
    # Load the optional dependency lazily so the app still runs without it.
    pipeline = importlib.import_module("transformers").pipeline
    return pipeline("image-classification", model=AI_VIDEO_FRAME_MODEL)

def run_neural_detection(raw_bytes: bytes) -> Tuple[float, float, str]:
    try:
        classifier = load_neural_pipeline()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        preds = classifier(img)
        fake_score, real_score = 0.0, 0.0
        for p in preds:
            label = p['label'].lower()
            if 'fake' in label:
                fake_score = max(fake_score, p['score'])
            elif 'real' in label:
                real_score = max(real_score, p['score'])
        total = fake_score + real_score
        if total > 0:
            return round((fake_score / total) * 100, 2), round((real_score / total) * 100, 2), "Neural ViT"
    except Exception:
        return 50.0, 50.0, "Fallback (Error)"
    return 50.0, 50.0, "Fallback (Error)"

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

def analyze_image(raw_bytes: bytes, evidence_id: str, filename: str, sha256_hash: str, custom_ela_thresh: float, custom_gradient_thresh: float, verification_mode: str):
    findings = []
    header_text = extract_header_text(raw_bytes)
    detected_ai = [name for key, name in AI_IMAGE_SIGNATURES if key in header_text]
    detected_editors = [ed for ed in EDITING_SIGNATURES if ed in header_text]
    has_exif = ("exif" in header_text) or ("datetimeoriginal" in header_text)
    has_camera_make = any(cam in header_text for cam in CAMERA_SIGNATURES)

    exif_date_match = re.search(r"(20\d{2}):(\d{2}):(\d{2})", header_text)
    exif_date = exif_date_match.group(0) if exif_date_match else None
    
    img = Image.open(io.BytesIO(raw_bytes))
    orig_w, orig_h = img.size

    metadata = {
        "resolution": f"{orig_w}x{orig_h}",
        "format": img.format,
        "has_exif": has_exif,
        "exif_date": exif_date,
        "camera_provenance_detected": has_exif and has_camera_make and len(detected_ai) == 0,
        "ai_platform_traces": detected_ai,
        "editing_software_traces": detected_editors,
        "raw_bytes": raw_bytes
    }

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
            "explanation": "The file header explicitly contains metadata or C2PA provenance tags associated with an AI image generator.",
            "raw_score": 0.90,
            "confidence_interval": "90% ± 6% (High)",
            "severity": "HIGH",
            "alternative_explanations": ["An authentic photo edited using an AI-assisted tool (e.g., Generative Fill) can inherit C2PA AI tags."],
            "limitations": ["Metadata can be stripped when re-saved or shared on social media."],
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
            "explanation": "This image was exported from an editing program rather than straight from a camera sensor.",
            "raw_score": 0.62,
            "confidence_interval": "62% ± 15% (Moderate)",
            "severity": "MEDIUM",
            "alternative_explanations": ["Routine cropping, resizing, or color correction by the original photographer."],
            "limitations": ["Software tags confirm post-processing, not malicious manipulation."],
        })

    # Core Mathematical Image Forensics
    rgb_img = img.convert("RGB")
    max_dim = 512
    scale = min(1.0, max_dim / max(orig_w, orig_h))
    w, h = max(32, int(orig_w * scale)), max(32, int(orig_h * scale))
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

    # Build ELA colormap map (Simulated Thermal Spectrum: Blue=Low, Red=High)
    ela_vis = np.clip(diff_map * 12.0, 0, 255).astype(np.uint8)
    norm_ela = ela_vis / 255.0
    h_map, w_map = norm_ela.shape
    heatmap_rgb = np.zeros((h_map, w_map, 3), dtype=np.uint8)
    
    heatmap_rgb[:, :, 0] = np.clip(np.sin(norm_ela * np.pi - np.pi/2) * 255, 0, 255).astype(np.uint8) # Red
    heatmap_rgb[:, :, 1] = np.clip(np.sin(norm_ela * np.pi) * 255, 0, 255).astype(np.uint8)          # Green
    heatmap_rgb[:, :, 2] = np.clip(np.cos(norm_ela * np.pi/2) * 255, 0, 255).astype(np.uint8)        # Blue
    
    ela_heatmap_img = Image.fromarray(heatmap_rgb)

    grid = 8
    gh, gw = max(1, h // grid), max(1, w // grid)
    block_ela = []
    block_noise = []
    for r in range(grid):
        for c in range(grid):
            patch_ela = compensated_ela[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
            if patch_ela.size > 0:
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

    # Evaluate Localized Splicing
    if robust_z > custom_ela_thresh and peak_val > 3.0:
        provenance_discount = 0.12 if metadata["camera_provenance_detected"] else 0.0
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
            "explanation": "A localized interior region shows significantly different JPEG compression error than the rest of the photo.",
            "raw_score": score,
            "confidence_interval": f"{int(score*100)}% ± 14% (Moderate)",
            "severity": "HIGH" if score >= 0.7 else "MEDIUM",
            "alternative_explanations": ["Localized text overlays, watermarks, or bright specular reflections."],
            "limitations": ["Signal-level ELA is less effective on screenshots or heavily downsampled images."],
        })

    # Evaluate Noise Floor ratio (AI Patterning)
    if len(block_noise) >= 12:
        med_n = float(np.median(block_noise))
        ratio = max(block_noise) / max(min(block_noise), 0.3)
        if ratio > custom_gradient_thresh and med_n > 1.2:
            score = min(0.74, round(0.46 + (ratio - 4.8) * 0.05, 2))
            findings.append({
                "evidence_id": evidence_id,
                "filename": filename,
                "media_type": "image",
                "sha256_hash": sha256_hash,
                "location": "Smooth-Region Noise Floor",
                "indicator_category": "Sensor Grain / Noise Mismatch",
                "signal_type": "morphed",
                "observed_signal": f"Smooth-area noise variance ratio is {ratio:.1f}x across the frame.",
                "explanation": "Real camera sensors produce fairly uniform grain. A sharp mismatch points to AI inpainting or patch edits.",
                "raw_score": score,
                "confidence_interval": f"{int(score*100)}% ± 16% (Moderate)",
                "severity": "MEDIUM",
                "alternative_explanations": ["Portrait-mode bokeh blur, heavy low-light denoising, or out-of-focus backgrounds."],
                "limitations": ["Shallow depth-of-field photography naturally creates smoother background regions."],
            })

    # Neural classification integration if selected
    if "Neural" in verification_mode or "Ensemble" in verification_mode:
        fake_n, real_n, label_n = run_neural_detection(raw_bytes)
        if label_n != "Fallback (Error)":
            findings.append({
                "evidence_id": evidence_id,
                "filename": filename,
                "media_type": "image",
                "sha256_hash": sha256_hash,
                "location": "Deep Learning Pixel Patch",
                "indicator_category": "Vision Transformer (ViT) Classification",
                "signal_type": "fake",
                "observed_signal": f"Neural deepfake pattern index: {fake_n}%",
                "explanation": "Transformer network detected synthetic high-frequency noise textures across pixel patches.",
                "raw_score": fake_n / 100.0,
                "confidence_interval": f"{int(fake_n)}% ± 6%",
                "severity": "HIGH" if fake_n >= 70.0 else "MEDIUM",
                "alternative_explanations": ["Highly compressed artwork or digital drawing profiles."],
                "limitations": ["ViT classifications can be skewed by heavy re-encoding."],
            })

    return metadata, findings, resized, ela_heatmap_img

# Video Forensics
def analyze_video(raw_bytes: bytes, evidence_id: str, filename: str, sha256_hash: str, custom_flicker_thresh: float):
    findings = []
    header_text = extract_header_text(raw_bytes)
    detected_ai = [name for key, name in AI_VIDEO_SIGNATURES if key in header_text]
    if detected_ai:
        findings.append({
            "evidence_id": evidence_id,
            "filename": filename,
            "media_type": "video",
            "sha256_hash": sha256_hash,
            "location": "Video Container Metadata",
            "indicator_category": "AI Video Platform Signature",
            "signal_type": "info",
            "observed_signal": f"AI video generator trace detected: {', '.join(detected_ai)}.",
            "explanation": "Container metadata references a known generative video platform or C2PA AI tag; this is provenance evidence, not a probability that the video is AI-generated.",
            "raw_score": 0.0,
            "confidence_interval": "Provenance trace; no probability calculated",
            "severity": "INFO",
            "alternative_explanations": ["Authentic footage edited or exported through an AI video suite."],
            "limitations": ["Re-encoding on social platforms strips container metadata."],
        })

    suffix = os.path.splitext(filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
    cap = None
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {"error": "Unable to decode video", "ai_platform_traces": detected_ai}, findings

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0

        sample_indices = set(
            np.linspace(0, frame_count - 1, min(AI_VIDEO_FRAME_SAMPLE_COUNT, frame_count), dtype=int).tolist()
        ) if frame_count > 0 else set()
        frame_ai_scores = []
        frame_ai_error = None
        expected_ai_samples = (
            min(AI_VIDEO_FRAME_SAMPLE_COUNT, frame_count) if frame_count > 0 else None
        )
        metadata = {
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "resolution": f"{width}x{height}",
            "duration_sec": duration_sec,
            "ai_platform_traces": detected_ai,
        }

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
                len(frame_ai_scores) < AI_VIDEO_FRAME_SAMPLE_COUNT and idx % step == 0
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
            mean_d, std_d = float(diffs_only.mean()), float(diffs_only.std()) + 1e-5
            anomalies = [(f, v) for f, v in hist_diffs if (v - mean_d) / std_d > 2.8 and v > custom_flicker_thresh]
            if anomalies:
                top_f, top_v = max(anomalies, key=lambda x: x[1])
                ts = round(top_f / fps, 2)
                score = min(0.80, round(0.50 + top_v * 0.35, 2))
                findings.append({
                    "evidence_id": evidence_id,
                    "filename": filename,
                    "media_type": "video",
                    "sha256_hash": sha256_hash,
                    "location": f"Frame {top_f} ({ts:.2f}s)",
                    "indicator_category": "Temporal Discontinuity",
                    "signal_type": "morphed",
                    "observed_signal": f"Abrupt histogram shift (Bhattacharyya={top_v:.2f}) at {ts:.2f}s.",
                    "explanation": "Sudden frame-to-frame luminance shifts can indicate spliced segments or temporal flicker.",
                    "raw_score": score,
                    "confidence_interval": f"{int(score*100)}% ± 18% (Moderate)",
                    "severity": "MEDIUM",
                    "alternative_explanations": ["Intentional scene cuts, camera flash, or rapid panning."],
                    "limitations": ["Requires human review of the exact timestamp to rule out normal cuts."],
                })

        if len(sharpness_series) > 10:
            s_arr = np.array(sharpness_series)
            cv_sharp = float(s_arr.std() / (s_arr.mean() + 1e-5))
            if cv_sharp > 0.70:
                score = min(0.75, round(0.45 + (cv_sharp - 0.70) * 0.35, 2))
                findings.append({
                    "evidence_id": evidence_id,
                    "filename": filename,
                    "media_type": "video",
                    "sha256_hash": sha256_hash,
                    "location": f"Timeline (0.00s - {duration_sec:.2f}s)",
                    "indicator_category": "Inconsistent Frame Sharpness",
                    "signal_type": "morphed",
                    "observed_signal": f"Frame sharpness varies substantially over time (coefficient of variation={cv_sharp:.2f}).",
                    "explanation": "Large changes in sharpness can indicate inconsistent processing or compositing across video frames.",
                    "raw_score": score,
                    "confidence_interval": f"{int(score*100)}% ± 18% (Moderate)",
                    "severity": "MEDIUM",
                    "alternative_explanations": ["Autofocus changes, motion blur, or scenes with varying depth of field."],
                    "limitations": ["Natural camera focus changes can produce similar sharpness variation."],
                })

        return metadata, findings
    finally:
        if cap is not None:
            cap.release()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)