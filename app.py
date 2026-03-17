import streamlit as st
import streamlit.components.v1 as components
import os
import hashlib
from copy import deepcopy
from pathlib import Path
import io
import re
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter

from retrieve_v2 import HybridRetrieverV2
from rerank_v2 import TwoStageCalibratedReranker
from llm import GenerationClient
from html_renderer import build_answer_html
from query import expand_query
from config import (
    EXAMPLE_QUERIES,
    INGEST_EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    SPARSE_MODE,
    VECTOR_DB_BACKEND,
    ENABLE_IMAGE_PIPELINE,
    ENABLE_YOLO_PIPELINE,
    WEB_FALLBACK_ENABLED,
)

RUNTIME_CACHE_ROOT = Path(".runtime-cache").resolve()
PADDLE_HOME_DIR = RUNTIME_CACHE_ROOT / "paddle"
PADDLEX_HOME_DIR = RUNTIME_CACHE_ROOT / "paddlex"
YOLO_CONFIG_DIR = RUNTIME_CACHE_ROOT / "ultralytics"

for runtime_dir in (RUNTIME_CACHE_ROOT, PADDLE_HOME_DIR, PADDLEX_HOME_DIR, YOLO_CONFIG_DIR):
    runtime_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PADDLE_HOME", str(PADDLE_HOME_DIR))
os.environ.setdefault("PADDLEX_HOME", str(PADDLEX_HOME_DIR))
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
os.environ.setdefault("ULTRALYTICS_SETTINGS", str((YOLO_CONFIG_DIR / "settings.json").resolve()))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _resolve_yolo_weights() -> Path:
    project_dir = Path(__file__).resolve().parent
    candidates = [
        project_dir / "v3_yolo11m" / "weights" / "best.pt",
        project_dir.parent.parent / "v3_yolo11m" / "weights" / "best.pt",
        RUNTIME_CACHE_ROOT / "yolov8n.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return RUNTIME_CACHE_ROOT / "yolov8n.pt"

try:
    import voice
except Exception:
    voice = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

st.set_page_config(
    page_title="RAG for Climate Challenges",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# UI Styling
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    .stApp { background-color: #ffffff; }
    .block-container {
        max-width: 720px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
    h1 { font-weight: 500; font-size: 1.6rem; color: #111; letter-spacing: -0.02em; }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 12px 16px;
        font-size: 15px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #999;
        box-shadow: none;
    }
    .stButton > button {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        background: #fafafa;
        color: #333;
        font-size: 13px;
        padding: 8px 14px;
        font-weight: 400;
    }
    .stButton > button:hover {
        background: #f0f0f0;
        border-color: #ccc;
    }
    .stSpinner > div { color: #666; }
    .voice-status {
        font-size: 12px;
        color: #888;
        margin-top: 2px;
        margin-bottom: 4px;
    }
    .monitor-note {
        font-size: 12px;
        color: #666;
        margin-top: -4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_retriever():
    sparse_mode = SPARSE_MODE if SPARSE_MODE in {"none", "bm42", "splade"} else "bm42"

    def _has_qdrant_collection(path: str, collection: str) -> bool:
        storage = Path(path) / "collection" / collection / "storage.sqlite"
        return storage.exists()

    candidates = [
        (QDRANT_PATH, QDRANT_COLLECTION),
        ("./qdrant_db_ci", "hvac_documents_qdrant_ci"),
        ("./qdrant_db", "hvac_documents_qdrant"),
    ]
    qdrant_path, qdrant_collection = next(
        ((path, collection) for path, collection in candidates if _has_qdrant_collection(path, collection)),
        (QDRANT_PATH, QDRANT_COLLECTION),
    )

    return HybridRetrieverV2(
        backend="qdrant",
        embedding_model=INGEST_EMBEDDING_MODEL,
        sparse_mode=sparse_mode,
        qdrant_path=qdrant_path,
        qdrant_collection=qdrant_collection,
    )


@st.cache_resource
def get_reranker_model():
    return TwoStageCalibratedReranker()


@st.cache_resource
def get_generator():
    return GenerationClient()


@st.cache_resource
def get_paddle_ocr_model():
    if PaddleOCR is None:
        return None
    return PaddleOCR(lang="en", use_textline_orientation=True)


@st.cache_resource
def get_yolo_model():
    if YOLO is None:
        return None
    yolo_weights = _resolve_yolo_weights()
    return YOLO(str(yolo_weights))


@st.cache_resource
def load_whisper_model():
    if voice is None:
        return None
    try:
        return voice.load_model()
    except Exception:
        return None


def _init_voice_state():
    defaults = {
        "voice_query": "",
        "voice_status": "",
        "show_recorder": False,
        "just_transcribed": False,
        "use_image_input": False,
        "last_monitor": None,
        "image_pipeline_cache": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _image_cache_key(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def _record_stage(monitor: dict[str, Any], stage: str, start_time: float, status: str = "ok", **extra: Any) -> None:
    entry = {
        "stage": stage,
        "duration_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
        "status": status,
    }
    entry.update(extra)
    monitor.setdefault("stages", []).append(entry)


def _extract_fields_from_ocr_text(text: str, lines: list[dict[str, Any]] | None = None) -> dict[str, str]:
    """Extract HVAC nameplate-like fields from OCR text using robust regex patterns."""
    fields: dict[str, str] = {}
    compact = " ".join((text or "").split())
    if not compact:
        return fields

    compact_upper = compact.upper()
    normalized = compact_upper.replace("–", "-").replace("—", "-")

    def _extract(pattern: str) -> str | None:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            return None
        value = next((group for group in match.groups() if group is not None), None)
        if value is None:
            value = match.group(0)
        return str(value).strip(" :;,.|\t\n\r")

    def _set(field_name: str, value: str | None):
        if value:
            fields[field_name] = value

    def _sanitize_voltage(value: str | None) -> str | None:
        if not value:
            return None
        raw = str(value).upper().replace("TO", "-").replace(" ", "")
        numbers = re.findall(r"\d{2,3}", raw)
        if not numbers:
            return None
        if len(numbers) == 1:
            return numbers[0]
        if len(numbers) == 2:
            if "/" in raw:
                return f"{numbers[0]}/{numbers[1]}"
            return f"{numbers[0]}-{numbers[1]}"
        if len(numbers) >= 4 and ("/" in raw or "-" in raw):
            return f"{numbers[0]}-{numbers[1]}/{numbers[2]}-{numbers[3]}"
        return "/".join(numbers[:2])

    def _voltage_score(value: str) -> tuple[int, int, int]:
        numbers = [int(n) for n in re.findall(r"\d{2,3}", value)]
        if not numbers:
            return (-1, -1, -1)
        max_n = max(numbers)
        min_n = min(numbers)
        score = 0
        if max_n >= 100:
            score += 3
        if max_n >= 200:
            score += 1
        if "/" in value or "-" in value:
            score += 2
        if len(numbers) >= 2:
            score += 1
        if min_n < 50 and max_n < 100:
            score -= 2
        return (score, max_n, len(numbers))

    def _looks_like_code(value: str | None, *, require_digit: bool = True, min_len: int = 4) -> bool:
        if not value:
            return False
        token = str(value).strip().upper()
        if len(token) < min_len:
            return False
        blocked = {
            "CONTAINS",
            "WARNING",
            "CAUTION",
            "MODEL",
            "SERIAL",
            "NUMBER",
            "NOMINAL",
            "INPUT",
            "OUTPUT",
            "PHASE",
        }
        if token in blocked:
            return False
        if require_digit and not any(ch.isdigit() for ch in token):
            return False
        return bool(re.match(r"^[A-Z0-9][A-Z0-9\-_/\.]{2,}$", token))

    def _looks_like_serial(value: str | None, model_value: str | None = None) -> bool:
        if not _looks_like_code(value, require_digit=True, min_len=6):
            return False
        token = str(value).strip().upper()
        if model_value and token == str(model_value).strip().upper():
            return False
        blocked_fragments = {"COS", "PF", "VOLT", "INPUT", "PHASE", "AMP", "CURRENT"}
        if any(fragment in token for fragment in blocked_fragments):
            return False
        if re.match(r"^\d{2,3}(?:[-/]\d{2,3})+V?-?$", token):
            return False
        letter_count = sum(ch.isalpha() for ch in token)
        digit_count = sum(ch.isdigit() for ch in token)
        return letter_count >= 1 and digit_count >= 3

    # Model / Serial
    model_value = _extract(r"\b(?:MODEL|M/N|M#|M\s*NO\.?|MODEL\s*NO\.?)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{3,})")
    if _looks_like_code(model_value, require_digit=True, min_len=4):
        _set("model", model_value)

    serial_value = _extract(r"\b(?:SERIAL|S/N|S#|SER\.?\s*NO\.?|SERIAL\s*NO\.?)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{4,})")
    if _looks_like_serial(serial_value, model_value=model_value):
        _set("serial", serial_value)

    # Refrigerant
    allowed_refs = {
        "R22",
        "R32",
        "R134A",
        "R290",
        "R410A",
        "R407C",
        "R404A",
        "R600A",
        "R1234YF",
        "R1234ZE",
    }

    refrigerant = _extract(r"\b(?:REFRIGERANT|REFRIGERAT|REF\.?|COOLANT)[^A-Z0-9]{0,12}(R\s*[-]?\s*\d{2,3}[A-Z]?)")
    if not refrigerant:
        generic_ref = _extract(r"\b(R\s*[-]?\s*\d{2,3}[A-Z]?)\b")
        if generic_ref:
            ref_token = generic_ref.replace(" ", "").replace("-", "")
            if ref_token in allowed_refs:
                refrigerant = generic_ref
    if refrigerant:
        cleaned_ref = refrigerant.replace(" ", "").replace("-", "")
        if cleaned_ref in allowed_refs:
            _set("refrigerant", refrigerant.replace(" ", ""))
    elif "HCFC-22" in normalized or "HCFC22" in normalized:
        _set("refrigerant", "R-22")
    elif "R22" in normalized:
        _set("refrigerant", "R-22")

    # Electrical fields
    voltage_candidates: list[str] = []
    voltage_patterns = [
        r"\b(?:AC\s*VOLT(?:S)?|VOLT(?:S)?|VOLTAGE|INPUT)\s*[:\-]?\s*([0-9]{2,3}[A-Z]?(?:\s*(?:/|-|TO)\s*[0-9]{2,3}[A-Z]?){0,3})",
        r"\bU\s*[:\-]?\s*([0-9]{2,3}[A-Z]?(?:\s*(?:/|-|TO)\s*[0-9]{2,3}[A-Z]?){0,3})",
        r"\b([0-9]{2,3}[A-Z]?(?:\s*(?:/|-|TO)\s*[0-9]{2,3}[A-Z]?){0,3})\s*V(?:OLTS?)?\b",
    ]
    for pattern in voltage_patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            captured = match.group(match.lastindex or 0)
            sanitized = _sanitize_voltage(captured)
            if sanitized:
                voltage_candidates.append(sanitized)

    if lines:
        for item in lines:
            line_text = str(item.get("text", "")).upper().strip()
            if not line_text:
                continue
            if any(tag in line_text for tag in ("VOLT", "V ", " V", "INPUT", " U ")):
                for match in re.finditer(r"([0-9]{2,3}[A-Z]?(?:\s*(?:/|-)\s*[0-9]{2,3}[A-Z]?){0,3})", line_text):
                    sanitized = _sanitize_voltage(match.group(1))
                    if sanitized:
                        voltage_candidates.append(sanitized)

    if voltage_candidates:
        unique_candidates = list(dict.fromkeys(voltage_candidates))
        chosen_voltage = sorted(unique_candidates, key=_voltage_score, reverse=True)[0]
        _set("voltage", chosen_voltage)

    frequency_value = _extract(r"\b(\d{2})\s*H(?:Z|B)?\b")
    if not frequency_value and lines:
        for item in lines:
            line_text = str(item.get("text", "")).upper()
            if "HZ" in line_text or "HB" in line_text:
                match = re.search(r"\b(\d{2})\b", line_text)
                if match:
                    frequency_value = match.group(1)
                    break
    _set("frequency", frequency_value)

    phase_value = _extract(r"\b(\d)\s*(?:PH|PHASE|PHASI)\b")
    if not phase_value:
        phase_match = re.search(
            r"\b\d{2,3}(?:\s*(?:/|-|TO)\s*\d{2,3})?\s*V?\s*[,;:/\-\s]*([123])\s*P\b",
            normalized,
            flags=re.IGNORECASE,
        )
        if phase_match:
            phase_value = phase_match.group(1)
    _set("phase", phase_value)

    # Charge fields
    charge_match = re.search(
        r"\b(?:CHARGE|FACTORY\s*CHARGE)?\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(KG|G|GRAMS?|LBS?|LB|OZS?|OZ)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if charge_match:
        _set("charge_value", charge_match.group(1))
        _set("charge_unit", charge_match.group(2).upper().replace("GRAMS", "G").replace("GRAM", "G"))

    # Current / capacity metrics
    _set("rla", _extract(r"\bRLA\b\s*[:\-]?\s*(\d+(?:\.\d+)?)"))
    _set("fla", _extract(r"\bFLA\b\s*[:\-]?\s*(\d+(?:\.\d+)?)"))
    _set("lra", _extract(r"\bLRA\b\s*[:\-]?\s*(\d+(?:\.\d+)?)"))
    _set("capacity_btu", _extract(r"\b(\d{4,6})\s*(?:BTU|BTUH)\b"))

    current_value = _extract(r"\b(?:AMP(?:S)?|CURRENT|IN)\s*[:\-]?\s*(\d+(?:\.\d+)?(?:\s*(?:/|-)\s*\d+(?:\.\d+)?)?)\s*A\b")
    _set("current_amp", current_value)

    input_power = _extract(r"\b(?:RATED\s*)?INPUT\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(KW|W)\b")
    if input_power:
        power_match = re.search(r"(\d+(?:\.\d+)?)\s*(KW|W)", input_power, flags=re.IGNORECASE)
        if power_match:
            _set("input_power_value", power_match.group(1))
            _set("input_power_unit", power_match.group(2).upper())

    kva_value = _extract(r"\b(?:KVA|kVA)\s*[:\-]?\s*(\d+(?:\.\d+)?)\b")
    _set("kva", kva_value)

    # Manufacturer
    brands = [
        "DAIKIN",
        "LG",
        "SAMSUNG",
        "PANASONIC",
        "MITSUBISHI",
        "HITACHI",
        "CARRIER",
        "TRANE",
        "LLOYD",
        "BLUE STAR",
        "VOLTAS",
        "WHIRLPOOL",
        "GODREJ",
    ]
    for brand in brands:
        if brand in normalized:
            _set("manufacturer", brand)
            break

    # Canonical aliases for compatibility
    if "model" in fields:
        fields.setdefault("model_number", fields["model"])
    if "serial" in fields:
        fields.setdefault("serial_number", fields["serial"])
    if "voltage" in fields:
        fields.setdefault("voltage_volts", fields["voltage"])
    if "frequency" in fields:
        fields.setdefault("frequency_hz", fields["frequency"])

    # Additional token-level fallback for model/serial when keywords are fragmented
    if lines:
        tokens = [str(item.get("text", "")).strip().upper() for item in lines if str(item.get("text", "")).strip()]
        if "model" not in fields:
            model_candidate = next(
                (
                    token
                    for token in tokens
                    if re.match(r"^[A-Z0-9]{2,}[A-Z0-9\-_/\.]{3,}$", token)
                    and any(ch.isdigit() for ch in token)
                ),
                None,
            )
            _set("model", model_candidate)
            if model_candidate:
                fields.setdefault("model_number", model_candidate)

        if "serial" not in fields:
            serial_candidate = next(
                (
                    token
                    for token in tokens
                    if _looks_like_serial(token, model_value=fields.get("model"))
                    and token != fields.get("model", "")
                ),
                None,
            )
            if serial_candidate:
                _set("serial", serial_candidate)
                fields.setdefault("serial_number", serial_candidate)

    return fields


def _run_paddle_ocr(
    image_bytes: bytes,
    monitor: dict[str, Any],
    detections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    if PaddleOCR is None:
        _record_stage(
            monitor,
            "image_ocr_paddle",
            start,
            status="error",
            reason="paddleocr package not installed",
        )
        return {
            "available": False,
            "error": "PaddleOCR is not installed. Install with: pip install paddleocr paddlepaddle",
            "ocr_text": "",
            "avg_confidence": 0.0,
            "line_count": 0,
            "fields": {},
        }

    try:
        model = get_paddle_ocr_model()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        def _as_list(value: Any) -> list[Any]:
            if value is None:
                return []
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            if hasattr(value, "tolist"):
                converted = value.tolist()
                if isinstance(converted, list):
                    return converted
                return [converted]
            try:
                return list(value)
            except TypeError:
                return [value]

        def _run_ocr_variant(pil_image: Image.Image, x_offset: int = 0, y_offset: int = 0, variant: str = "orig") -> dict[str, Any]:
            work_image = pil_image
            scale_x = 1.0
            scale_y = 1.0
            max_side = 1800
            if max(pil_image.width, pil_image.height) > max_side:
                resize_ratio = max_side / float(max(pil_image.width, pil_image.height))
                resized_w = max(1, int(pil_image.width * resize_ratio))
                resized_h = max(1, int(pil_image.height * resize_ratio))
                work_image = pil_image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
                scale_x = pil_image.width / float(resized_w)
                scale_y = pil_image.height / float(resized_h)

            prediction = model.predict(np.array(work_image))
            if prediction is None:
                result = {}
            else:
                prediction_list = _as_list(prediction)
                result = prediction_list[0] if prediction_list else {}

            rec_texts = _as_list(result.get("rec_texts"))
            rec_scores = _as_list(result.get("rec_scores"))
            rec_boxes = _as_list(result.get("rec_boxes"))

            lines: list[dict[str, Any]] = []
            confidences: list[float] = []
            for idx, raw_text in enumerate(rec_texts):
                text = str(raw_text).strip()
                if not text:
                    continue
                score = None
                if idx < len(rec_scores):
                    try:
                        score = float(rec_scores[idx])
                        confidences.append(score)
                    except Exception:
                        score = None

                bbox = None
                if idx < len(rec_boxes):
                    raw_box = rec_boxes[idx]
                    try:
                        points = _as_list(raw_box)
                        if points:
                            first_point = points[0]
                            first_seq = _as_list(first_point)
                            if len(first_seq) >= 2:
                                xs = [float(_as_list(point)[0]) for point in points if len(_as_list(point)) >= 2]
                                ys = [float(_as_list(point)[1]) for point in points if len(_as_list(point)) >= 2]
                                if xs and ys:
                                    bbox = [
                                        int(min(xs) * scale_x + x_offset),
                                        int(min(ys) * scale_y + y_offset),
                                        int(max(xs) * scale_x + x_offset),
                                        int(max(ys) * scale_y + y_offset),
                                    ]
                    except Exception:
                        bbox = None

                lines.append(
                    {
                        "text": text,
                        "confidence": round(score, 4) if score is not None else None,
                        "box": bbox,
                    }
                )

            joined = "\n".join(item["text"] for item in lines)
            avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
            fields = _extract_fields_from_ocr_text(joined, lines=lines)
            return {
                "ocr_text": joined,
                "avg_confidence": avg_conf,
                "line_count": len(lines),
                "fields": fields,
                "lines": lines,
                "variant": variant,
            }

        def _ocr_from_image(pil_image: Image.Image, x_offset: int = 0, y_offset: int = 0) -> dict[str, Any]:
            base = _run_ocr_variant(pil_image, x_offset=x_offset, y_offset=y_offset, variant="orig")
            variants: list[tuple[str, Image.Image]] = []
            area = pil_image.width * pil_image.height
            small_image = area <= 700_000 and (pil_image.width < 1100 or pil_image.height < 700)
            tiny_image = area <= 250_000 and (pil_image.width < 850 or pil_image.height < 600)

            if base.get("line_count", 0) == 0 or len(base.get("fields") or {}) <= 2:
                if small_image:
                    upscaled = pil_image.resize(
                        (max(1, pil_image.width * 2), max(1, pil_image.height * 2)),
                        Image.Resampling.LANCZOS,
                    )
                    variants.append(("up2x", upscaled))

                if tiny_image:
                    upscaled_3x = pil_image.resize(
                        (max(1, pil_image.width * 3), max(1, pil_image.height * 3)),
                        Image.Resampling.LANCZOS,
                    )
                    variants.append(("up3x", upscaled_3x))

                gray = ImageOps.grayscale(pil_image)
                contrast = ImageOps.autocontrast(gray)
                variants.append(("autocontrast", contrast.convert("RGB")))

                sharpened = contrast.filter(ImageFilter.SHARPEN)
                variants.append(("autocontrast_sharpen", sharpened.convert("RGB")))

                if small_image:
                    contrast_up = contrast.resize(
                        (max(1, contrast.width * 2), max(1, contrast.height * 2)),
                        Image.Resampling.LANCZOS,
                    )
                    variants.append(("autocontrast_up2x", contrast_up.convert("RGB")))

                if tiny_image:
                    sharpen_up = sharpened.resize(
                        (max(1, sharpened.width * 2), max(1, sharpened.height * 2)),
                        Image.Resampling.LANCZOS,
                    )
                    variants.append(("autocontrast_sharpen_up2x", sharpen_up.convert("RGB")))

            candidates = [base]
            for variant_name, variant_image in variants:
                try:
                    candidate = _run_ocr_variant(
                        variant_image,
                        x_offset=x_offset,
                        y_offset=y_offset,
                        variant=variant_name,
                    )
                    candidates.append(candidate)
                except Exception:
                    continue

            def _variant_key(candidate: dict[str, Any]) -> tuple[float, int, float]:
                return (
                    float(len(candidate.get("fields") or {})),
                    int(candidate.get("line_count") or 0),
                    float(candidate.get("avg_confidence") or 0.0),
                )

            best = max(candidates, key=_variant_key)
            best["variant_candidates"] = [
                {
                    "variant": item.get("variant"),
                    "line_count": int(item.get("line_count") or 0),
                    "field_count": len(item.get("fields") or {}),
                    "avg_confidence": round(float(item.get("avg_confidence") or 0.0), 4),
                }
                for item in candidates
            ]
            return best

        candidates: list[dict[str, Any]] = []
        base_result = _ocr_from_image(image)
        base_result["source"] = "full_image"
        candidates.append(base_result)

        detections = detections or []
        for index, detection in enumerate(detections[:3]):
            bbox = detection.get("box")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            width = max(1, image.width)
            height = max(1, image.height)
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            small_box = box_w * box_h < 120_000
            pad_ratio = 0.2 if small_box else 0.1
            pad_x = max(10, int(box_w * pad_ratio))
            pad_y = max(10, int(box_h * pad_ratio))
            crop_box = (
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(width, x2 + pad_x),
                min(height, y2 + pad_y),
            )
            if crop_box[2] - crop_box[0] < 30 or crop_box[3] - crop_box[1] < 30:
                continue
            crop = image.crop(crop_box)
            crop_result = _ocr_from_image(crop, x_offset=crop_box[0], y_offset=crop_box[1])
            crop_result["source"] = f"detected_crop_{index}"
            crop_result["crop_box"] = list(crop_box)
            candidates.append(crop_result)

        def _candidate_key(candidate: dict[str, Any]) -> tuple[float, float, int]:
            return (
                float(len(candidate.get("fields") or {})),
                float(candidate.get("avg_confidence") or 0.0),
                int(candidate.get("line_count") or 0),
            )

        selected = max(candidates, key=_candidate_key)
        fields = selected.get("fields") or {}
        _record_stage(
            monitor,
            "image_ocr_paddle",
            start,
            status="ok",
            line_count=int(selected.get("line_count") or 0),
            avg_confidence=round(float(selected.get("avg_confidence") or 0.0), 4),
            field_count=len(fields),
            selected_source=selected.get("source"),
            candidate_count=len(candidates),
        )
        return {
            "available": True,
            "error": None,
            "ocr_text": selected.get("ocr_text") or "",
            "avg_confidence": float(selected.get("avg_confidence") or 0.0),
            "line_count": int(selected.get("line_count") or 0),
            "fields": fields,
            "lines": selected.get("lines") or [],
            "selected_source": selected.get("source"),
            "selected_variant": selected.get("variant"),
            "variant_candidates": selected.get("variant_candidates") or [],
            "candidates": [
                {
                    "source": item.get("source"),
                    "variant": item.get("variant"),
                    "line_count": item.get("line_count"),
                    "field_count": len(item.get("fields") or {}),
                    "avg_confidence": round(float(item.get("avg_confidence") or 0.0), 4),
                    "crop_box": item.get("crop_box"),
                }
                for item in candidates
            ],
        }
    except Exception as exc:
        _record_stage(
            monitor,
            "image_ocr_paddle",
            start,
            status="error",
            reason=str(exc),
        )
        return {
            "available": False,
            "error": str(exc),
            "ocr_text": "",
            "avg_confidence": 0.0,
            "line_count": 0,
            "fields": {},
        }


def _run_yolo_detection(image_bytes: bytes, monitor: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    if not ENABLE_YOLO_PIPELINE:
        _record_stage(monitor, "image_yolo", start, status="skipped", reason="disabled")
        return {"available": False, "error": "YOLO disabled", "objects": []}

    if YOLO is None:
        _record_stage(
            monitor,
            "image_yolo",
            start,
            status="error",
            reason="ultralytics package not installed",
        )
        return {"available": False, "error": "ultralytics not installed", "objects": []}

    try:
        model = get_yolo_model()
        if model is None:
            _record_stage(monitor, "image_yolo", start, status="error", reason="YOLO model unavailable")
            return {"available": False, "error": "YOLO model unavailable", "objects": []}

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        array = np.array(image)
        predictions = model.predict(array, verbose=False, conf=0.15, max_det=10)

        objects: list[dict[str, Any]] = []
        if predictions:
            first = predictions[0]
            names = first.names if hasattr(first, "names") else {}
            boxes = getattr(first, "boxes", None)
            if boxes is not None and getattr(boxes, "cls", None) is not None:
                for idx, cls_id in enumerate(boxes.cls.tolist()):
                    class_name = names.get(int(cls_id), str(int(cls_id))) if isinstance(names, dict) else str(int(cls_id))
                    confidence = None
                    if getattr(boxes, "conf", None) is not None and idx < len(boxes.conf.tolist()):
                        confidence = float(boxes.conf.tolist()[idx])
                    bbox = None
                    if getattr(boxes, "xyxy", None) is not None and idx < len(boxes.xyxy.tolist()):
                        box_row = boxes.xyxy.tolist()[idx]
                        bbox = [int(box_row[0]), int(box_row[1]), int(box_row[2]), int(box_row[3])]
                    objects.append(
                        {
                            "label": class_name,
                            "confidence": round(confidence, 4) if confidence is not None else None,
                            "box": bbox,
                            "x1": bbox[0] if bbox else None,
                            "y1": bbox[1] if bbox else None,
                            "x2": bbox[2] if bbox else None,
                            "y2": bbox[3] if bbox else None,
                        }
                    )

        weights_path = getattr(getattr(model, "ckpt", None), "path", None)
        _record_stage(
            monitor,
            "image_yolo",
            start,
            status="ok",
            detected=len(objects),
            model_weights=str(weights_path) if weights_path else str(_resolve_yolo_weights()),
        )
        return {
            "available": True,
            "error": None,
            "objects": objects,
            "image_size": {"width": image.width, "height": image.height},
        }
    except Exception as exc:
        _record_stage(monitor, "image_yolo", start, status="error", reason=str(exc))
        return {"available": False, "error": str(exc), "objects": []}


def _build_image_augmented_query(user_query: str, image_artifacts: dict[str, Any], yolo_artifacts: dict[str, Any]) -> str:
    query = (user_query or "").strip()
    if not query:
        query = "Explain the HVAC equipment details and key operational/safety guidance from this image"

    fields = image_artifacts.get("fields") or {}
    ocr_text = " ".join(str(image_artifacts.get("ocr_text") or "").split())
    objects = yolo_artifacts.get("objects") or []
    parts = [query]
    if fields:
        field_text = "; ".join(f"{key}: {value}" for key, value in fields.items())
        parts.append(f"Extracted fields: {field_text}")
    if objects:
        object_text = "; ".join(
            f"{obj.get('label')} ({round((obj.get('confidence') or 0.0) * 100, 1)}%)"
            for obj in objects
        )
        parts.append(f"Detected objects: {object_text}")
    if ocr_text:
        parts.append(f"OCR text: {ocr_text[:500]}")
    return "\n".join(parts)


def _draw_yolo_overlay(image_bytes: bytes | None, detections: list[dict[str, Any]]) -> Image.Image | None:
    if not image_bytes or not detections:
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        for detection in detections:
            bbox = detection.get("box")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            label = str(detection.get("label", "object"))
            confidence = detection.get("confidence")
            if confidence is not None:
                label = f"{label} ({confidence:.2f})"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1 + 4, max(0, y1 - 14)), label, fill="red")
        return image
    except Exception:
        return None


def _render_voice_recorder(whisper_model):
    """Render text/voice/image inputs while preserving existing UI layout."""
    url_query = st.query_params.get("q", "")
    auto_run_from_url = bool(url_query)
    
    if url_query:
        st.session_state["query_input"] = url_query
        st.session_state["last_search"] = url_query
        st.query_params.clear()
    elif st.session_state["voice_query"]:
        st.session_state["query_input"] = st.session_state["voice_query"]
        st.session_state["last_search"] = ""
        st.session_state["voice_query"] = ""

    col_form, col_mic = st.columns([11, 1])
    with col_form:
        with st.form("query_form", clear_on_submit=False):
            query = st.text_input(
                "Ask a question",
                placeholder="e.g. What is India's cooling action plan?",
                label_visibility="collapsed",
                key="query_input",
            )
            form_submitted = st.form_submit_button("Search")

    with col_mic:
        mic_clicked = st.button(
            "Mic",
            key="mic_btn",
            help="Record a question. Supports English and Indian languages.",
            use_container_width=True,
            disabled=(whisper_model is None),
        )

    if mic_clicked:
        st.session_state["show_recorder"] = not st.session_state["show_recorder"]

    if st.session_state["show_recorder"]:
        audio_value = st.audio_input(
            "Speak your question",
            label_visibility="collapsed",
            key="voice_recorder",
        )
        if audio_value is not None:
            with st.spinner("Transcribing..."):
                try:
                    audio_np = voice.decode_audio(audio_value)
                    text, status = voice.transcribe(whisper_model, audio_np)
                    if text:
                        st.session_state["voice_query"] = text
                        st.session_state["voice_status"] = status
                    else:
                        st.session_state["voice_status"] = "No speech detected. Please try again."
                    st.session_state["show_recorder"] = False
                    st.session_state["just_transcribed"] = True
                except Exception as exc:
                    st.session_state["voice_status"] = f"Transcription error: {exc}"
            st.rerun()

    if st.session_state["voice_status"]:
        st.markdown(
            f'<p class="voice-status">{st.session_state["voice_status"]}</p>',
            unsafe_allow_html=True,
        )

    if ENABLE_IMAGE_PIPELINE:
        st.session_state["use_image_input"] = st.checkbox(
            "Use image input",
            value=st.session_state.get("use_image_input", False),
            key="use_image_input_checkbox",
            help="Enable this to parse uploaded image text with PaddleOCR and include it in retrieval.",
        )
    else:
        st.session_state["use_image_input"] = False

    image_file = None
    if st.session_state["use_image_input"]:
        image_file = st.file_uploader(
            "Upload equipment image",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="image_input_file",
            help="PaddleOCR will parse text from this image and add it to the query pipeline.",
        )

    if form_submitted:
        st.session_state["last_search"] = query

    should_search = bool(form_submitted or auto_run_from_url)
    return st.session_state.get("last_search", ""), image_file, should_search


def _retrieve(
    query: str,
    retriever: HybridRetrieverV2,
    reranker: TwoStageCalibratedReranker,
    generator: GenerationClient,
    monitor: dict[str, Any],
) -> list:
    """Expansion, retrieval, and reranking pipeline with stage monitoring."""
    # Use generator's groq client for expansion (or just use dedicated client)
    expand_started = time.perf_counter()
    queries = expand_query(query, generator.groq)
    _record_stage(
        monitor,
        "expand_queries",
        expand_started,
        status="ok",
        expanded_queries=len(queries),
        expanded_query_list=queries,
    )

    seen_ids = set()
    candidates = []
    retrieval_started = time.perf_counter()
    for q in queries:
        for result in retriever.search(q):
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                candidates.append(result)
    _record_stage(
        monitor,
        "hybrid_retrieve",
        retrieval_started,
        status="ok",
        candidate_count=len(candidates),
    )

    rerank_started = time.perf_counter()
    reranked = reranker.rerank(query, candidates)
    _record_stage(
        monitor,
        "rerank",
        rerank_started,
        status="ok",
        reranked_count=len(reranked),
    )
    return reranked


def _render_monitoring_panel(
    monitor: dict[str, Any],
    image_artifacts: dict[str, Any] | None,
    yolo_artifacts: dict[str, Any] | None,
    generation_artifacts: dict[str, Any] | None,
    image_bytes: bytes | None = None,
):
    with st.expander("Monitoring / Observability", expanded=False):
        st.markdown("<p class='monitor-note'>Stage-level visibility for query execution, OCR, retrieval and generation.</p>", unsafe_allow_html=True)
        stages = monitor.get("stages", [])
        stage_index: dict[str, list[dict[str, Any]]] = {}
        for item in stages:
            stage_name = str(item.get("stage", "unknown"))
            stage_index.setdefault(stage_name, []).append(item)

        def _latest(stage_name: str) -> dict[str, Any]:
            entries = stage_index.get(stage_name) or []
            return entries[-1] if entries else {}

        latest_expand = _latest("expand_queries")
        latest_retrieve = _latest("hybrid_retrieve")
        latest_rerank = _latest("rerank")

        total_latency = float(monitor.get("total_latency_ms", 0.0) or 0.0)
        candidate_count = int(latest_retrieve.get("candidate_count", 0) or 0)
        reranked_count = int(latest_rerank.get("reranked_count", 0) or 0)

        with st.expander("1) Run Overview", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Latency (s)", f"{(total_latency / 1000.0):.2f}")
            col2.metric("Stages", str(len(stages)))
            col3.metric("Candidates", str(candidate_count))
            col4.metric("Reranked", str(reranked_count))
            st.caption(f"Exact total latency: {total_latency:.2f} ms")

            summary = {
                "mode": monitor.get("mode", "text"),
                "effective_query_preview": (monitor.get("effective_query", "") or "")[:300],
                "qdrant_backend": "qdrant",
                "sparse_mode": SPARSE_MODE,
                "embedding_model": INGEST_EMBEDDING_MODEL,
                "reranker": type(get_reranker_model()).__name__,
                "web_fallback_enabled": WEB_FALLBACK_ENABLED,
                "image_cache_hit": bool(monitor.get("image_cache_hit", False)),
                "image_cache_size": len(st.session_state.get("image_pipeline_cache") or {}),
            }
            st.json(summary)

            if stages:
                timeline_rows = [
                    {
                        "Stage": item.get("stage", ""),
                        "Status": item.get("status", ""),
                        "Duration (ms)": item.get("duration_ms", 0.0),
                    }
                    for item in stages
                ]
                st.dataframe(timeline_rows, use_container_width=True)

        with st.expander("2) Query & Reconstruction", expanded=False):
            effective_query = (monitor.get("effective_query", "") or "").strip()
            if effective_query:
                st.text_area("Effective query", value=effective_query, height=120)

            reconstruction = monitor.get("query_reconstruction") or {}
            if reconstruction:
                st.markdown("**Reconstruction details**")
                st.json(reconstruction)

            expanded_queries = latest_expand.get("expanded_query_list") or []
            if expanded_queries:
                st.markdown("**Expanded query list**")
                st.dataframe(
                    [{"index": index + 1, "query": text} for index, text in enumerate(expanded_queries)],
                    use_container_width=True,
                )

            if latest_expand:
                st.markdown("**Expansion stage metadata**")
                stage_payload = {k: v for k, v in latest_expand.items() if k not in {"stage", "status", "duration_ms", "expanded_query_list"}}
                if stage_payload:
                    st.json(stage_payload)

        if image_artifacts or yolo_artifacts:
            with st.expander("3) Image Detection & OCR", expanded=False):
                if yolo_artifacts:
                    detections = yolo_artifacts.get("objects") or []
                    yolo_summary = {
                        "available": yolo_artifacts.get("available", False),
                        "error": yolo_artifacts.get("error"),
                        "detected_regions": len(detections),
                        "image_size": yolo_artifacts.get("image_size"),
                    }
                    st.markdown("**YOLO detection summary**")
                    st.json(yolo_summary)
                    if detections:
                        st.dataframe(detections, use_container_width=True)
                        overlay = _draw_yolo_overlay(image_bytes, detections)
                        if overlay is not None:
                            st.image(overlay, caption="YOLO region detections", use_column_width=True)
                        if image_bytes:
                            try:
                                base_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                                with st.expander("Detection crop previews", expanded=False):
                                    preview_count = min(4, len(detections))
                                    cols = st.columns(preview_count)
                                    for idx, detection in enumerate(detections[:preview_count]):
                                        bbox = detection.get("box")
                                        if not bbox or len(bbox) != 4:
                                            continue
                                        x1, y1, x2, y2 = bbox
                                        crop = base_image.crop((x1, y1, x2, y2))
                                        cols[idx].image(
                                            crop,
                                            caption=f"{detection.get('label', 'object')} ({detection.get('confidence', 0.0):.2f})",
                                            use_column_width=True,
                                        )
                            except Exception:
                                pass

                if image_artifacts:
                    ocr_summary = {
                        "available": image_artifacts.get("available", False),
                        "avg_confidence": image_artifacts.get("avg_confidence", 0.0),
                        "line_count": image_artifacts.get("line_count", 0),
                        "selected_source": image_artifacts.get("selected_source"),
                        "selected_variant": image_artifacts.get("selected_variant"),
                        "field_count": len(image_artifacts.get("fields") or {}),
                        "error": image_artifacts.get("error"),
                    }
                    st.markdown("**OCR summary**")
                    st.json(ocr_summary)

                    extracted_fields = image_artifacts.get("fields") or {}
                    if extracted_fields:
                        st.markdown("**Extracted fields**")
                        st.json(extracted_fields)

                    variant_candidates = image_artifacts.get("variant_candidates") or []
                    if variant_candidates:
                        with st.expander("OCR variant attempts", expanded=False):
                            st.dataframe(variant_candidates, use_container_width=True)

                    candidate_summary = image_artifacts.get("candidates") or []
                    if candidate_summary:
                        with st.expander("OCR source candidates", expanded=False):
                            st.dataframe(candidate_summary, use_container_width=True)

                    ocr_text = image_artifacts.get("ocr_text")
                    if ocr_text:
                        st.text_area("OCR text preview", value=ocr_text[:2500], height=180)

                    ocr_lines = image_artifacts.get("lines") or []
                    if ocr_lines:
                        with st.expander("OCR line details", expanded=False):
                            st.dataframe(ocr_lines, use_container_width=True)

        with st.expander("4) Retrieval & Reranking", expanded=False):
            retrieval_summary = {
                "expanded_queries": int(latest_expand.get("expanded_queries", 0) or 0),
                "candidate_count": candidate_count,
                "reranked_count": reranked_count,
                "retrieve_duration_ms": latest_retrieve.get("duration_ms"),
                "rerank_duration_ms": latest_rerank.get("duration_ms"),
            }
            st.json(retrieval_summary)

            if latest_retrieve:
                with st.expander("Hybrid retrieve stage metadata", expanded=False):
                    payload = {k: v for k, v in latest_retrieve.items() if k not in {"stage", "status", "duration_ms"}}
                    st.json(payload if payload else {"note": "No extra metadata"})

            if latest_rerank:
                with st.expander("Rerank stage metadata", expanded=False):
                    payload = {k: v for k, v in latest_rerank.items() if k not in {"stage", "status", "duration_ms"}}
                    st.json(payload if payload else {"note": "No extra metadata"})

        with st.expander("5) Generation & Web Fallback", expanded=False):
            if generation_artifacts:
                web_snippets = generation_artifacts.get("web_snippets") or []
                st.json(
                    {
                        "web_used": generation_artifacts.get("web_used", False),
                        "web_snippet_count": len(web_snippets),
                    }
                )

                answer_preview = generation_artifacts.get("answer")
                if answer_preview:
                    st.text_area("LLM answer preview", value=str(answer_preview)[:3000], height=200)

                if web_snippets:
                    st.text_area(
                        "Web snippets preview",
                        value="\n\n".join(
                            f"[{idx}] {snippet.get('title', '')}\n{snippet.get('snippet', '')}\n{snippet.get('url', '')}"
                            for idx, snippet in enumerate(web_snippets, 1)
                        )[:3000],
                        height=200,
                    )
            else:
                st.caption("No generation artifacts captured for this run.")

        if stages:
            with st.expander("6) Full Stage Log (Raw)", expanded=False):
                for item in stages:
                    stage_name = item.get("stage", "unknown")
                    status = item.get("status", "unknown")
                    duration = item.get("duration_ms", 0.0)
                    with st.expander(f"{stage_name} | {status} | {duration} ms", expanded=False):
                        detail_payload = {k: v for k, v in item.items() if k not in {"stage", "status", "duration_ms"}}
                        st.json(detail_payload if detail_payload else {"note": "No extra metadata"})


def _render_answer(
    query: str,
    retriever: HybridRetrieverV2,
    reranker: TwoStageCalibratedReranker,
    generator: GenerationClient,
    image_file=None,
):
    """Execute full RAG pipeline (text/voice/image) and render answer + monitoring."""
    monitor: dict[str, Any] = {"stages": []}
    mode = "image" if image_file is not None else "text_or_voice"
    monitor["mode"] = mode
    image_artifacts = None
    yolo_artifacts = None
    generation_artifacts: dict[str, Any] | None = None
    image_bytes: bytes | None = None

    manual_query = query
    if image_file is not None:
        image_stage_started = time.perf_counter()
        image_bytes = image_file.read()
        cache_hit = False
        cache_key = _image_cache_key(image_bytes)
        image_cache = st.session_state.get("image_pipeline_cache") or {}
        cached_payload = image_cache.get(cache_key)

        if cached_payload:
            cache_hit = True
            yolo_artifacts = deepcopy(cached_payload.get("yolo_artifacts") or {"available": False, "objects": []})
            image_artifacts = deepcopy(cached_payload.get("image_artifacts") or {"available": False, "fields": {}})
            _record_stage(
                monitor,
                "image_cache",
                image_stage_started,
                status="ok",
                cache_hit=True,
                cache_key=cache_key[:12],
            )
        else:
            yolo_artifacts = _run_yolo_detection(image_bytes=image_bytes, monitor=monitor)
            image_artifacts = _run_paddle_ocr(
                image_bytes=image_bytes,
                monitor=monitor,
                detections=(yolo_artifacts or {}).get("objects") or [],
            )
            image_cache[cache_key] = {
                "yolo_artifacts": deepcopy(yolo_artifacts),
                "image_artifacts": deepcopy(image_artifacts),
            }
            cache_limit = 16
            if len(image_cache) > cache_limit:
                oldest_key = next(iter(image_cache.keys()))
                image_cache.pop(oldest_key, None)
            st.session_state["image_pipeline_cache"] = image_cache

        if image_artifacts.get("available"):
            manual_query = _build_image_augmented_query(query, image_artifacts, yolo_artifacts or {})
        monitor["image_cache_hit"] = cache_hit
        _record_stage(
            monitor,
            "image_query_construct",
            image_stage_started,
            status="ok" if image_artifacts.get("available") else "error",
            has_fields=bool((image_artifacts or {}).get("fields")),
            cache_hit=cache_hit,
            reason=(image_artifacts or {}).get("error") if not image_artifacts.get("available") else None,
        )

    reconstruction_started = time.perf_counter()
    reconstruction = generator.reconstruct_query_with_metadata(
        question=query,
        manual_query=manual_query or query,
        ocr_text=str((image_artifacts or {}).get("ocr_text") or "")[:2000],
        fields=(image_artifacts or {}).get("fields") or {},
        objects=(yolo_artifacts or {}).get("objects") or [],
    )
    effective_query = reconstruction.get("reconstructed_query") or manual_query or query
    monitor["query_reconstruction"] = {
        "used_llm": reconstruction.get("used_llm", False),
        "reason": reconstruction.get("reason"),
        "manual_query": (manual_query or "")[:500],
        "reconstructed_query": (effective_query or "")[:500],
        "fallback_query": (reconstruction.get("fallback_query") or "")[:500],
    }
    _record_stage(
        monitor,
        "query_reconstruct",
        reconstruction_started,
        status="ok",
        used_llm=reconstruction.get("used_llm", False),
        reconstruction_reason=reconstruction.get("reason"),
        reconstructed_query=effective_query,
    )

    monitor["effective_query"] = effective_query

    total_started = time.perf_counter()
    with st.spinner("Searching..."):
        try:
            results = _retrieve(effective_query, retriever, reranker, generator, monitor)
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            _record_stage(monitor, "pipeline_error", total_started, status="error", reason=str(e))
            st.session_state["last_monitor"] = monitor
            _render_monitoring_panel(monitor, image_artifacts, yolo_artifacts, generation_artifacts, image_bytes=image_bytes)
            st.stop()

        if not results:
            st.info("No relevant documents found. Try a different query.")
            _record_stage(monitor, "pipeline_result", total_started, status="error", reason="no_results")
            st.session_state["last_monitor"] = monitor
            _render_monitoring_panel(monitor, image_artifacts, yolo_artifacts, generation_artifacts, image_bytes=image_bytes)
            st.stop()

        # Selection of top 5 for generation
        top_results = results[:5]
        generate_started = time.perf_counter()
        extra_input_signals = ""
        if image_artifacts and image_artifacts.get("available"):
            fields = image_artifacts.get("fields") or {}
            ocr_text = str(image_artifacts.get("ocr_text") or "")[:1000]
            extra_input_signals = f"OCR fields: {fields}\nOCR text: {ocr_text}"

        generation_artifacts = generator.generate_with_metadata(
            effective_query,
            top_results,
            extra_context=extra_input_signals,
            allow_web_fallback=WEB_FALLBACK_ENABLED,
        )
        answer = generation_artifacts.get("answer", "")
        _record_stage(
            monitor,
            "generate_answer",
            generate_started,
            status="ok",
            used_top_k=len(top_results),
            web_used=generation_artifacts.get("web_used", False),
            web_snippet_count=len(generation_artifacts.get("web_snippets", []) or []),
            answer_chars=len(answer or ""),
        )

    answer_html = build_answer_html(answer, top_results)
    answer_lines = answer.count("\n") + 1
    estimated_height = 350 + (answer_lines * 22) + (len(top_results) * 55)
    estimated_height = min(max(estimated_height, 450), 1800)
    components.html(answer_html, height=estimated_height, scrolling=True)

    monitor["total_latency_ms"] = round((time.perf_counter() - total_started) * 1000.0, 2)
    st.session_state["last_monitor"] = monitor
    _render_monitoring_panel(
        monitor,
        image_artifacts,
        yolo_artifacts,
        generation_artifacts,
        image_bytes=image_bytes,
    )


def _render_example_queries():
    """Display clickable example query buttons."""
    st.markdown("")
    st.markdown("##### Try asking")
    cols = st.columns(2)
    for idx, example in enumerate(EXAMPLE_QUERIES):
        with cols[idx % 2]:
            if st.button(example, key=f"ex_{idx}", use_container_width=True):
                st.query_params.update({"q": example})
                st.rerun()


def main():
    st.title("Retrieval Augmented Generation for Climate Challenges")
    st.caption("Search across your document collection")

    retriever = get_retriever()
    reranker = get_reranker_model()
    generator = get_generator()
    whisper_model = load_whisper_model()

    _init_voice_state()
    query, image_file, should_search = _render_voice_recorder(whisper_model)

    if st.session_state["just_transcribed"]:
        st.session_state["just_transcribed"] = False

    if should_search and (query or image_file is not None):
        _render_answer(query, retriever, reranker, generator, image_file=image_file)
    else:
        _render_example_queries()


if __name__ == "__main__":
    main()
