"""Evaluation backends and metrics for structured tile annotations."""

import base64
import http.client
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from openai import OpenAI

from datacenter_watch.annotator import (
    GEMINI_RESPONSE_SCHEMA,
    MODEL as ANTHROPIC_MODEL,
    SYSTEM_PROMPT,
    annotate,
    build_user_text,
)

EVAL_FIELDS: list[str] = [
    "detections",
    "tile_context",
]

DETECTION_IOU_THRESHOLD = 0.5
DETECTION_EVAL_FIELDS: list[str] = [
    "site_class",
    "construction_stage",
    "cooling_signature_visible",
    "cooling_type",
    "substation_adjacent",
    "backup_generators_visible",
    "roof_bright_membrane",
    "bare_soil_present",
    "water_feature_present",
    "vegetation_buffer_present",
    "dock_doors_or_truck_courts",
]
TILE_CONTEXT_EVAL_FIELDS: list[str] = [
    "residential_proximity",
    "residential_density",
    "agricultural_land_adjacent",
    "arid_landscape",
    "shared_water_body_nearby",
    "visible_water_body_type",
    "vegetation_stress_surrounding",
    "other_industrial_cluster",
    "image_quality_limited",
]

_RESPONSE_SCHEMA: dict[str, object] = GEMINI_RESPONSE_SCHEMA


def _bbox_iou(a: object, b: object) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != 4 or len(b) != 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = (float(x) for x in a)
        bx1, by1, bx2, by2 = (float(x) for x in b)
    except (TypeError, ValueError):
        return 0.0
    values = (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
    if any(x < 0.0 or x > 1.0 for x in values):
        return 0.0
    ax1, ax2 = sorted((ax1, ax2))
    ay1, ay2 = sorted((ay1, ay2))
    bx1, bx2 = sorted((bx1, bx2))
    by1, by2 = sorted((by1, by2))
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def _match_detections(
    prediction: object,
    ground_truth: object,
) -> tuple[bool, list[tuple[int, int, float]]]:
    if not isinstance(prediction, list) or not isinstance(ground_truth, list):
        return False, []

    matched_pairs: list[tuple[int, int, float]] = []
    used_pred: set[int] = set()

    for gt_idx, gt_det in enumerate(ground_truth):
        if not isinstance(gt_det, dict):
            return False, []
        best_pred_idx: int | None = None
        best_iou = 0.0
        for pred_idx, pred_det in enumerate(prediction):
            if pred_idx in used_pred or not isinstance(pred_det, dict):
                continue
            if pred_det.get("site_class") != gt_det.get("site_class"):
                continue
            iou = _bbox_iou(pred_det.get("bbox"), gt_det.get("bbox"))
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        if best_pred_idx is None or best_iou < DETECTION_IOU_THRESHOLD:
            return False, []
        pred_det = prediction[best_pred_idx]
        assert isinstance(pred_det, dict)
        for field in DETECTION_EVAL_FIELDS:
            if pred_det.get(field) != gt_det.get(field):
                return False, []
        used_pred.add(best_pred_idx)
        matched_pairs.append((gt_idx, best_pred_idx, best_iou))

    if len(used_pred) != len(prediction):
        return False, []
    return True, matched_pairs


def _match_tile_context(prediction: object, ground_truth: object) -> bool:
    if not isinstance(prediction, dict) or not isinstance(ground_truth, dict):
        return False
    return all(
        prediction.get(field) == ground_truth.get(field)
        for field in TILE_CONTEXT_EVAL_FIELDS
    )


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class PredictFn(Protocol):
    def __call__(
        self,
        rgb_bytes: bytes,
        swir_bytes: bytes,
        *,
        index_bytes: bytes | None = None,
        mapbox_bytes: bytes | None = None,
    ) -> dict[str, object]: ...


def anthropic_backend() -> PredictFn:
    """Return a predict function that calls claude-opus-4-6 via the Anthropic SDK."""
    return annotate


def gemini_backend(model: str = "gemini-3-flash-preview") -> PredictFn:
    """Return a predict function that calls Gemini with the annotation schema."""
    def predict(
        rgb_bytes: bytes,
        swir_bytes: bytes,
        *,
        index_bytes: bytes | None = None,
        mapbox_bytes: bytes | None = None,
    ) -> dict[str, object]:
        return annotate(
            rgb_bytes,
            swir_bytes,
            index_bytes=index_bytes,
            mapbox_bytes=mapbox_bytes,
            model=model,
            provider="gemini",
        )

    return predict


def transformers_backend(model_path: str) -> PredictFn:
    """Return a predict function that loads a HuggingFace safetensors checkpoint.

    Used to evaluate fine-tuned checkpoints that have not been converted to GGUF.
    Requires: transformers, torch, Pillow (already in the project dependencies).
    """
    import io

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText  # type: ignore[import-untyped]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_path = Path(model_path)
    # Fine-tuning never modifies the processor. Load it from HF if the path is
    # local — newer transformers (5.x) rejects absolute paths in AutoProcessor.
    processor_source = "LiquidAI/LFM2.5-VL-450M" if local_path.is_dir() else model_path
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(local_path) if local_path.is_dir() else model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=local_path.is_dir(),
    ).to(device)
    model.eval()

    def predict(
        rgb_bytes: bytes,
        swir_bytes: bytes,
        *,
        index_bytes: bytes | None = None,
        mapbox_bytes: bytes | None = None,
    ) -> dict[str, object]:
        rgb_image  = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        swir_image = Image.open(io.BytesIO(swir_bytes)).convert("RGB")
        images = [rgb_image, swir_image]
        if index_bytes is not None:
            images.append(Image.open(io.BytesIO(index_bytes)).convert("RGB"))
        if mapbox_bytes is not None:
            images.append(Image.open(io.BytesIO(mapbox_bytes)).convert("RGB"))

        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image"} for _ in images]
                    + [{
                        "type": "text",
                        "text": (
                            f"{SYSTEM_PROMPT.strip()}\n\n"
                            f"{build_user_text(index_bytes is not None, mapbox_bytes is not None)}"
                        ),
                    }]
                ),
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=text,
            images=images,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        raw = processor.decode(generated, skip_special_tokens=True).strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0].strip()
        return json.loads(raw)  # type: ignore[no-any-return]

    return predict


def llama_backend(model: str, port: int = 8080) -> PredictFn:
    """Return a predict function that calls a local llama-server via the OpenAI API."""
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed")

    def predict(
        rgb_bytes: bytes,
        swir_bytes: bytes,
        *,
        index_bytes: bytes | None = None,
        mapbox_bytes: bytes | None = None,
    ) -> dict[str, object]:
        def _data_url(image_bytes: bytes) -> str:
            return "data:image/png;base64," + base64.standard_b64encode(image_bytes).decode()

        content: list[dict[str, object]] = [
            {"type": "image_url", "image_url": {"url": _data_url(rgb_bytes)}},
            {"type": "image_url", "image_url": {"url": _data_url(swir_bytes)}},
        ]
        if index_bytes is not None:
            content.append({"type": "image_url", "image_url": {"url": _data_url(index_bytes)}})
        if mapbox_bytes is not None:
            content.append({"type": "image_url", "image_url": {"url": _data_url(mapbox_bytes)}})
        content.append({
            "type": "text",
            "text": build_user_text(index_bytes is not None, mapbox_bytes is not None),
        })

        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "TileAnnotation", "schema": _RESPONSE_SCHEMA},
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
        content = response.choices[0].message.content or ""
        return json.loads(content)  # type: ignore[no-any-return]

    return predict


# ---------------------------------------------------------------------------
# llama-server lifecycle (mirrors invoice-parser pattern)
# ---------------------------------------------------------------------------

def start_llama_server(
    model: str,
    quant: str | None = None,
    port: int = 8080,
    verbose: bool = False,
    mmproj: str | None = None,
) -> subprocess.Popen[bytes]:
    local_path = Path(model)
    if local_path.is_file():
        cmd = ["llama-server", "-m", str(local_path), "--jinja", "--port", str(port)]
    else:
        if "/" not in model and model.startswith("LFM"):
            model = f"LiquidAI/{model}"
        hf_repo = f"{model}:{quant}" if quant else model
        cmd = ["llama-server", "-hf", hf_repo, "--jinja", "--port", str(port)]
    if mmproj:
        cmd += ["--mmproj", mmproj]
    kwargs: dict[str, object] = {}
    if not verbose:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return subprocess.Popen(cmd, **kwargs)  # type: ignore[call-overload]


def wait_for_server(port: int = 8080, timeout: int = 120) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, http.client.BadStatusLine):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become healthy within {timeout}s")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


# ---------------------------------------------------------------------------
# Per-sample result
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    id: str
    valid_json: bool
    fields_present: bool
    field_matches: dict[str, bool]  # field -> match against ground truth
    latency_s: float = 0.0
    prediction: dict[str, object] | None = None
    ground_truth: dict[str, object] | None = None

    @property
    def all_fields_match(self) -> bool:
        return all(self.field_matches.values())


def evaluate_sample(
    location_id: str,
    rgb_bytes: bytes,
    swir_bytes: bytes,
    index_bytes: bytes | None,
    mapbox_bytes: bytes | None,
    ground_truth: dict[str, object],
    predict: PredictFn,
) -> SampleResult:
    t0 = perf_counter()
    try:
        prediction = predict(
            rgb_bytes,
            swir_bytes,
            index_bytes=index_bytes,
            mapbox_bytes=mapbox_bytes,
        )
        valid_json = True
    except Exception:
        return SampleResult(
            id=location_id,
            valid_json=False,
            fields_present=False,
            field_matches={f: False for f in EVAL_FIELDS},
            latency_s=perf_counter() - t0,
            prediction=None,
            ground_truth=ground_truth,
        )

    latency_s = perf_counter() - t0
    fields_present = all(f in prediction for f in EVAL_FIELDS)
    detections_match, _matched_pairs = _match_detections(
        prediction.get("detections"),
        ground_truth.get("detections"),
    )
    tile_context_match = _match_tile_context(
        prediction.get("tile_context"),
        ground_truth.get("tile_context"),
    )
    field_matches = {
        "detections": detections_match,
        "tile_context": tile_context_match,
    }
    return SampleResult(
        id=location_id,
        valid_json=valid_json,
        fields_present=fields_present,
        field_matches=field_matches,
        latency_s=latency_s,
        prediction=prediction,
        ground_truth=ground_truth,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalSummary:
    results: list[SampleResult]

    def valid_json_accuracy(self) -> float:
        return sum(r.valid_json for r in self.results) / len(self.results) if self.results else 0.0

    def fields_present_accuracy(self) -> float:
        return sum(r.fields_present for r in self.results) / len(self.results) if self.results else 0.0

    def field_accuracy(self, field: str) -> float:
        matches = [r.field_matches[field] for r in self.results if r.fields_present]
        return sum(matches) / len(matches) if matches else 0.0

    def overall_accuracy(self) -> float:
        all_matches = [
            match
            for r in self.results
            if r.fields_present
            for match in r.field_matches.values()
        ]
        return sum(all_matches) / len(all_matches) if all_matches else 0.0

    def avg_latency_s(self) -> float:
        return sum(r.latency_s for r in self.results) / len(self.results) if self.results else 0.0


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _tick(value: bool) -> str:
    return "✓" if value else "✗"


def render_report(
    summary: EvalSummary,
    dataset: str,
    backend: str,
    model: str,
    eval_run_id: str,
) -> str:
    lines: list[str] = []

    lines.append(f"# Tile Annotation Eval — {eval_run_id}")
    lines.append("")
    lines.append(f"**Dataset:** {dataset}  ")
    lines.append(f"**Backend:** {backend}  ")
    lines.append(f"**Model:** {model}")
    lines.append("")

    # Accuracy summary first
    lines.append("## Accuracy summary")
    lines.append("")
    lines.append("| field | accuracy |")
    lines.append("|---|---|")
    lines.append(f"| valid_json | {summary.valid_json_accuracy():.2f} |")
    lines.append(f"| fields_present | {summary.fields_present_accuracy():.2f} |")
    for field in EVAL_FIELDS:
        acc = summary.field_accuracy(field)
        lines.append(f"| {field} | {acc:.2f} |")
    lines.append(f"| **overall** | **{summary.overall_accuracy():.2f}** |")
    lines.append(f"| **avg latency (s)** | **{summary.avg_latency_s():.2f}** |")
    lines.append("")

    # Per-sample table
    lines.append("## Per-sample results")
    lines.append("")
    header = (
        "| id | latency (s) | valid_json | fields_present | detections | tile_context |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|")
    for r in summary.results:
        fm = r.field_matches
        lines.append(
            f"| {r.id}"
            f" | {r.latency_s:.2f}"
            f" | {_tick(r.valid_json)}"
            f" | {_tick(r.fields_present)}"
            f" | {_tick(fm.get('detections', False))}"
            f" | {_tick(fm.get('tile_context', False))}"
            " |"
        )
    lines.append("")

    return "\n".join(lines)


def model_name(backend: str, llama_model: str, quant: str = "") -> str:
    if backend == "anthropic":
        return ANTHROPIC_MODEL
    if backend == "hf":
        return llama_model
    return f"{llama_model}:{quant}" if quant else llama_model


# ---------------------------------------------------------------------------
# Structured result persistence
# ---------------------------------------------------------------------------

def save_results(
    eval_dir: Path,
    summary: EvalSummary,
    dataset: str,
    backend: str,
    model: str,
    split: str,
    eval_run_id: str,
) -> None:
    """Write results.json and meta.json into eval_dir."""
    meta = {
        "eval_run_id": eval_run_id,
        "dataset": dataset,
        "backend": backend,
        "model": model,
        "split": split,
    }
    (eval_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    records = [
        {
            "id": r.id,
            "valid_json": r.valid_json,
            "fields_present": r.fields_present,
            "field_matches": r.field_matches,
            "latency_s": r.latency_s,
            "prediction": r.prediction,
            "ground_truth": r.ground_truth,
        }
        for r in summary.results
    ]
    (eval_dir / "results.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
