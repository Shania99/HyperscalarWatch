import base64
import json
import os

import anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types

from datacenter_watch.compact_schema import RESPONSE_SCHEMA

load_dotenv()

# MODEL = "claude-opus-4-6"
MODEL = "gemini-3.1-flash-lite-preview"
MAX_TOKENS = 256
GEMINI_MAX_OUTPUT_TOKENS = 2048
PROVIDER = "anthropic"

GEMINI_RESPONSE_SCHEMA: dict[str, object] = RESPONSE_SCHEMA


class AnnotationParseError(ValueError):
    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


SYSTEM_PROMPT = f"""\
You are a remote sensing analyst specialising in data-center and industrial-site \
detection from satellite imagery.

You will be given up to four images of the same tile:
1. RGB composite (B4-B3-B2): buildings, roads, construction patterns, and land cover.
2. SWIR composite (B12-B8-B4): roof materials, bare soil, staging areas, and industrial signatures.
3. Index composite (NDVI-MNDWI-NDBI), when available: vegetation, water, and built-up intensity.
4. Mapbox high-resolution imagery, when available: finer site-layout detail, but it may be older.

Return ONLY a valid JSON object with this structure:
{{
  "detections": [
    {{
      "bbox": [x1, y1, x2, y2] | null,
      "site_class": "data_center" | "industrial_site" | "no_site",
      "construction_stage": "undisturbed" | "active_construction" | "operational",
      "roof_bright_membrane": true | false,
      "bare_soil_present": true | false,
      "reasoning": ""
    }}
  ],
  "tile_context": {{
    "image_quality_limited": true | false
  }}
}}

Site class guidance:
- data_center: strong evidence for a data-center campus, such as large rectangular buildings, \
campus-like layout, limited dock-door presence, power-heavy morphology, or distinctive bright roofs.
- industrial_site: another industrial/logistics/manufacturing/power site, or a large construction site \
that cannot yet be confidently identified as a data center.
- no_site: explicit negative sentinel. Prefer an empty detections array when no relevant site is visible. \
If you emit no_site, set bbox to null.

Construction stage guidance:
- undisturbed: site footprint is present or suspected, but no visible construction activity is underway.
- active_construction: any visible clearing, grading, foundations, partial structures, roof-complete shell, \
or expansion work.
- operational: completed or mostly completed active site.

Field guidance:
- bbox: normalized coordinates in the 0..1 range, clipped to the tile, or null only for no_site.
- roof_bright_membrane: true when at least one building shows an obvious bright high-albedo roof, especially \
in RGB and SWIR.
- bare_soil_present: true when exposed soil is visible within the facility footprint or immediate staging area.
- reasoning: one or two short sentences grounded in visible evidence from the provided images.
- image_quality_limited: true when cloud, haze, shadow, snow, or missing data materially limits interpretation.

Important rules:
- Detect all relevant sites in the tile.
- Do not invent facilities that are not visible.
- Do not default to data_center when the evidence only supports a generic industrial construction site.
- Return JSON only. No markdown and no explanatory text outside the JSON.
"""

USER_TEXT = "Return the data center detection JSON for this tile."


def _encode(image_bytes: bytes) -> str:
    return base64.standard_b64encode(image_bytes).decode()


def _image_content(image_bytes: bytes) -> dict[str, object]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _encode(image_bytes),
        },
    }


def build_user_text(has_index: bool, has_mapbox: bool) -> str:
    lines = [
        "Image 1 is the RGB composite.",
        "Image 2 is the SWIR composite.",
    ]
    if has_index:
        lines.append("Image 3 is the index composite (NDVI-MNDWI-NDBI).")
    if has_mapbox:
        image_num = 4 if has_index else 3
        lines.append(f"Image {image_num} is the Mapbox high-resolution imagery.")
    lines.append(USER_TEXT)
    return " ".join(lines)


def annotate(
    rgb_bytes: bytes,
    swir_bytes: bytes,
    *,
    index_bytes: bytes | None = None,
    mapbox_bytes: bytes | None = None,
    model: str = MODEL,
    provider: str = "auto",
) -> dict[str, object]:
    """Call the selected multimodal model and return the parsed JSON.

    Args:
        rgb_bytes: Raw PNG bytes of the RGB composite.
        swir_bytes: Raw PNG bytes of the SWIR composite.
        index_bytes: Optional raw PNG bytes of the NDVI-MNDWI-NDBI composite.
        mapbox_bytes: Optional raw PNG bytes of the high-resolution Mapbox image.
        model: Model ID to use for annotation. Anthropic and Gemini are supported.
        provider: "anthropic", "gemini", or "auto" to infer from model name.

    Returns:
        Parsed JSON dict matching the detection schema in SYSTEM_PROMPT.

    Raises:
        ValueError: if the model response cannot be parsed as JSON.
    """
    raw = annotate_raw(
        rgb_bytes,
        swir_bytes,
        index_bytes=index_bytes,
        mapbox_bytes=mapbox_bytes,
        model=model,
        provider=provider,
    )
    return _parse_json_response(raw)


def annotate_raw(
    rgb_bytes: bytes,
    swir_bytes: bytes,
    *,
    index_bytes: bytes | None = None,
    mapbox_bytes: bytes | None = None,
    model: str = MODEL,
    provider: str = "auto",
) -> str:
    """Call the selected multimodal model and return the raw text response."""
    user_content = _build_user_content(
        rgb_bytes,
        swir_bytes,
        index_bytes=index_bytes,
        mapbox_bytes=mapbox_bytes,
    )
    resolved_provider = _resolve_provider(model, provider)
    if resolved_provider == "gemini":
        return _annotate_gemini(model, user_content)
    return _annotate_anthropic(model, user_content)


def _build_user_content(
    rgb_bytes: bytes,
    swir_bytes: bytes,
    *,
    index_bytes: bytes | None = None,
    mapbox_bytes: bytes | None = None,
) -> list[dict[str, object]]:
    user_content: list[dict[str, object]] = [
        _image_content(rgb_bytes),
        _image_content(swir_bytes),
    ]
    if index_bytes is not None:
        user_content.append(_image_content(index_bytes))
    if mapbox_bytes is not None:
        user_content.append(_image_content(mapbox_bytes))
    user_content.append(
        {
            "type": "text",
            "text": build_user_text(
                has_index=index_bytes is not None,
                has_mapbox=mapbox_bytes is not None,
            ),
        }
    )
    return user_content


def _annotate_anthropic(model: str, user_content: list[dict[str, object]]) -> str:
    client = anthropic.Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": user_content,
            }
        ],
    )
    return message.content[0].text.strip()


def _annotate_gemini(model: str, user_content: list[dict[str, object]]) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini models.")

    client = genai.Client(api_key=api_key)
    contents = [_to_gemini_part(part) for part in user_content]
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_json_schema=GEMINI_RESPONSE_SCHEMA,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            ),
        )
    except Exception as exc:
        raise ValueError(f"Gemini request failed: {exc}") from exc

    raw = (response.text or "").strip()
    if raw:
        return raw

    if not response.candidates:
        raise ValueError("Gemini returned no candidates.")

    texts: list[str] = []
    for candidate in response.candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None)
        if not parts:
            continue
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text.strip())
    if not texts:
        raise ValueError("Gemini returned no text response.")
    return "\n".join(texts).strip()


def _to_gemini_part(part: dict[str, object]) -> types.Part | str:
    if part.get("type") == "text":
        return str(part["text"])
    source = part.get("source")
    if isinstance(source, dict):
        return types.Part.from_bytes(
            data=base64.standard_b64decode(str(source["data"])),
            mime_type=str(source["media_type"]),
        )
    raise ValueError(f"Unsupported content part for Gemini: {part}")


def _is_gemini_model(model: str) -> bool:
    return model.startswith("gemini")


def _resolve_provider(model: str, provider: str) -> str:
    if provider == "auto":
        return "gemini" if _is_gemini_model(model) else "anthropic"
    if provider not in {"anthropic", "gemini"}:
        raise ValueError(f"Unsupported provider: {provider}")
    return provider


def _parse_json_response(raw: str) -> dict[str, object]:
    # Strip markdown code fences if the model wraps the JSON.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]  # drop the opening ```json line
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise AnnotationParseError(
            f"Model returned non-JSON response:\n{raw}", raw
        ) from exc
