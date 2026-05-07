import base64
import json
import os

import anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# MODEL = "claude-opus-4-6"
MODEL = "gemini-3.1-flash-lite-preview"
MAX_TOKENS = 256
GEMINI_MAX_OUTPUT_TOKENS = 2048
PROVIDER = "anthropic"

GEMINI_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "site_class": {
                        "type": "string",
                        "enum": [
                            "data_center",
                            "under_construction",
                            "industrial_logistics_warehouse",
                            "industrial_manufacturing",
                            "industrial_semiconductor_fab",
                            "power_generation_facility",
                            "ambiguous_large_industrial",
                            "no_industrial_site_present",
                        ],
                    },
                    "construction_stage": {
                        "type": "string",
                        "enum": [
                            "undisturbed",
                            "land_clearing",
                            "earthworks",
                            "foundations",
                            "structural_shell",
                            "roof_complete",
                            "operational",
                            "expansion",
                        ],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "cooling_signature_visible": {"type": "boolean"},
                    "cooling_type": {
                        "type": "string",
                        "enum": [
                            "air_cooled",
                            "evaporative_towers",
                            "cooling_pond",
                            "not_visible",
                        ],
                    },
                    "substation_adjacent": {"type": "boolean"},
                    "backup_generators_visible": {"type": "boolean"},
                    "roof_bright_membrane": {"type": "boolean"},
                    "bare_soil_present": {"type": "boolean"},
                    "water_feature_present": {"type": "boolean"},
                    "vegetation_buffer_present": {"type": "boolean"},
                    "dock_doors_or_truck_courts": {"type": "boolean"},
                    "reasoning": {"type": "string"},
                },
                "required": [
                    "bbox",
                    "site_class",
                    "construction_stage",
                    "confidence",
                    "cooling_signature_visible",
                    "cooling_type",
                    "substation_adjacent",
                    "backup_generators_visible",
                    "roof_bright_membrane",
                    "bare_soil_present",
                    "water_feature_present",
                    "vegetation_buffer_present",
                    "dock_doors_or_truck_courts",
                    "reasoning",
                ],
            },
        },
        "tile_context": {
            "type": "object",
            "properties": {
                "residential_proximity": {
                    "type": "string",
                    "enum": ["adjacent", "within_1km", "distant", "none_visible"],
                },
                "residential_density": {
                    "type": "string",
                    "enum": ["dense_urban", "suburban", "rural_scattered", "none"],
                },
                "agricultural_land_adjacent": {"type": "boolean"},
                "arid_landscape": {"type": "boolean"},
                "shared_water_body_nearby": {"type": "boolean"},
                "visible_water_body_type": {
                    "type": "string",
                    "enum": ["reservoir", "river", "lake", "canal", "none"],
                },
                "vegetation_stress_surrounding": {"type": "boolean"},
                "other_industrial_cluster": {"type": "boolean"},
                "image_quality_limited": {"type": "boolean"},
            },
            "required": [
                "residential_proximity",
                "residential_density",
                "agricultural_land_adjacent",
                "arid_landscape",
                "shared_water_body_nearby",
                "visible_water_body_type",
                "vegetation_stress_surrounding",
                "other_industrial_cluster",
                "image_quality_limited",
            ],
        },
    },
    "required": ["detections", "tile_context"],
}


class AnnotationParseError(ValueError):
    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


SYSTEM_PROMPT = """\
You are a remote sensing analyst specialising in data center and large-scale \
compute infrastructure detection and characterisation.
The target locations in this dataset are known data-center candidates. \
That means if you detect a facility, the default positive class should be \
data_center unless the evidence clearly shows a different industrial use. \
Do not randomly assign ambiguous_large_industrial just because the \
evidence is weak or incomplete. Use it only when a large industrial site is \
visible but the subtype cannot be determined reliably.
You will be given up to four images of the same land tile:
  1. RGB composite (bands B4-B3-B2): natural colour, useful for building \
footprints, roads, construction activity, and land cover context.
  2. SWIR composite (bands B12-B8-B4): highlights roof materials, bare soil \
vs paved surfaces, and industrial structure signatures. Bright white/cyan \
indicates high-albedo membrane roofing typical of data centers. Orange/red \
indicates dry vegetation or exposed soil. Magenta/pink indicates bare \
earth or gravel staging areas.
  3. Index composite (NDVI-MNDWI-NDBI): three normalised indices stacked \
as RGB. Red channel = NDVI (vegetation health: bright red = dense healthy \
vegetation, dark = bare/built). Green channel = MNDWI (water presence: \
bright green = open water, dark = dry land). Blue channel = NDBI \
(built-up intensity: bright blue = dense impervious surface, dark = \
natural land cover).
  4. Mapbox high-resolution imagery (~50cm, when available): provides fine \
detail for counting cooling units, identifying transformer yards, \
confirming dock-door absence, and reading site layout. May be 6-24 \
months older than the Sentinel-2 tiles.

Identify ALL large industrial facilities in the tile. For each one, provide \
a bounding box and per-site classification. Then assess tile-level context \
once. Return ONLY a valid JSON object — no markdown, no explanation outside \
the JSON — with exactly this structure:

{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "site_class": "data_center",
      "construction_stage": "operational",
      "confidence": "high",
      "cooling_signature_visible": true | false,
      "cooling_type": "air_cooled",
      "substation_adjacent": true | false,
      "backup_generators_visible": true | false,
      "roof_bright_membrane": true | false,
      "bare_soil_present": true | false,
      "water_feature_present": true | false,
      "vegetation_buffer_present": true | false,
      "dock_doors_or_truck_courts": true | false,
      "reasoning": ""
    }
  ],
  "tile_context": {
    "residential_proximity": "adjacent",
    "residential_density": "dense_urban",
    "agricultural_land_adjacent": true | false,
    "arid_landscape": true | false,
    "shared_water_body_nearby": true | false,
    "visible_water_body_type": "reservoir",
    "vegetation_stress_surrounding": true | false,
    "other_industrial_cluster": true | false,
    "image_quality_limited": true | false
  }
}

If no industrial facility is present, return an empty detections array \
with tile_context still populated, or use a single detection with \
site_class="no_industrial_site_present" if that is cleaner for the image. \
It is acceptable for some tiles to have no detectable facility at all.

Bounding box format:
- [x1, y1, x2, y2] as normalised coordinates where (0,0) is top-left \
and (1,1) is bottom-right of the tile.
- Always normalize bbox coordinates to the 0..1 range before returning \
them. Never return pixel coordinates or values based on 512/1000/etc. \
If the facility touches the tile edge, clip the normalized box to 0 or 1.
- The box should tightly enclose the full campus footprint including \
parking, staging areas, and adjacent substation if clearly part of \
the same facility. Do not include surrounding roads or unrelated parcels.
- If a facility extends beyond the tile edge, clip the box to the tile \
boundary and note this in reasoning.

Per-detection field definitions:
- site_class: classification for this specific facility:
    - data_center: operational or near-operational data center campus with \
large rectangular buildings, visible cooling or power infrastructure, no \
truck loading docks, low parking-to-building ratio
    - under_construction: active construction of a data-center candidate \
with bare soil, geometric footprints, partial structures, staging areas, or \
expansion work visible
    - industrial_logistics_warehouse: logistics, distribution, or storage \
facility with truck courts, loading docks, trailer parking, and warehouse \
geometry
    - industrial_manufacturing: manufacturing plant or industrial process \
facility with chimneys, tanks, process yards, or non-warehouse production \
geometry
    - industrial_semiconductor_fab: semiconductor fab or related cleanroom \
facility with large controlled buildings and support infrastructure, but \
not a generic warehouse
    - power_generation_facility: power plant, substation-heavy generation \
site, or grid-scale energy infrastructure
    - ambiguous_large_industrial: large industrial site visible, but the \
exact class is not confidently separable
    - no_industrial_site_present: no industrial facility present in the tile
- construction_stage: current phase of development visible at this site:
    - undisturbed: natural land cover, no construction activity
    - land_clearing: vegetation removed, bare soil exposed in geometric \
pattern, NDVI sharply lower than surrounding context
    - earthworks: graded terrain, access roads cut, retention features \
visible in MNDWI, no building footprints yet
    - foundations: concrete pads or footings visible as bright rectangular \
patches distinct from bare soil in SWIR, higher B4 reflectance \
than surrounding earth
    - structural_shell: building frame up but roof incomplete — mixed \
spectral signature over footprint, shadows from vertical structure \
visible in RGB
    - roof_complete: bright uniform roof membrane or metal visible — \
high B4 reflectance, distinctive B11/B12 SWIR ratio, building \
footprint sharply defined
    - operational: completed buildings with stable spectral signature, \
paved access, cooling infrastructure active, landscaping visible \
at perimeter in NDVI
    - expansion: operational buildings coexist with adjacent active \
construction in a new quadrant of the same campus
- confidence: how certain the classification is:
    - high: morphology, spectral signature, and infrastructure all \
consistent with the assigned class
    - medium: morphology matches but key confirming features (cooling, \
substation) are ambiguous or at resolution limit
    - low: site is weakly visible or partially obscured, but still more \
consistent with a data-center candidate than a generic industrial site
- cooling_signature_visible: any cooling infrastructure identifiable — \
rooftop units, cooling plant area, evaporative plume, or pond. In \
Mapbox, look for rows of fan units or cellular tower structures. In \
Sentinel-2, look for linear brightness patterns along building edges \
in SWIR or localised MNDWI signal from evaporative moisture.
- cooling_type: best estimate of primary cooling method:
    - air_cooled: rooftop dry coolers or fan arrays along building edge, \
no water features associated with cooling
    - evaporative_towers: distinctive circular or cellular tower pattern \
in Mapbox, sometimes with visible moisture plume in Sentinel-2 B9
    - cooling_pond: adjacent water body with geometric shape and \
proximity to facility inconsistent with natural hydrology, \
strong MNDWI signal
    - not_visible: cooling infrastructure not resolvable or not present
- substation_adjacent: high-voltage substation or transformer yard \
within ~500m of this facility — identifiable in Mapbox as a fenced \
area with regular rows of transformer units and buswork, and in \
Sentinel-2 as a distinct rectangular cleared area with high NDBI \
and transmission line corridor.
- backup_generators_visible: generator yard or row of container-sized \
units adjacent to buildings — typically a linear arrangement of \
uniform rectangular objects on a concrete pad. Primarily identifiable \
in Mapbox; in Sentinel-2, may appear as a narrow high-reflectance \
strip parallel to a building.
- roof_bright_membrane: at least one building has a bright, high-albedo \
roof visible as elevated reflectance in B4 with characteristic \
B11/B12 SWIR ratio — typical of TPO or PVC membrane roofing common \
on data centers. In Mapbox, appears as uniform light grey or white \
roof surface.
- bare_soil_present: exposed earth within this facility boundary, \
indicating active construction, staging, or unpaved areas. \
Identifiable by low NDVI and characteristic SWIR soil signature \
(orange/red in the SWIR composite).
- water_feature_present: any water body within or immediately adjacent \
to this facility — cooling pond, retention basin, or stormwater \
feature. Identifiable via MNDWI green channel signal.
- vegetation_buffer_present: maintained vegetation strip along this \
facility perimeter — common in data center campuses for visual \
screening and zoning compliance. Visible as elevated NDVI (red \
in index composite) bordering the high-NDBI site footprint.
- dock_doors_or_truck_courts: loading bays, truck backing courts, or \
dense trailer parking visible at this facility — strong negative \
indicator for data center, strong positive for logistics warehouse. \
In Mapbox, look for regular indentations along building edge with \
adjacent paved apron and trailer rectangles. In Sentinel-2, appears \
as a bright strip pattern along the long axis of a building.
- reasoning: one to three sentences grounding this detection in \
specific visual and spectral features. Cite which image (RGB, \
SWIR, index, or Mapbox) provided each key piece of evidence. \
Example: "SWIR shows high-albedo roof with B11/B12 ratio \
consistent with TPO membrane. Mapbox confirms evaporative cooling \
units along south facade, no dock doors. Substation visible 300m \
south in RGB with transmission corridor running west."

Tile-level context field definitions (assess once for the whole tile):
- residential_proximity: nearest visible residential area relative to \
any detected facility:
    - adjacent: residential rooftops or subdivision patterns directly \
border a facility perimeter with no buffer
    - within_1km: residential visible in tile but separated by buffer, \
road, or open land
    - distant: residential barely visible or at tile edge, estimated \
over 1km from nearest facility
    - none_visible: no residential land use identifiable in the tile
- residential_density: character of the nearest residential area:
    - dense_urban: continuous urban fabric, row housing, apartment blocks
    - suburban: detached houses with yards, cul-de-sac street patterns
    - rural_scattered: isolated farmsteads or small clusters
    - none: no residential structures identifiable
- agricultural_land_adjacent: active cropland (regular field geometry, \
seasonal NDVI consistent with cultivation) directly borders or is \
within 500m of any detected facility.
- arid_landscape: surrounding context is predominantly dry — low NDVI \
baseline across non-irrigated areas, bare soil or desert scrub \
dominant in the index composite.
- shared_water_body_nearby: a river, reservoir, lake, or canal within \
the tile that likely serves multiple users. Identified by MNDWI \
signal and non-geometric shape distinguishing it from purpose-built \
cooling infrastructure.
- visible_water_body_type: classification of the shared water feature:
    - reservoir: large impounded body with dam structure visible
    - river: linear water feature with natural sinuosity
    - lake: natural-geometry standing water body
    - canal: linear artificial water channel, often irrigation-related
    - none: no shared water body in tile
- vegetation_stress_surrounding: NDVI in the ~500m buffer around any \
detected facility is notably lower than the broader tile context, \
suggesting localised environmental impact.
- other_industrial_cluster: multiple large industrial facilities \
detected in the same tile, indicating an industrial zone rather \
than isolated development.
- image_quality_limited: cloud, haze, snow, shadow, or no-data \
obscures a significant portion of the tile area in the \
Sentinel-2 composites.
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
                temperature=0.0,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
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
