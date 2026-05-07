"""Direct Gemini API smoke test for one saved sample directory.

Usage:
    uv run scripts/test_gemini_api.py \
      --sample-dir data/20260506_115616/train/attica_greece/s00_t00 \
      --model gemini-3-flash-preview
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from datacenter_watch.annotator import (
    GEMINI_RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    build_user_text,
    GEMINI_MAX_OUTPUT_TOKENS,
)

load_dotenv()


def _image_part(path: Path) -> types.Part:
    return types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Gemini directly on one saved sample.")
    parser.add_argument(
        "--sample-dir",
        required=True,
        metavar="DIR",
        help="Path to one sample directory containing rgb.png, swir.png, and optional index.png/mapbox.png.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        metavar="MODEL",
        help="Gemini model ID.",
    )
    parser.add_argument(
        "--raw-out",
        default="gemini_raw.txt",
        metavar="FILE",
        help="Filename for the raw response saved inside the sample dir.",
    )
    parser.add_argument(
        "--no-schema",
        action="store_true",
        help="Disable response_json_schema to compare structured vs free-form behavior.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")

    sample_dir = Path(args.sample_dir).resolve()
    rgb_path = sample_dir / "rgb.png"
    swir_path = sample_dir / "swir.png"
    index_path = sample_dir / "index.png"
    mapbox_path = sample_dir / "mapbox.png"

    if not rgb_path.exists() or not swir_path.exists():
        raise FileNotFoundError(f"Missing rgb.png or swir.png in {sample_dir}")

    contents: list[types.Part | str] = [
        _image_part(rgb_path),
        _image_part(swir_path),
    ]
    has_index = index_path.exists()
    has_mapbox = mapbox_path.exists()
    if has_index:
        contents.append(_image_part(index_path))
    if has_mapbox:
        contents.append(_image_part(mapbox_path))
    contents.append(build_user_text(has_index, has_mapbox))

    config_kwargs: dict[str, object] = {
        "system_instruction": SYSTEM_PROMPT,
        "response_mime_type": "application/json",
        "temperature": 0.0,
        "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
    }
    if not args.no_schema:
        config_kwargs["response_json_schema"] = GEMINI_RESPONSE_SCHEMA

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=args.model,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    raw_text = (response.text or "").strip()
    raw_path = sample_dir / args.raw_out
    raw_path.write_text(raw_text, encoding="utf-8")
    print(f"Raw response saved to {raw_path}")

    if raw_text:
        print("\n--- response.text ---\n")
        print(raw_text)
    else:
        print("\n--- no response.text; dumping candidates ---\n")
        payload = response.model_dump(mode="json")
        dump_path = sample_dir / "gemini_response.json"
        dump_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Full response saved to {dump_path}")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
