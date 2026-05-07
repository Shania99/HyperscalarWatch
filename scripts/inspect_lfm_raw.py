"""Inspect raw local LFM output for a saved sample folder."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import socket
import sys
from pathlib import Path

from openai import OpenAI

from datacenter_watch.annotator import GEMINI_RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_text
from datacenter_watch.evaluator import start_llama_server, stop_server, wait_for_server


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _choose_local_port(port: int) -> int:
    if _port_is_available(port):
        return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _data_url(image_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.standard_b64encode(image_bytes).decode()


def _content_from_sample(sample_dir: Path) -> tuple[list[dict[str, object]], str]:
    rgb_path = sample_dir / "rgb.png"
    swir_path = sample_dir / "swir.png"
    if not rgb_path.exists() or not swir_path.exists():
        raise SystemExit(f"Missing rgb.png or swir.png in {sample_dir}")

    index_path = sample_dir / "index.png"
    mapbox_path = sample_dir / "mapbox.png"
    index_bytes = index_path.read_bytes() if index_path.exists() else None
    mapbox_bytes = mapbox_path.read_bytes() if mapbox_path.exists() else None

    content: list[dict[str, object]] = [
        {"type": "image_url", "image_url": {"url": _data_url(rgb_path.read_bytes())}},
        {"type": "image_url", "image_url": {"url": _data_url(swir_path.read_bytes())}},
    ]
    if index_bytes is not None:
        content.append({"type": "image_url", "image_url": {"url": _data_url(index_bytes)}})
    if mapbox_bytes is not None:
        content.append({"type": "image_url", "image_url": {"url": _data_url(mapbox_bytes)}})
    content.append(
        {
            "type": "text",
            "text": build_user_text(index_bytes is not None, mapbox_bytes is not None),
        }
    )
    return content, sample_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw local LFM response for a saved sample.")
    parser.add_argument("--sample-dir", required=True, help="Saved sample directory with rgb/swir/index/mapbox images.")
    parser.add_argument("--model", required=True, help="GGUF model path or HF repo.")
    parser.add_argument("--quant", default="", help="Quant level for llama-server when using an HF repo.")
    parser.add_argument("--mmproj", default=None, help="Optional mmproj GGUF for local backend.")
    parser.add_argument("--port", type=int, default=8080, help="Preferred llama-server port.")
    parser.add_argument("--verbose-server", action="store_true", help="Show llama-server output.")
    parser.add_argument("--out", default=None, help="Optional output file for raw response text.")
    args = parser.parse_args()

    if not shutil.which("llama-server"):
        print("llama-server not found on PATH.")
        sys.exit(1)

    sample_dir = Path(args.sample_dir)
    if not sample_dir.is_dir():
        raise SystemExit(f"Sample directory not found: {sample_dir}")

    content, label = _content_from_sample(sample_dir)
    port = _choose_local_port(args.port)
    if port != args.port:
        print(f"Port {args.port} is busy; using port {port} for llama-server.")

    print(f"Starting llama-server with model {args.model} on port {port} ...")
    server = start_llama_server(
        args.model,
        quant=args.quant or None,
        port=port,
        verbose=args.verbose_server,
        mmproj=args.mmproj,
    )
    try:
        wait_for_server(port=port)
        client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed")
        response = client.chat.completions.create(
            model=args.model,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "TileAnnotation", "schema": GEMINI_RESPONSE_SCHEMA},
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
        raw = response.choices[0].message.content or ""
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(raw, encoding="utf-8")
            print(f"Saved raw output to {out_path}")
        print(f"=== RAW LFM OUTPUT: {label} ===")
        print(raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print("=== JSON PARSE ===")
            print("invalid json")
        else:
            print("=== JSON PARSE ===")
            print(json.dumps(parsed, indent=2, sort_keys=True))
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
