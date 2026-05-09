# DataCenterWatch

Satellite monitoring for data-center and industrial-site detection using Sentinel-2 composites, SimSat, Gemini/LFM inference, a packet downlink pipeline, and a live React dashboard.

![DataCenterWatch demo](demo.png)

## What It Does

DataCenterWatch has four moving parts:

1. `SimSat/` simulates the satellite and serves imagery on `http://localhost:9005`.
2. `scripts/satellite.py` fetches tiles, runs the vision model, and writes downlink packets to `downlink_packets.jsonl`.
3. `scripts/ground.py` ingests those packets into `ground.db`.
4. The dashboard stack shows the results:
   - FastAPI backend: `app/api.py` on `http://localhost:8001`
   - React frontend: `dashboard/` on `http://localhost:5173`

## Repo Layout

- `src/datacenter_watch/`: core inference, SimSat fetch, change detection, and DB logic
- `scripts/satellite.py`: satellite-side inference runner
- `scripts/ground.py`: ground-side packet ingest
- `app/api.py`: API for the React dashboard
- `dashboard/`: React globe UI shown in the demo screenshot
- `SimSat/`: simulator and imagery API

## Prerequisites

Install these first:

- Python 3.11+
- `uv`
- Node.js 20+
- Docker Desktop

## Environment Files

There are two environment files involved.

### 1. Root `.env`

Place this file at:

```text
datacenter_watch/.env
```

This is used by the Python code in this repo, including `app/api.py` and the Gemini annotator.

Put in:

```env
GEMINI_API_KEY=your_google_ai_api_key
```

You can also use:

```env
GOOGLE_API_KEY=your_google_ai_api_key
```

Only one of `GEMINI_API_KEY` or `GOOGLE_API_KEY` is needed for the Gemini-backed flow.

### 2. SimSat `.env`

Place this file at:

```text
datacenter_watch/SimSat/.env
```

This is read by `SimSat/docker-compose.yaml`.

Put in:

```env
MAPBOX_ACCESS_TOKEN=your_mapbox_token
```

Mapbox is optional for some flows, but the SimSat stack expects this when serving Mapbox imagery.

## Install

From the repo root:

```bash
uv sync
cd dashboard
npm install
cd ..
```

## Run The Full Stack

Use five terminals from the repo root unless noted otherwise.

### Terminal 1: Start SimSat

```bash
cd SimSat
docker compose up --build
```

What you get:

- SimSat dashboard: `http://localhost:8000`
- SimSat imagery API: `http://localhost:9005`

After it starts, open `http://localhost:8000` and press the start button in the SimSat UI so the simulation leaves the zero-state.

### Terminal 2: Start Ground Ingest

From the repo root:

```bash
uv run scripts/ground.py --downlink downlink_packets.jsonl --db ground.db --follow
```

This watches `downlink_packets.jsonl` and continuously updates `ground.db`.

### Terminal 3: Start The Dashboard API

From the repo root:

```bash
uv run uvicorn app.api:app --reload --port 8001
```

This serves `/api/state`, `/api/sites/:id`, and image routes for the React UI.

### Terminal 4: Start The React Frontend

From the repo root:

```bash
cd dashboard
npm run dev
```

Open:

```text
http://localhost:5173
```

The Vite dev server proxies `/api` requests to `http://localhost:8001`.

### Terminal 5: Start Satellite Inference

From the repo root:

```bash
uv run scripts/satellite.py --backend gemini --watchlist-loop --interval-seconds 30 --downlink downlink_packets.jsonl
```

This continuously cycles through the watchlist, fetches imagery from SimSat, runs inference, and appends changes to `downlink_packets.jsonl`.

## Common Run Modes

### Watchlist Loop

```bash
uv run scripts/satellite.py --backend gemini --watchlist-loop --interval-seconds 30 --downlink downlink_packets.jsonl
```

### Single Watchlist Location

```bash
uv run scripts/satellite.py --backend gemini --watchlist-loop --watchlist-location aubix_data_center --interval-seconds 30 --downlink downlink_packets.jsonl
```

### Current Satellite Position Loop

```bash
uv run scripts/satellite.py --backend gemini --current-loop --interval-seconds 30 --downlink downlink_packets.jsonl
```

### One-Off Live Tile At A Specific Location And Time

```bash
uv run scripts/satellite.py --backend gemini --location aubix_data_center --timestamp 2026-05-08T23:00:00Z --downlink downlink_packets.jsonl
```

### Replay A Saved Dataset

```bash
uv run scripts/satellite.py --backend gemini --dataset-dir data/datacenter_watch --downlink downlink_packets.jsonl
```

## Minimal Dashboard Bring-Up

If `ground.db` already exists and you only want to inspect the UI, you do not need to run the satellite loop first.

Start just these:

```bash
uv run uvicorn app.api:app --reload --port 8001
cd dashboard
npm run dev
```

Then open `http://localhost:5173`.

## Notes

- `scripts/satellite.py` requires exactly one execution mode: `--watchlist-loop`, `--current-loop`, `--location`, `--sample-dir`, or `--dataset-dir`.
- If SimSat has not started yet, live modes will fail because the current position is still the zero-state.
- `scripts/ground.py --follow` should run against the same downlink file that `scripts/satellite.py` is writing.
- The React dashboard uses the ingested SQLite state, not the raw JSONL file directly.
- The legacy Streamlit dashboard still exists at `app/app.py`, but the screenshot in this README is the React dashboard in `dashboard/`.

## Main Entrypoints

- `scripts/satellite.py`: satellite-side inference and packet generation
- `scripts/ground.py`: ingest packets into SQLite
- `app/api.py`: backend for the React dashboard
- `dashboard/`: globe frontend
- `app/app.py`: older Streamlit dashboard
