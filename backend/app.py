from __future__ import annotations

import offline_policy
import network_status

import json
import sys
import hashlib
import os
import re
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import httpx
import http.client
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mc2_client import MCPMapClient, private_http_client
from job_runner import execute_job, prepare_output_review
from model_workers import prepare_prithvi
from raster_pipeline import read_stack
from algorithm_registry import MODEL_DETAILS, algorithm_training_rule
import retraining
import mission_control
from vector_import import read_polygons, import_folder, MAX_BYTES
from segmentation import execute as execute_segmentation, SPEC as SEGMENTATION_SPEC
import anomaly_api
import layer_insights
import basemap_manager
import local_imagery


ROOT = Path(__file__).parent
DATA_DIR = Path(os.getenv("SPECTRASCOPE_DATA_DIR", str(ROOT / "data")))
DATABASE_PATH = Path(os.getenv("SPECTRASCOPE_DB_PATH", str(DATA_DIR / "spectrascope.db")))
ARTIFACT_DIR = DATA_DIR / "aligned"
RASTER_CACHE_DIR = DATA_DIR / "raster-cache"
LOCAL_RASTER_ROOT = Path(os.getenv("LOCAL_RASTER_ROOT", str(DATA_DIR / "local-rasters"))).resolve()
LOCAL_RASTER_DISPLAY_ROOT = os.getenv("LOCAL_RASTER_DISPLAY_ROOT", "./data/spectrascope/local-rasters")
MAX_RASTER_DOWNLOAD_BYTES = int(os.getenv("MAX_RASTER_DOWNLOAD_BYTES", str(5 * 1024 * 1024 * 1024)))
MC2_URL = os.getenv("MC2_INTERNAL_URL", "http://mapcontrol:8000")
MC2_PUBLIC_URL = os.getenv("MC2_PUBLIC_URL", "http://localhost:8000")
CDSE_STAC = "https://stac.dataspace.copernicus.eu/v1/search"
USGS_STAC = "https://landsatlook.usgs.gov/stac-server/search"
HLS_STAC = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search"
INPE_STAC = "https://data.inpe.br/bdc/stac/v1/search"
COLLECTIONS = {"sentinel-2": "sentinel-2-l2a", "sentinel-2-l1c":"sentinel-2-l1c", "landsat": "landsat-c2l2-sr",
               "eo1-hyperion": "eo-1-hyperion-l1t", "hls-l30": "HLSL30_2.0", "hls-s30": "HLSS30_2.0",
               "cbers-4a-wpm": "CB4A-WPM-L4-DN-1"}
CATALOG_ENDPOINTS = {"sentinel-2": CDSE_STAC, "sentinel-2-l1c": CDSE_STAC, "landsat": USGS_STAC, "eo1-hyperion": USGS_STAC,
                     "hls-l30": HLS_STAC, "hls-s30": HLS_STAC, "cbers-4a-wpm": INPE_STAC}

def _application_version():
    for candidate in (ROOT.parent/'manifest.json',ROOT/'desktop/package.json'):
        try:return json.loads(candidate.read_text())['version']
        except (OSError,KeyError,ValueError):pass
    return 'development'
APP_VERSION=_application_version()
app = FastAPI(title="SpectraScope", version=APP_VERSION)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
map_client = MCPMapClient(MC2_URL)
map_state: dict[str, str] = {}
search_sessions: dict[str, dict[str, Any]] = {}
from runtime_profiles import performance_profile
from resource_admission import run_reserved
from public_raster_transport import public_http_client
model_executor = ThreadPoolExecutor(max_workers=performance_profile()['local_analysis_slots'], thread_name_prefix="spectrascope-inference")
auto_training_lock = asyncio.Lock()
server_loop = None


class SearchRequest(BaseModel):
    catalog: str = Field(pattern="^(" + "|".join((*COLLECTIONS, "local")) + ")$")
    geometry: dict[str, Any]
    start: date
    end: date
    cloud_max: float = Field(default=100, ge=0, le=100)
    limit: int = Field(default=50, ge=1, le=100)


class PointRequest(BaseModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)


class ZoomRequest(BaseModel):
    bbox: list[float] = Field(min_length=4, max_length=4)


class MoreRequest(BaseModel):
    search_id: str


class WorkspaceRequest(BaseModel):
    missions: list[dict[str, Any]] = Field(default_factory=list)


class InferenceRequest(BaseModel):
    mission_id: str
    job_id: str
    image_ids: list[str] = Field(default_factory=list)
    algorithms: list[str] = Field(default_factory=list)
    band_alignment: bool = True
    combination: str = Field(default="adaptive", pattern="^(adaptive|separate|ensemble)$")
    max_size: int = Field(default=384, ge=32, le=1024)
    epochs: int = Field(default=20, ge=1, le=200)
    geometry: dict[str, Any] | None = None
    label_ids: list[str] = Field(default_factory=list)


class GroundTruthRequest(BaseModel):
    mission_id: str
    job_id: str
    image_id: str | None = None
    label: str = Field(min_length=1, max_length=80)
    geometry: dict[str, Any] | None = None
    confidence: int = Field(default=3, ge=1, le=5)
    notes: str = Field(default="", max_length=500)
    role: str = Field(default="train", pattern="^(train|validation)$")


class ManualReviewRequest(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    geometry: dict[str, Any]
    source_model: str | None = None
    notes: str = Field(default='', max_length=500)
    confidence: int = Field(default=3, ge=1, le=5)
    role: str = Field(default='train', pattern='^(train|validation)$')
    segmentation_id: str | None = None
    segmentation_candidate: int | None = Field(default=None, ge=0)


class SegmentationPoint(BaseModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    label: int = Field(default=1, ge=0, le=1)


class SegmentationRequest(BaseModel):
    points: list[SegmentationPoint] = Field(default_factory=list, max_length=32)
    box_geometry: dict[str, Any] | None = None
    rgb_bands: list[str] = Field(default_factory=lambda:['RED','GREEN','BLUE'], min_length=3, max_length=3)


class FineTuneRequest(BaseModel):
    mission_id: str
    job_id: str
    model: str = Field(min_length=1, max_length=100)
    label_ids: list[str] = Field(default_factory=list)
    epochs: int = Field(default=20, ge=1, le=200)
    max_size: int = Field(default=384, ge=32, le=1024)


class RasterAccessRequest(BaseModel):
    mission_id: str
    job_id: str
    image_ids: list[str] = Field(min_length=1)
    mode: str = Field(pattern="^(download|stream|local_path)$")
    local_paths: dict[str, str] = Field(default_factory=dict)
    band_names: list[str] = Field(default_factory=list)
    processing_level: str = Field(default='unknown',pattern='^(unknown|L1C|L2A|RGB|HLS)$')
    temporal_dates: list[date] = Field(default_factory=list,max_length=3)


class VectorImportRequest(BaseModel):
    mission_id: str
    job_id: str
    image_id: str | None = None
    class_field: str | None = None
    class_mapping: dict[str,str] = Field(default_factory=dict)
    fixed_label: str = Field(default='', max_length=80)
    role: str = Field(default='train',pattern='^(train|validation)$')
    confidence: int = Field(default=3,ge=1,le=5)
    notes: str = Field(default='',max_length=500)
    skip_unlabelled: bool = False


def _database() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_database() -> None:
    with _database() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS workspace (id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL, updated_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS inference_runs (id TEXT PRIMARY KEY, mission_id TEXT NOT NULL, job_id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS ground_truth (id TEXT PRIMARY KEY, mission_id TEXT NOT NULL, job_id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS tuning_runs (id TEXT PRIMARY KEY, mission_id TEXT NOT NULL, job_id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS raster_access (id TEXT PRIMARY KEY, mission_id TEXT NOT NULL, job_id TEXT NOT NULL, image_id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS manual_reviews (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, image_index INTEGER NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)")
        conn.execute('CREATE TABLE IF NOT EXISTS segmentation_runs (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, image_index INTEGER NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)')
        mission_control.initialize(conn)
        anomaly_api.initialize(conn)
        retraining.initialize(conn)
        conn.execute('CREATE TABLE IF NOT EXISTS vector_import_commits (import_id TEXT NOT NULL, mission_id TEXT NOT NULL, job_id TEXT NOT NULL, scope TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(import_id,mission_id,job_id,scope))')


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _recommended_models(image_count: int, requested: list[str]) -> list[str]:
    if requested:
        unknown = set(requested) - set(MODEL_DETAILS)
        if unknown:
            raise HTTPException(422, f"Unknown models: {', '.join(sorted(unknown))}")
        return list(dict.fromkeys(requested))
    return [key for key,value in MODEL_DETAILS.items() if not value.get('pretrained_lulc')]


def _science_assets(image: dict[str, Any]) -> list[tuple[str, str]]:
    """Prefer imagery assets and exclude thumbnails, metadata, and product records."""
    preferred: list[tuple[str, str]] = []
    from data_policy import public_raster_url
    for key, href in (image.get("assets") or {}).items():
        if not isinstance(href, str) or not href.startswith(("http://", "https://")):
            continue
        try:public_raster_url(image.get('catalog'),href)
        except ValueError:continue
        path = urlparse(href).path.lower()
        key_lower = str(key).lower()
        is_raster = path.endswith((".tif", ".tiff", ".jp2")) or bool(re.fullmatch(r"b\d{1,2}a?(?:_\d+m)?", key_lower))
        if is_raster and not any(word in key_lower for word in ("thumbnail", "preview", "browse", "metadata", "xml", "qa", "mask", "visual", "overview", "scl")):
            preferred.append((str(key), href))
    return preferred


def _safe_local_raster_path(value: str) -> Path:
    candidate = Path(value)
    path = (LOCAL_RASTER_ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        path.relative_to(LOCAL_RASTER_ROOT)
    except ValueError as exc:
        raise HTTPException(422, f"Local paths must be inside {LOCAL_RASTER_DISPLAY_ROOT}") from exc
    if not path.is_file():
        raise HTTPException(404, f"Local raster was not found: {path}")
    if path.suffix.lower() not in (".tif", ".tiff", ".jp2"):
        raise HTTPException(415, "Local adapter accepts GeoTIFF or JPEG 2000 rasters")
    return path


def _tool_data(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("isError"):
        detail = " ".join(x.get("text", "") for x in result.get("content", []))
        raise HTTPException(502, detail or "MC² MCP tool failed")
    data = result.get("structuredContent")
    if isinstance(data, dict):
        return data
    for block in result.get("content", []):
        if block.get("type") == "text":
            try:
                parsed = json.loads(block.get("text", ""))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return {}


async def _ensure_map() -> dict[str, str]:
    if map_state.get("map_id"):
        return map_state
    _init_database()
    with _database() as conn:
        row=conn.execute("SELECT payload FROM settings WHERE key='map_session'").fetchone()
        theme_row=conn.execute("SELECT payload FROM settings WHERE key='map_theme'").fetchone()
    if row:
        saved=json.loads(row['payload'])
        try:
            _tool_data(await map_client.call('list_assets',{'map_id':saved['map_id']}))
            map_state.update({'map_id':saved['map_id'],'url':f"{MC2_PUBLIC_URL}/map/{saved['map_id']}?ui=controls"})
            return map_state
        except Exception:
            pass
    theme=json.loads(theme_row['payload']) if theme_row else 'dark'
    result = _tool_data(await map_client.call("create_map", {"theme": theme, "basemap": "offline" if offline_policy.enabled() else "satellite"}))
    map_id = result.get("map_id")
    if not map_id:
        raise HTTPException(503, "MC² did not return a map ID")
    map_state.update({"map_id": map_id, "url": f"{MC2_PUBLIC_URL}/map/{map_id}?ui=controls"})
    with _database() as conn:
        conn.execute("INSERT INTO settings(key,payload) VALUES('map_session',?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",(json.dumps(map_state),))
    return map_state


@app.get("/")
async def index(request: Request):
    # Preserve existing mission/report deep links, including their URL fragment.
    if request.query_params.get("mission"):
        return FileResponse(ROOT / "static" / "index.html")
    return FileResponse(ROOT / "static" / "welcome.html")


@app.get("/workspace")
async def workspace_page():
    return FileResponse(ROOT / "static" / "index.html")


@app.on_event("startup")
async def startup():
    global server_loop
    server_loop = asyncio.get_running_loop()
    _init_database()
    LOCAL_RASTER_ROOT.mkdir(parents=True, exist_ok=True)
    with _database() as conn:
        for row in conn.execute('SELECT id,payload FROM inference_runs').fetchall():
            run=json.loads(row['payload'])
            if run.get('status') in ('running','queued','pausing','paused','stopping'):
                if run.get('execution_route')=='enrolled worker pool':continue
                run.update(status='interrupted',stage='Server restarted during this job; submit it again',error='Interrupted by server restart')
                conn.execute('UPDATE inference_runs SET payload=? WHERE id=?',(json.dumps(run),row['id']))
                conn.execute('UPDATE tuning_runs SET payload=? WHERE id=?',(json.dumps(run),row['id']))
        workspace=conn.execute('SELECT payload FROM workspace WHERE id=1').fetchone()
        for row in conn.execute('SELECT id,payload FROM segmentation_runs').fetchall():
            task=json.loads(row['payload'])
            if task.get('status') in ('queued','running','pausing','paused','stopping'):
                task.update(status='interrupted',stage='Server restarted; submit segmentation again')
                conn.execute('UPDATE segmentation_runs SET payload=? WHERE id=?',(json.dumps(task),row['id']))
        for row in conn.execute('SELECT id,payload FROM anomaly_runs').fetchall():
            run=json.loads(row['payload'])
            if run.get('status') in ('queued','running','pausing','paused','stopping'):
                run.update(status='interrupted',stage='Server restarted; resubmit the anomaly run')
                conn.execute('UPDATE anomaly_runs SET payload=? WHERE id=?',(json.dumps(run),row['id']))
    async def resume_training():
        from mac_web_gate import wait_for_commit
        await wait_for_commit()
        for mission in (json.loads(workspace['payload']).get('missions',[]) if workspace else []):
            for job in mission.get('jobs',[]):
                asyncio.create_task(_check_auto_training(mission['id'],job['id']))
                asyncio.create_task(check_anomaly_training(mission['id'],job['id']))
    from mac_web_gate import pending
    if pending():
        asyncio.create_task(resume_training())
    else:
        await resume_training()


@app.get("/api/config")
async def config():
    network_status.start_monitor()
    network=await network_status.refresh()
    state = await _ensure_map()
    with _database() as db:
        row=db.execute("SELECT payload FROM settings WHERE key='map_theme'").fetchone()
        basemap_row=db.execute("SELECT payload FROM settings WHERE key='basemap_active'").fetchone()
    theme=json.loads(row['payload']) if row else 'dark'
    from resource_admission import configuration
    return {"app_version":APP_VERSION,"map_id": state["map_id"], "map_url": state["url"],
        "map_theme":theme, "basemap":json.loads(basemap_row['payload']) if basemap_row else None, "network":network, "offline": offline_policy.enabled(), "memory_profile": os.getenv('SPECTRASCOPE_MEMORY_PROFILE','auto'), "processing_resources":{**performance_profile(),"shared_host":configuration()}, "catalogs": [
        {"id":"local","name":"Local imagery","source":"Imported local science rasters"},
        {"id": "sentinel-2", "name": "Sentinel-2 L2A", "source": "Copernicus Data Space"},
        {"id": "landsat", "name": "Landsat Collection 2 L2", "source": "USGS LandsatLook"},
        {"id": "eo1-hyperion", "name": "EO-1 Hyperion L1T", "source": "USGS LandsatLook"},
        {"id": "hls-l30", "name": "HLS L30 · Landsat 30 m", "source": "NASA Earthdata"},
        {"id": "hls-s30", "name": "HLS S30 · Sentinel-2 30 m", "source": "NASA Earthdata"},
        {"id": "cbers-4a-wpm", "name": "CBERS-4A WPM · Level 4", "source": "INPE Brazil Data Cube"},
    ]}


@app.on_event('shutdown')
async def stop_network_monitor():
    await network_status.stop_monitor()

@app.get('/api/connectivity')
async def connectivity():
    network_status.start_monitor()
    return await network_status.refresh()

@app.get("/api/health")
async def health():
    try:
        state = await _ensure_map()
        return {"ok": True, "map_id": state["map_id"]}
    except HTTPException as exc:
        raise exc


@app.get('/api/ready')
async def ready():
    """Process readiness after ASGI startup; never creates an MCP map session."""
    return {'ok':True,'offline':offline_policy.enabled()}


def _cloud_cover(props: dict[str, Any]) -> float | None:
    for key in ("eo:cloud_cover", "landsat:cloud_cover_land", "landsat:cloud_cover"):
        value = props.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _asset_href(assets: dict[str, Any]) -> str | None:
    for key in ("thumbnail", "rendered_preview", "overview", "browse", "visual"):
        asset = assets.get(key)
        if isinstance(asset, dict) and asset.get("href"):
            return asset["href"]
    return None


def _normalize_item(item: dict[str, Any], catalog: str) -> dict[str, Any]:
    props = item.get("properties") or {}
    assets = item.get("assets") or {}
    platform = props.get("platform") or props.get("constellation") or ""
    return {
        "id": item.get("id", ""),
        "catalog": catalog,
        "processing_level":{"sentinel-2-l1c":"L1C","sentinel-2":"L2A","hls-l30":"HLS","hls-s30":"HLS","cbers-4a-wpm":"L4"}.get(catalog,props.get('processing:level','unknown')),
        **({"sensor":catalog} if catalog in ("hls-l30","hls-s30","cbers-4a-wpm") else {}),
        "collection": item.get("collection") or COLLECTIONS[catalog],
        "datetime": props.get("datetime") or props.get("start_datetime") or "",
        "cloud_cover": _cloud_cover(props),
        "platform": platform,
        "instrument": ", ".join(props.get("instruments", [])) if isinstance(props.get("instruments"), list) else props.get("instruments", ""),
        "geometry": item.get("geometry"),
        "bbox": item.get("bbox"),
        "thumbnail": None if catalog in ("hls-l30","hls-s30") else _asset_href(assets),
        "assets": {k: v.get("href") for k, v in assets.items() if isinstance(v, dict) and v.get("href")},
        "asset_metadata": {k: {field: v[field] for field in ("eo:bands", "raster:bands", "type", "roles") if field in v} for k, v in assets.items() if isinstance(v, dict)},
        "product_url": next((link.get("href") for link in item.get("links", []) if link.get("rel") in ("self", "canonical")), ""),
    }


def _next_link(data: dict[str, Any], endpoint: str) -> dict[str, Any] | None:
    link = next((entry for entry in data.get("links", []) if entry.get("rel") == "next" and entry.get("href")), None)
    if not link:
        return None
    href = urljoin(endpoint, link["href"])
    if not _catalog_origin_matches(href, endpoint):
        return None
    return {**link, "href": href}


def _catalog_origin_matches(href: str, endpoint: str) -> bool:
    try:
        target, origin = urlparse(href), urlparse(endpoint)
        return (
            target.scheme == origin.scheme == "https"
            and target.hostname == origin.hostname
            and (target.port if target.port is not None else 443) == (origin.port if origin.port is not None else 443)
            and target.path.rstrip("/") == origin.path.rstrip("/")
            and target.username is None and target.password is None
            and not target.fragment
        )
    except ValueError:
        return False


async def _catalog_request(client: httpx.AsyncClient, endpoint: str, initial_payload: dict[str, Any], link: dict[str, Any] | None = None) -> dict[str, Any]:
    if offline_policy.enabled():
        raise HTTPException(409, "Reconnect before requesting an online catalog page.")
    headers = {"Accept": "application/geo+json, application/json"}
    try:
        if link:
            if not _catalog_origin_matches(link["href"], endpoint):
                raise ValueError("Catalog pagination must stay on the configured HTTPS origin")
            method = str(link.get("method", "GET")).upper()
            if method == "POST":
                response = await client.post(link["href"], json=link.get("body", initial_payload), headers=headers, follow_redirects=False)
            elif method == "GET":
                response = await client.get(link["href"], headers=headers, follow_redirects=False)
            else:
                raise ValueError("Catalog pagination supports only GET and read-only search POST")
        else:
            response = await client.post(endpoint, json=initial_payload, headers=headers, follow_redirects=False)
        if response.is_redirect:
            raise ValueError("Catalog redirects are disabled to keep the search on its configured provider")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, f"Catalog returned {exc.response.status_code}: {exc.response.text[:400]}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, f"Catalog request failed: {exc}") from exc


async def _replace_footprints(state: dict[str, str], items: list[dict[str, Any]]) -> None:
    current = _tool_data(await map_client.call("list_assets", {"map_id": state["map_id"]}))
    for asset in current.get("assets", []):
        if str(asset.get("name", "")).startswith("Catalog footprints"):
            await map_client.call("delete_asset", {"map_id": state["map_id"], "asset_id": asset["asset_id"]})
    features = [
        {"type": "Feature", "geometry": item["geometry"], "properties": {"id": item["id"], "datetime": item["datetime"], "cloud_cover": item["cloud_cover"], "catalog": item["catalog"]}}
        for item in items if item.get("geometry")
    ]
    if features:
        await map_client.call("add_geojson", {
            "map_id": state["map_id"],
            "geojson": json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")),
            "name": f"Catalog footprints · {len(features)} scenes",
            # STAC scenes can overlap heavily. Use alpha in the fill color
            # (rather than whole-asset opacity) so boundaries stay crisp while
            # overlapping polygons do not read as one large highlighted region.
            "style": {"fill_color": "rgba(73, 212, 177, 0.08)", "stroke_color": "#baf9e8", "stroke_width": 1.4},
        })


@app.post("/api/search")
async def search(request: SearchRequest):
    if offline_policy.enabled() and request.catalog != 'local':
        raise HTTPException(409, 'Online catalogs are disabled in the offline desktop application. Import local imagery to search it here.')
    if request.start > request.end:
        raise HTTPException(422, "Start date must be before end date")
    if request.geometry.get("type") not in ("Point", "Polygon", "MultiPolygon"):
        raise HTTPException(422, "AOI must be a Point, Polygon, or MultiPolygon")

    if request.catalog == 'local':
        items = search_local_imagery(request.geometry, request.start, request.end)
        await _replace_footprints(await _ensure_map(), items)
        return {'items':items,'count':len(items),'catalog_count':len(items),'has_more':False,'search_id':None}

    from data_policy import public_search_geometry
    try:geometry=public_search_geometry(request.geometry)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    endpoint = CATALOG_ENDPOINTS[request.catalog]
    payload = {
        "collections": [COLLECTIONS[request.catalog]],
        "intersects": geometry,
        "datetime": f"{request.start.isoformat()}T00:00:00Z/{request.end.isoformat()}T23:59:59Z",
        "limit": request.limit,
    }
    items: list[dict[str, Any]] = []
    catalog_count = 0
    pages_fetched = 0
    async with public_http_client(CATALOG_ENDPOINTS.values(),timeout=45,allow_search_post=True) as client:
        data = await _catalog_request(client, endpoint, payload)
        while True:
            pages_fetched += 1
            raw_items = data.get("features", [])
            catalog_count += len(raw_items)
            page_items = [_normalize_item(item, request.catalog) for item in raw_items]
            existing_ids = {item["id"] for item in items}
            items.extend(item for item in page_items if (item["cloud_cover"] is None or item["cloud_cover"] <= request.cloud_max) and item["id"] not in existing_ids)
            next_link = _next_link(data, endpoint)
            if len(items) >= request.limit or not next_link or pages_fetched >= 10:
                break
            data = await _catalog_request(client, endpoint, payload, next_link)
    search_id = str(uuid4())
    search_sessions.clear()
    search_sessions[search_id] = {
        "catalog": request.catalog, "endpoint": endpoint, "payload": payload,
        "cloud_max": request.cloud_max, "next": next_link, "items": items,
        "catalog_count": catalog_count, "page_size": request.limit,
    }

    # Replace the prior result layer, leaving uploaded rasters and AOIs intact.
    state = await _ensure_map()
    try:
        await _replace_footprints(state, items)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Could not update MC² map: {exc}") from exc

    return {"items": items, "count": len(items), "catalog_count": catalog_count, "has_more": bool(next_link), "search_id": search_id}


@app.post("/api/search/more")
async def search_more(request: MoreRequest):
    if offline_policy.enabled():
        raise HTTPException(409, 'Reconnect before requesting another online catalog page.')
    session = search_sessions.get(request.search_id)
    if not session or not session.get("next"):
        raise HTTPException(404, "There are no more catalog pages for this search. Run the search again.")
    new_items: list[dict[str, Any]] = []
    existing_ids = {item["id"] for item in session["items"]}
    pages_fetched = 0
    async with public_http_client(CATALOG_ENDPOINTS.values(),timeout=45,allow_search_post=True) as client:
        while len(new_items) < session["page_size"] and session["next"] and pages_fetched < 10:
            pages_fetched += 1
            data = await _catalog_request(client, session["endpoint"], session["payload"], session["next"])
            raw_items = data.get("features", [])
            session["catalog_count"] += len(raw_items)
            page_items = [_normalize_item(item, session["catalog"]) for item in raw_items]
            new_items.extend(item for item in page_items if (item["cloud_cover"] is None or item["cloud_cover"] <= session["cloud_max"]) and item["id"] not in existing_ids)
            existing_ids.update(item["id"] for item in page_items)
            session["next"] = _next_link(data, session["endpoint"])
    session["items"].extend(new_items)
    try:
        await _replace_footprints(await _ensure_map(), session["items"])
    except Exception as exc:
        raise HTTPException(502, f"Could not update MC² map: {exc}") from exc
    return {"items": new_items, "count": len(session["items"]), "catalog_count": session["catalog_count"], "has_more": bool(session["next"])}


@app.post("/api/zoom")
async def zoom(request: ZoomRequest):
    state = await _ensure_map()
    await map_client.call("zoom", {"map_id": state["map_id"], "target": "bbox", "bbox": request.bbox})
    return {"ok": True}


@app.post('/api/map/footprint')
async def show_image_footprint(request: dict[str, Any]):
    """Draw a distinct selected-image outline, including local/mission imagery."""
    from rasterio.features import bounds as geometry_bounds, is_valid_geom
    geometry = request.get('geometry')
    if geometry is None:
        box = request.get('bbox')
        if not isinstance(box, list) or len(box) != 4:
            raise HTTPException(422, 'This image has no georeferenced footprint')
        try:
            west, south, east, north = [float(v) for v in box]
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, 'Invalid image bounds') from exc
        if not np.isfinite([west,south,east,north]).all() or not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise HTTPException(422, 'Invalid image bounds')
        geometry = {'type':'Polygon','coordinates':[[[west,south],[east,south],[east,north],[west,north],[west,south]]]}
    try:
        if geometry.get('type') not in ('Polygon','MultiPolygon') or not is_valid_geom(geometry):
            raise ValueError('Image footprint must be a polygon')
        box = list(geometry_bounds(geometry))
        if not np.isfinite(box).all() or not (-180 <= box[0] < box[2] <= 180 and -90 <= box[1] < box[3] <= 90):
            raise ValueError('Image footprint must use valid longitude/latitude bounds')
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    state = await _ensure_map()
    current = _tool_data(await map_client.call('list_assets', {'map_id':state['map_id']}))
    result = _tool_data(await map_client.call('add_geojson', {
        'map_id':state['map_id'], 'name':'Selected image footprint',
        'geojson':json.dumps({'type':'Feature','geometry':geometry,'properties':{'image_id':str(request.get('id',''))}}),
        'style':{'fill_color':'rgba(255, 195, 86, 0.12)','stroke_color':'#ffc356','stroke_width':3}}))
    for asset in current.get('assets', []):
        if asset.get('name') == 'Selected image footprint':
            await map_client.call('delete_asset', {'map_id':state['map_id'],'asset_id':asset['asset_id']})
    await map_client.call('zoom', {'map_id':state['map_id'],'target':'bbox','bbox':box})
    return {'ok':True,'bbox':box,'asset_id':result.get('asset_id')}


@app.post("/api/basemap")
async def basemap(request:dict[str,Any]):
    key=request.get('key')
    if key not in ('offline','offline_light','satellite','osm','carto_dark'):raise HTTPException(422,'Unsupported basemap')
    if key not in ('offline','offline_light'):
        if offline_policy.enabled():raise HTTPException(409,'Online maps cannot currently be reached. Local maps remain available.')
        if network_status.automatic() and not network_status.snapshot()['maps'].get(key,False):raise HTTPException(409,'This map provider is currently unreachable. A local world map is available.')
    _tool_data(await map_client.call('set_basemap',{'map_id':(await _ensure_map())['map_id'],'basemap':key}))
    if not request.get('automatic'):
        with _database() as db:db.execute("INSERT INTO settings(key,payload) VALUES('basemap_active',?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",(json.dumps(key),))
    return {'ok':True,'basemap':key}

@app.post('/api/map-theme')
async def map_theme(request:dict[str,Any]):
    theme=request.get('theme')
    if theme not in ('dark','light','auto'):raise HTTPException(422,'Choose Light, Dark or Follow system')
    _tool_data(await map_client.call('set_theme',{'map_id':(await _ensure_map())['map_id'],'theme':theme}))
    with _database() as db:
        db.execute("INSERT INTO settings(key,payload) VALUES('map_theme',?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",(json.dumps(theme),))
    return {'ok':True,'theme':theme}


@app.post("/api/point")
async def point(request: PointRequest):
    state = await _ensure_map()
    feature = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [request.longitude, request.latitude]}, "properties": {"name": "Point AOI"}}
    result = _tool_data(await map_client.call("add_geojson", {
        "map_id": state["map_id"], "geojson": json.dumps(feature), "name": "AOI · point",
        "style": {"fill_color": "#f5ae4e", "stroke_color": "#fff1d8", "stroke_width": 2},
    }))
    await map_client.call("zoom", {"map_id": state["map_id"], "target": "point", "lon": request.longitude, "lat": request.latitude, "zoom": 11})
    return {"ok": True, "asset_id": result.get("asset_id"), "geometry": feature["geometry"]}


@app.get("/api/aoi")
async def aoi():
    state = await _ensure_map()
    result = _tool_data(await map_client.call("list_assets", {"map_id": state["map_id"]}))
    assets = result.get("assets", [])
    drawn = [asset for asset in assets if str(asset.get("asset_type", "")).startswith("drawn_") or str(asset.get("name", "")).startswith("AOI")]
    if not drawn:
        raise HTTPException(422, "Draw a polygon or rectangle with the map toolbar, or place a point AOI first")
    asset = drawn[-1]
    geojson = asset.get("geojson")
    if isinstance(geojson, str):
        geojson = json.loads(geojson)
    if geojson and geojson.get("type") == "Feature":
        geometry = geojson.get("geometry")
    elif geojson and geojson.get("type") in ("Point", "Polygon", "MultiPolygon"):
        geometry = geojson
    else:
        raise HTTPException(422, "Could not read a valid AOI from the map")
    return {"geometry": geometry, "name": asset.get("name", "AOI"), "asset_id": asset.get("asset_id")}


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    mode: str = Form("rgb"),
    bands: str = Form("1,2,3"),
    band: int = Form(1),
    colormap: str = Form("viridis"),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".tif", ".tiff"):
        raise HTTPException(415, "MC² manual raster preview currently accepts GeoTIFF (.tif/.tiff). Convert other formats to GeoTIFF first.")
    if mode not in ("rgb", "singleband"):
        raise HTTPException(422, "Mode must be rgb or singleband")
    state = await _ensure_map()
    form = {"mode": mode, "name": file.filename or "Uploaded GeoTIFF", "bands": bands, "band": str(band), "colormap": colormap}
    async with private_http_client(MC2_URL,timeout=180) as client:
        try:
            response = await client.post(
                f"{MC2_URL}/api/maps/{state['map_id']}/upload-geotiff",
                files={"file": (file.filename, file.file, file.content_type or "image/tiff")},
                data=form,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise HTTPException(exc.response.status_code, detail) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"GeoTIFF upload failed: {exc}") from exc


@app.get("/api/workspace")
async def get_workspace():
    """Return the durable local mission graph and its analysis records."""
    _init_database()
    with _database() as conn:
        row = conn.execute("SELECT payload FROM workspace WHERE id = 1").fetchone()
        missions = json.loads(row["payload"]).get("missions", []) if row else []
        reports = [_honest_report(json.loads(item["payload"])) for item in conn.execute("SELECT payload FROM inference_runs ORDER BY created_at DESC")]
        labels = [json.loads(item["payload"]) for item in conn.execute("SELECT payload FROM ground_truth ORDER BY created_at DESC")]
        tuning = [json.loads(item["payload"]) for item in conn.execute("SELECT payload FROM tuning_runs ORDER BY created_at DESC")]
        raster_access = [json.loads(item["payload"]) for item in conn.execute("SELECT payload FROM raster_access ORDER BY created_at DESC")]
    with _database() as conn:
        controls = {r['mission_id']:r['action'] for r in conn.execute('SELECT * FROM mission_controls')}
    missions = [{**m, 'control':controls.get(m['id'],'resume')} for m in missions if controls.get(m['id'])!='delete']
    return {"workspace_exists": row is not None, "missions": missions, "reports": reports, "ground_truth": labels, "tuning_runs": tuning, "raster_access": raster_access, "local_raster_root": LOCAL_RASTER_DISPLAY_ROOT}


@app.put("/api/workspace")
async def save_workspace(request: WorkspaceRequest):
    _init_database()
    now = _now()
    def persist():
      with _database() as conn:
        # Freeze every selected band before opening a write transaction. Large
        # local copies must not block the event loop or hold a SQLite write lock.
        attach_local_imagery(conn, request.missions, now)
        archived_ids = {r['mission_id'] for r in conn.execute("SELECT mission_id FROM mission_controls WHERE action='delete'")}
        previous = conn.execute('SELECT payload FROM workspace WHERE id=1').fetchone()
        archived = [m for m in json.loads(previous['payload']).get('missions',[]) if m['id'] in archived_ids] if previous else []
        incoming=[m for m in request.missions if m['id'] not in archived_ids]
        old_missions=json.loads(previous['payload']).get('missions',[]) if previous else []
        # Automatic pipeline jobs are durable server-owned snapshots. An older
        # browser workspace save must not erase jobs created by the scheduler.
        for old in old_missions:
            target=next((m for m in incoming if m['id']==old['id']),None)
            if target is not None:
                known={j['id'] for j in target.get('jobs',[])}
                target.setdefault('jobs',[]).extend(j for j in old.get('jobs',[]) if j.get('pipeline_id') and j['id'] not in known)
        merged = incoming + archived
        conn.execute(
            "INSERT INTO workspace (id, payload, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at",
            (json.dumps({"missions":merged}), now),
        )
    from starlette.concurrency import run_in_threadpool
    try:
        await run_in_threadpool(persist)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    return {"ok": True, "updated_at": now}


@app.post('/api/missions/{mission_id}/control/{command}')
async def control_mission(mission_id: str, command: str):
    _init_database()
    try:
        result=mission_control.request(_database, mission_id, command)
        if globals().get('workflow_store'):
            for p in workflow_store.all('workflow_pipelines',mission_id):
                if command in ('stop','delete'):
                    p['state']='deleted' if command=='delete' else 'stopped';workflow_store.put('workflow_pipelines',p)
            for ex in workflow_store.all('workflow_executions',mission_id):
                if ex['status'] in mission_control.ACTIVE and command in ('stop','delete'):
                    ex['control']='stop';workflow_store.put('workflow_executions',ex,control_change=True)
            workflow_store.event('mission.'+command,'Mission control requested',mission_id=mission_id,audit=True)
        return result
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def _require_active_mission(mission_id):
    if not mission_control.allowed(_database, mission_id):
        raise HTTPException(409, 'Mission is paused, stopped, or deleted. Resume it before starting another run.')


@app.post('/api/raster-access/{access_id}/supplier-metadata')
async def attach_supplier_metadata(access_id: str, file: UploadFile = File(...)):
    """Version the calibration recipe while preserving the saved band pixels."""
    import tempfile
    from sensor_metadata import SAFE_NAMES, MAX_METADATA_BYTES, read_safe, calibrate_access, is_metadata, read_landsat, calibrate_landsat_access
    if not is_metadata(file.filename or ''):
        raise HTTPException(422,'Choose matching Sentinel SAFE XML or Landsat MTL metadata')
    content=await file.read(MAX_METADATA_BYTES+1)
    if len(content)>MAX_METADATA_BYTES:raise HTTPException(413,'Supplier metadata exceeds 2 MB')
    with _database() as db:
        row=db.execute('SELECT payload FROM raster_access WHERE id=?',(access_id,)).fetchone()
    if not row:raise HTTPException(404,'Saved raster access was not found')
    original=json.loads(row['payload'])
    metadata_root=DATA_DIR/'local-rasters'/'snapshots'/'metadata';metadata_root.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='metadata-',dir=metadata_root) as tmp:
            source=Path(tmp)/Path(file.filename).name;source.write_bytes(content)
            sentinel=source.name.upper() in SAFE_NAMES
            metadata=read_safe(source) if sentinel else read_landsat(source)
            destination=metadata_root/(metadata['metadata_sha256']+source.suffix.lower())
            record=(calibrate_access if sentinel else calibrate_landsat_access)(original,metadata,destination)
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest()!=metadata['metadata_sha256']:raise ValueError('Saved metadata hash mismatch')
            else:source.replace(destination)
        record.update(id=uuid4().hex,created_at=_now(),status='local-ready',message='Supplier radiometry attached as a new input recipe; original band files and earlier runs unchanged.')
        with _database() as db:
            db.execute('INSERT INTO raster_access VALUES(?,?,?,?,?,?)',(record['id'],record['mission_id'],record['job_id'],record['image_id'],json.dumps(record),record['created_at']))
        return {'record':record}
    except (ValueError,OSError) as exc:raise HTTPException(422,str(exc)) from exc


@app.get("/api/raster-access")
async def list_raster_access(mission_id: str, job_id: str | None = None):
    _init_database()
    query, values = "SELECT payload FROM raster_access WHERE mission_id = ?", [mission_id]
    if job_id:
        query += " AND job_id = ?"; values.append(job_id)
    query += " ORDER BY created_at DESC"
    with _database() as conn:
        return {"records": [json.loads(row["payload"]) for row in conn.execute(query, values)], "local_raster_root": LOCAL_RASTER_DISPLAY_ROOT}


async def _download_asset(catalog: str, url: str, destination: Path) -> int:
    from public_raster_transport import complete_download
    from starlette.concurrency import run_in_threadpool

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{uuid4().hex}.part{destination.suffix}")
    written = 0
    try:
        async with complete_download(catalog, url, MAX_RASTER_DOWNLOAD_BYTES) as (_, chunks):
            with temporary.open("xb") as target:
                async for chunk in chunks:
                    written += len(chunk)
                    target.write(chunk)
        await run_in_threadpool(offline_policy.raster_driver, str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return written


@app.post("/api/raster-access")
async def configure_raster_access(request: RasterAccessRequest):
    """Attach durable download, streaming, or local-path access to job imagery."""
    _init_database()
    if offline_policy.enabled() and request.mode != "local_path":
        raise HTTPException(409, "Offline desktop raster access requires local files. Streaming and remote downloads are disabled.")
    with _database() as conn:
        row = conn.execute("SELECT payload FROM workspace WHERE id = 1").fetchone()
    missions = json.loads(row["payload"]).get("missions", []) if row else []
    mission = next((item for item in missions if item.get("id") == request.mission_id), None)
    job = next((item for item in (mission or {}).get("jobs", []) if item.get("id") == request.job_id), None)
    if not mission or not job:
        raise HTTPException(404, "Mission or job was not found in the local workspace database")
    images = [image for image in job.get("images", []) if image.get("id") in request.image_ids]
    if len(images) != len(set(request.image_ids)):
        raise HTTPException(422, "One or more selected images do not belong to this job")
    records = []
    for image in images:
        if request.mode != 'local_path' and image.get('catalog') in ('hls-l30','hls-s30'):
            raise HTTPException(422, 'Download these HLS science files with NASA Earthdata Login, then import them as local imagery.')
        now = _now()
        record = {"id": str(uuid4()), "mission_id": request.mission_id, "job_id": request.job_id, "image_id": image.get("id"), "mode": request.mode, "created_at": now, "status": "ready", "band_names": request.band_names}
        record['processing_level']=request.processing_level if request.processing_level!='unknown' else image.get('processing_level','unknown')
        record['temporal_dates']=[value.isoformat() for value in request.temporal_dates] or image.get('temporal_dates',[])
        if request.mode == "stream":
            assets = _science_assets(image)
            if not assets:
                raise HTTPException(422, f"No streamable science raster asset was found for {image.get('id')}")
            record.update({"asset_urls": [{"key": key, "url": href} for key, href in assets], "status": "stream-ready", "message": "Remote raster URLs retained for range-request or COG streaming."})
        elif request.mode == "local_path":
            supplied = request.local_paths.get(str(image.get("id")), "")
            with _database() as conn:
                saved=conn.execute('SELECT payload FROM raster_access WHERE mission_id=? AND job_id=? AND image_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1',(request.mission_id,request.job_id,image['id'])).fetchone()
            connected=json.loads(saved['payload']) if saved else {}
            if not supplied and connected.get('local_sources'):
                # A metadata edit must not replace a logical product with its
                # first file. Retain the entire immutable band-source schema.
                for key in ('local_path','local_sources','sensor','source_revision','grouping_reviewed_at'):
                    if key in connected:record[key]=connected[key]
                if request.processing_level=='unknown':record['processing_level']=connected.get('processing_level',record['processing_level'])
                if not request.temporal_dates:record['temporal_dates']=connected.get('temporal_dates',record['temporal_dates'])
                count=sum(s['count'] for s in record['local_sources'])
                if request.band_names:
                    if len(request.band_names)!=count:raise HTTPException(422,f'Provide all {count} channel names; no source band can be dropped')
                    offset=0
                    for source in record['local_sources']:
                        source['band_names']=request.band_names[offset:offset+source['count']];offset+=source['count']
                else:record['band_names']=connected.get('band_names',[])
                record.update(status='local-ready',message=f'All {len(record["local_sources"])} source files remain connected; metadata saved as a new access version.')
            else:
                if not supplied:raise HTTPException(422,f"Provide a local path for {image.get('id')}")
                path = _safe_local_raster_path(supplied)
                record.update({"local_path": str(path), "relative_path": str(path.relative_to(LOCAL_RASTER_ROOT)), "bytes": path.stat().st_size, "status": "local-ready", "message": "Local raster path verified and ready for an inference adapter."})
        else:
            assets = _science_assets(image)
            if not assets:
                raise HTTPException(422, f"No downloadable science raster asset was found for {image.get('id')}")
            downloaded = []
            for key, href in assets:
                suffix = Path(urlparse(href).path).suffix.lower() or ".tif"
                scene_key=record["id"]
                asset_key=re.sub(r'[^a-zA-Z0-9_-]', '_', key)[:45]+'-'+hashlib.sha256(key.encode()).hexdigest()[:12]
                path = RASTER_CACHE_DIR / scene_key / f"{asset_key}{suffix}"
                try:
                    size = await _download_asset(image.get('catalog'), href, path)
                except (ValueError, OSError, http.client.HTTPException) as exc:
                    raise HTTPException(502, f"Could not download {key} for {image.get('id')}: {exc}") from exc
                downloaded.append({"key": key, "path": str(path), "bytes": size, "url": href})
            record.update({"files": downloaded, "status": "downloaded", "message": f"Downloaded {len(downloaded)} science raster asset(s) into the local raster cache."})
        try:
            from starlette.concurrency import run_in_threadpool
            probe = await run_in_threadpool(read_stack, record, image, 32)
            record['raster_metadata'] = {'bands': probe.bands, 'crs': str(probe.profile['crs']), 'read_verified': True}
        except Exception as exc:
            raise HTTPException(422, f"Raster access failed for {image.get('id')}: {exc}. Catalog assets may require provider credentials.") from exc
        with _database() as conn:
            conn.execute("INSERT INTO raster_access (id, mission_id, job_id, image_id, payload, created_at) VALUES (?, ?, ?, ?, ?, ?)", (record["id"], record["mission_id"], record["job_id"], record["image_id"], json.dumps(record), record["created_at"]))
        records.append(record)
    return {"records": records, "local_raster_root": LOCAL_RASTER_DISPLAY_ROOT}


@app.post("/api/inference/runs")
async def create_inference_run(request: InferenceRequest):
    _require_active_mission(request.mission_id)
    raise HTTPException(409,'Use Manual inference with qualified models, or Training & qualification to fit a new candidate.')


def _radiometry_schema(records):
    schemas=set()
    for record in records.values():
        schema={'sensor':record.get('sensor'),'level':record.get('processing_level'),
                'bands':record.get('band_names'),'sources':[
                    {**{k:source.get(k) for k in ('band_names','scales','offsets','scale_override','offset_override')},
                     **({'stac_channel_calibration':source['stac_channel_calibration']} if source.get('stac_channel_calibration') is not None else {})}
                    for source in record.get('local_sources',[])]}
        schemas.add(json.dumps(schema,sort_keys=True))
    return sorted(schemas)


async def _queue_inference_run(request: InferenceRequest, metadata: dict | None = None, evidence: dict | None = None):
    """Queue a real raster-to-predictions job, with persisted progress and metrics."""
    _init_database()
    with _database() as conn:
        row = conn.execute("SELECT payload FROM workspace WHERE id = 1").fetchone()
    missions = json.loads(row["payload"]).get("missions", []) if row else []
    mission = next((item for item in missions if item.get("id") == request.mission_id), None)
    job = next((item for item in (mission or {}).get("jobs", []) if item.get("id") == request.job_id), None)
    if not mission or not job:
        raise HTTPException(404, "Mission or job was not found in the local workspace database")

    _require_active_mission(request.mission_id)
    all_images = job.get("images", [])
    images = [image for image in all_images if not request.image_ids or image.get("id") in request.image_ids]
    if not images:
        raise HTTPException(422, "Choose at least one image from this job")
    with _database() as conn:
        access_rows = conn.execute(
            "SELECT payload FROM raster_access WHERE mission_id = ? AND job_id = ? ORDER BY created_at DESC",
            (request.mission_id, request.job_id),
        ).fetchall()
    latest_access: dict[str, dict[str, Any]] = {}
    for row in access_rows:
        access = json.loads(row["payload"])
        latest_access.setdefault(str(access.get("image_id")), access)
    available_access = [latest_access.get(str(image.get("id"))) for image in images]
    ready_access = [record for record in available_access if record and record.get("status") in {"stream-ready", "local-ready", "downloaded"}]
    if request.image_ids and set(request.image_ids) != {image['id'] for image in images}:
        raise HTTPException(422, 'Some requested images do not belong to this job')
    from portable_lulc import MODEL as PORTABLE
    if request.algorithms == [PORTABLE] and (metadata or {}).get('run_kind') in ('portable_validation','operational_inference'):
        models = [PORTABLE]
    else:
        models = _recommended_models(len(images), request.algorithms or (job.get("analysisPlan") or {}).get("algorithms", []))
    run = {
        "id": str(uuid4()), "mission_id": request.mission_id, "job_id": request.job_id,
        "created_at": _now(), "status": "queued", "progress": 0, "stage": "Waiting for model worker", "image_count": len(images),
        "combination": request.combination, "models": [], "requested_models": models,
        "band_alignment": {"enabled": request.band_alignment, "method": "Conservative edge phase correlation; accept shifts only when match improves" if request.band_alignment else "disabled"},
        "consensus": {"agreement_percent": None, "needs_user_review": None, "message": "Awaiting prediction maps"},
        "data_status": f"Raster access configured for {len(ready_access)} of {len(images)} image(s)",
        "raster_access": [{"image_id": image.get("id"), "mode": latest_access.get(str(image.get("id")), {}).get("mode"), "status": latest_access.get(str(image.get("id")), {}).get("status", "not configured")} for image in images],
        "label_ids": [], "run_kind":"inference", **(metadata or {}),
    }
    run['raster_access_snapshot']={str(image['id']):latest_access[str(image['id'])] for image in images if str(image['id']) in latest_access}
    import distributed_execution
    try:use_pool=distributed_execution.route(_database,run,run['raster_access_snapshot'])
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    if run.get('unet_workers'):
        from unet_pool import MAX_GRID, MAX_IMAGES
        if request.max_size>MAX_GRID or len(images)>MAX_IMAGES:
            raise HTTPException(422,f'Synchronized U-Net supports up to {MAX_IMAGES} source images at a {MAX_GRID} px grid')
    with _database() as conn:
        labels = [json.loads(row['payload']) for row in conn.execute('SELECT payload FROM ground_truth WHERE mission_id = ? AND job_id = ? ORDER BY created_at,rowid', (request.mission_id, request.job_id))]
        if request.label_ids:
            labels = [label for label in labels if label['id'] in request.label_ids or label.get('role')=='validation' or label.get('benchmark_split') in ('validation','test')]
        run['label_ids']=[label['id'] for label in labels]
        if run.get('run_kind')=='training':
            run['label_count']=len(labels);run['class_count']=len({label['label'] for label in labels if label.get('role','train')=='train'})
        if evidence:
            batch={**evidence,'id':str(uuid4()),'run_id':run['id'],'mission_id':request.mission_id,'job_id':request.job_id}
            conn.execute('INSERT INTO auto_training_batches VALUES (?,?,?,?,?,?)',(batch['id'],request.mission_id,request.job_id,evidence['model_id'],evidence['evidence_key'],json.dumps(batch)))
        conn.execute("INSERT INTO inference_runs (id, mission_id, job_id, payload, created_at) VALUES (?, ?, ?, ?, ?)", (run["id"], run["mission_id"], run["job_id"], json.dumps(run), run["created_at"]))
        if run.get('run_kind')=='training':
            conn.execute('INSERT INTO tuning_runs VALUES (?,?,?,?,?)',(run['id'],request.mission_id,request.job_id,json.dumps(run),run['created_at']))
    initial_checkpoints={}
    with _database() as conn:
        previous=conn.execute('SELECT payload FROM inference_runs WHERE mission_id=? AND job_id=? ORDER BY created_at DESC',(request.mission_id,request.job_id)).fetchall()
    for row in previous:
        record=json.loads(row['payload'])
        prior_inputs=record.get('raster_access_snapshot')
        if prior_inputs and _radiometry_schema(prior_inputs)!=_radiometry_schema(run['raster_access_snapshot']):
            continue
        if not prior_inputs and any(source.get('radiometry') for access in run['raster_access_snapshot'].values() for source in access.get('local_sources',[])):
            # No evidence that a legacy checkpoint used the now-corrected units.
            continue
        # Candidates can seed further training, but ordinary inference keeps its
        # established model until a candidate is explicitly selected/promoted.
        if run.get('run_kind')!='training' and record.get('candidate_version'):
            continue
        for model in record.get('models',[]):
            path=model.get('training',{}).get('path')
            if model.get('status')=='complete' and path and Path(path).resolve().is_relative_to((DATA_DIR/'results').resolve()):
                initial_checkpoints.setdefault(model['id'],path)
    if run.get('run_kind')=='operational_inference':
        initial_checkpoints={k:v for k,v in run.get('qualified_checkpoints',{}).items() if v}
    if use_pool:
        pool_executor.submit(distributed_execution.execute,_database,dict(run),_persist_run,images,run['raster_access_snapshot'],labels,DATA_DIR,
                             {'epochs':request.epochs,'max_size':request.max_size,'geometry':request.geometry,'initial_checkpoints':initial_checkpoints},shutdown=pool_shutdown)
    else:
        model_executor.submit(mission_control.execute, _database, 'inference_runs', dict(run), execute_job, _persist_run,
                              images, latest_access, labels, DATA_DIR, epochs=request.epochs, max_size=request.max_size,
                              geometry=request.geometry, initial_checkpoints=initial_checkpoints)
    return run


def _persist_run(run: dict) -> None:
    if globals().get('workflow_store') and run.get('execution_route')!='enrolled worker pool':workflow_store.run_changed(run)
    with _database() as conn:
        if run.get('execution_route')=='enrolled worker pool':
            conn.execute('BEGIN IMMEDIATE');saved=conn.execute('SELECT payload FROM inference_runs WHERE id=?',(run['id'],)).fetchone()
            stored=json.loads(saved['payload']) if saved else {}
            for key in ('control_action','control_stop','archive_after_stop'):
                if key in stored:run[key]=stored[key]
            if stored.get('control_stop') and run.get('status')=='complete':run.update(status='cancelled',stage='Stopped before publication; verified artifacts preserved')
        conn.execute('UPDATE inference_runs SET payload = ? WHERE id = ?', (json.dumps(run, default=str), run['id']))
        row=conn.execute('SELECT payload FROM tuning_runs WHERE id=?',(run['id'],)).fetchone()
        if row:
            merged={**json.loads(row['payload']),**run}
            conn.execute('UPDATE tuning_runs SET payload = ? WHERE id = ?', (json.dumps(merged, default=str), run['id']))
    if globals().get('workflow_store') and run.get('execution_route')=='enrolled worker pool':workflow_store.run_changed(run)
    if run.get('status')=='cancelled':
        folder=DATA_DIR/'results'/run['id'];folder.mkdir(parents=True,exist_ok=True)
        (folder/'report.json').write_text(json.dumps(run,indent=2,default=str))
    if run.get('status') in ('complete','partial','failed','needs_input') and server_loop and server_loop.is_running():
        server_loop.call_soon_threadsafe(lambda:asyncio.create_task(_check_auto_training(run['mission_id'],run['job_id'])))


def _honest_report(run: dict) -> dict:
    for item in [*run.get('models',[]),*run.get('outputs',[])]:
        metrics=item.get('metrics',{})
        macro=metrics.get('per_class',{}).get('macro avg',{})
        if metrics.get('status')=='measured':
            if 'precision' in macro:metrics.setdefault('macro_precision',macro['precision'])
            if 'recall' in macro:metrics.setdefault('macro_recall',macro['recall'])
    if 'requested_models' not in run:
        run.update(status='legacy_placeholder', progress=0, stage='Legacy planning report; no models executed',
                   consensus={'agreement_percent': None, 'needs_user_review': None,
                              'message':'This legacy report contained simulated agreement. It is invalid for evaluation; run the connected workers again.'})
    elif run.get('status') in ('complete', 'partial'):
        for index, output in enumerate(run.get('outputs', [])):
            if (DATA_DIR/'results'/run['id']/f'{index}-consensus.tif').is_file():
                output.update(prepare_output_review(run, index, DATA_DIR))
                if output.get('agreement_file'):
                    output.setdefault('files', {})['agreement'] = output['agreement_file']
                folder = DATA_DIR/'results'/run['id']
                if (folder/f'{index}-reviewed.tif').is_file():
                    output['has_manual_reviews'] = True
                    output['files']['reviewed'] = f'/api/results/{run["id"]}/{index}-reviewed.tif'
                    output['review_history'] = f'/api/results/{run["id"]}/{index}-reviews.json'
        if run.get('consensus', {}).get('models_compared', 0) > 1:
            run['consensus']['message'] = 'Green: all models predict the same class. Orange: at least one model differs. Agreement is not accuracy. Transparent areas have no valid data.'
    return run


@app.get('/api/inference/runs/{run_id}')
async def get_inference_run(run_id: str):
    with _database() as conn:
        row = conn.execute('SELECT payload FROM inference_runs WHERE id = ?', (run_id,)).fetchone()
    if not row:
        raise HTTPException(404, 'Inference run not found')
    return _honest_report(json.loads(row['payload']))


@app.get('/api/results/{run_id}/{filename}')
async def result_artifact(run_id: str, filename: str):
    root = (DATA_DIR/'results').resolve()
    path = (root/run_id/filename).resolve()
    if not path.is_relative_to(root) or path.suffix not in ('.tif','.png','.json') or not path.is_file():
        raise HTTPException(404, 'Result artifact not found')
    media = 'image/png' if path.suffix == '.png' else 'application/json' if path.suffix == '.json' else 'image/tiff'
    return FileResponse(path, media_type=media, filename=path.name if path.suffix != '.png' else None)


@app.get('/api/models/status')
async def model_status():
    import importlib.util
    cache=DATA_DIR/'models'/'prithvi'
    weights=list(cache.rglob('*.pt')) if cache.exists() else []
    bundled=os.getenv('SPECTRASCOPE_BUNDLED_MODELS')
    if not weights and bundled:weights=list((Path(bundled)/'prithvi').rglob('*.pt'))
    from pretrained_lulc import PRETRAINED
    return {'models':[{'id':key,'name':value['name'], 'runtime_available':importlib.util.find_spec('tensorflow' if key=='dynamicworld' else 'segmentation_models_pytorch' if key=='openearthmap' else 'torch' if key in ('unet','prithvi','prithvi_crop') else 'xgboost' if key=='xgboost' else 'sklearn') is not None,
                       'pretrained_weights_cached':bool(weights) if key=='prithvi' else None,
                       'training':'Pretrained encoder and pixel head; inference needs no labels; separate supervised fine-tuning available' if key in PRETRAINED else 'Frozen Prithvi encoder with trainable downstream head' if key=='prithvi' else 'Trains on job ground truth',
                       'input_requirements':PRETRAINED[key]['input'] if key in PRETRAINED else None,
                       'native_classes':PRETRAINED[key]['classes'] if key in PRETRAINED else None, 'satellite_suitability':value['satellite_suitability']} for key,value in MODEL_DETAILS.items()]}


def _result_folder(run_id: str) -> Path:
    root = (DATA_DIR/'results').resolve()
    folder = (root/run_id).resolve()
    if not folder.is_relative_to(root):
        raise HTTPException(422, 'Invalid result path')
    return folder


@app.get('/api/results/{run_id}/{image_index}/inspect')
async def inspect_result(run_id: str, image_index: int, longitude: float, latitude: float):
    import rasterio
    from rasterio.warp import transform, transform_geom
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise HTTPException(422, 'Invalid map coordinates')
    report = await get_inference_run(run_id)
    outputs = report.get('outputs', [])
    if image_index < 0 or image_index >= len(outputs):
        raise HTTPException(404, 'Image result not found')
    folder = _result_folder(run_id)
    with rasterio.open(folder/f'{image_index}-consensus.tif') as source:
        xs, ys = transform('EPSG:4326', source.crs, [longitude], [latitude])
        row, col = source.index(xs[0], ys[0])
        if not (0 <= row < source.height and 0 <= col < source.width):
            raise HTTPException(422, 'Click inside the classified image')
        window = rasterio.windows.Window(col, row, 1, 1)
        value = int(source.read(1, window=window)[0, 0])
        if value < 1 or not source.read_masks(1, window=window)[0, 0]:
            raise HTTPException(422, 'This pixel has no valid prediction')
        corners = [source.transform*(x,y) for x,y in [(col,row),(col+1,row),(col+1,row+1),(col,row+1),(col,row)]]
        geometry = transform_geom(source.crs, 'EPSG:4326', {'type':'Polygon','coordinates':[corners]})
    classes = report['classes']
    votes = []
    for model in report.get('models', []):
        if model.get('status') != 'complete':
            votes.append({'id':model['id'], 'name':model['name'], 'status':model.get('status'), 'label':None})
            continue
        with rasterio.open(folder/f'{image_index}-{model["id"]}.tif') as source:
            vote = int(source.read(1, window=window)[0,0])
        votes.append({'id':model['id'], 'name':model['name'], 'status':'complete',
                      'label':classes[vote-1] if 0 < vote <= len(classes) else None})
    tally = {label:sum(item['label'] == label for item in votes) for label in classes}
    reviewed = None
    if (folder/f'{image_index}-reviewed-mask.tif').is_file():
        with rasterio.open(folder/f'{image_index}-reviewed-mask.tif') as source:
            approved = bool(source.read(1, window=window)[0,0])
        if approved:
            with rasterio.open(folder/f'{image_index}-reviewed.tif') as source:
                reviewed = classes[int(source.read(1, window=window)[0,0])-1]
    return {'image_id':outputs[image_index]['image_id'], 'longitude':longitude, 'latitude':latitude,
            'row':row, 'column':col, 'geometry':geometry, 'votes':votes, 'tally':tally,
            'final_label':classes[value-1], 'reviewed_label':reviewed,
            'disagreement':len({vote['label'] for vote in votes if vote['label'] is not None}) > 1,
            'combination':report.get('combination'), 'selected_model':report.get('selected_model'),
            'message':'Votes are model class predictions. The original final class follows the run combination policy; per-model pixel probabilities were not saved.'}


def _persist_segmentation(task):
    with _database() as conn:
        conn.execute('UPDATE segmentation_runs SET payload=? WHERE id=?',(json.dumps(task),task['id']))
    folder=DATA_DIR/'segmentations'/task['id']
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'report.json').write_text(json.dumps(task,indent=2))


@app.get('/api/segmentation/models')
async def segmentation_models():
    from sam3_runtime import runtime_available,weights_available
    return {'models':[{**SEGMENTATION_SPEC,'runtime_available':runtime_available(),'weights_cached':weights_available(DATA_DIR)}]}


@app.get('/api/segmentations')
async def list_segmentations(run_id: str, image_index: int):
    with _database() as conn:
        return [json.loads(r['payload']) for r in conn.execute(
            'SELECT payload FROM segmentation_runs WHERE run_id=? AND image_index=? ORDER BY created_at DESC', (run_id,image_index))]


@app.get('/api/segmentations/{task_id}')
async def get_segmentation(task_id: str):
    with _database() as conn:
        row=conn.execute('SELECT payload FROM segmentation_runs WHERE id=?',(task_id,)).fetchone()
    if not row:raise HTTPException(404,'Segmentation not found')
    return json.loads(row['payload'])


@app.get('/api/segmentations/{task_id}/{filename}')
async def segmentation_artifact(task_id: str, filename: str):
    await get_segmentation(task_id)
    root=(DATA_DIR/'segmentations'/task_id).resolve()
    path=(root/filename).resolve()
    if not path.is_relative_to(root) or path.suffix not in ('.png','.tif','.geojson','.json') or not path.is_file():
        raise HTTPException(404,'Segmentation artifact not found')
    return FileResponse(path,media_type='image/png' if path.suffix=='.png' else 'application/geo+json' if path.suffix=='.geojson' else 'application/json' if path.suffix=='.json' else 'image/tiff')


@app.post('/api/results/{run_id}/{image_index}/segmentations')
async def create_segmentation(run_id: str, image_index: int, request: SegmentationRequest):
    import rasterio
    from rasterio.features import bounds as geometry_bounds, is_valid_geom
    from sam3_runtime import runtime_available,weights_available
    if not runtime_available() or not weights_available(DATA_DIR):raise HTTPException(409,'Install the offline SAM 3 component before running segmentation. No download or SAM 2 fallback will occur.')
    report=await get_inference_run(run_id)
    _require_active_mission(report['mission_id'])
    if not 0<=image_index<len(report.get('outputs',[])):raise HTTPException(404,'Image result not found')
    if not request.points and not request.box_geometry:raise HTTPException(422,'Click inside an object or draw a box')
    if request.points and not request.box_geometry and not any(p.label==1 for p in request.points):
        raise HTTPException(422,'Include at least one foreground point')
    if request.box_geometry:
        try:
            if request.box_geometry.get('type') not in ('Polygon','MultiPolygon') or not is_valid_geom(request.box_geometry):
                raise ValueError('Draw a polygon or rectangle; its enclosing box is used as the prompt')
            b=geometry_bounds(request.box_geometry)
            if not np.all(np.isfinite(b)) or b[0]<-180 or b[2]>180 or b[1]<-90 or b[3]>90:raise ValueError('Use longitude/latitude geometry')
        except Exception as exc:raise HTTPException(422,str(exc)) from exc
    with _database() as conn:
        workspace=json.loads(conn.execute('SELECT payload FROM workspace WHERE id=1').fetchone()['payload'])
        mission=next((m for m in workspace['missions'] if m['id']==report['mission_id']),{})
        job=next((j for j in mission.get('jobs',[]) if j['id']==report['job_id']),{})
        image_id=report['outputs'][image_index]['image_id']
        image=next((i for i in job.get('images',[]) if i['id']==image_id),None)
        row=conn.execute('SELECT payload FROM raster_access WHERE mission_id=? AND job_id=? AND image_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1',
            (report['mission_id'],report['job_id'],image_id)).fetchone()
    if not image or not row:raise HTTPException(422,'Configure readable raster access for this image')
    task={'id':str(uuid4()),'run_id':run_id,'image_index':image_index,'mission_id':report['mission_id'],
        'job_id':report['job_id'],'image_id':image_id,'model':SEGMENTATION_SPEC['id'],
        'created_at':_now(),'status':'queued','progress':0,'stage':'Waiting for segmentation worker',**request.model_dump()}
    with _database() as conn:
        conn.execute('INSERT INTO segmentation_runs VALUES(?,?,?,?,?)',(task['id'],run_id,image_index,json.dumps(task),task['created_at']))
    model_executor.submit(mission_control.execute,_database,'segmentation_runs',dict(task),execute_segmentation,
        _persist_segmentation,DATA_DIR,_result_folder(run_id)/f'{image_index}-consensus.tif',image,report.get('raster_access_snapshot',{}).get(str(image_id)) or json.loads(row['payload']))
    return task


@app.post('/api/segmentations/{task_id}/candidates/{candidate_id}/map')
async def map_segmentation(task_id: str, candidate_id: int):
    from rasterio.features import bounds as geometry_bounds
    task=await get_segmentation(task_id)
    candidate=next((c for c in task.get('candidates',[]) if c['id']==candidate_id),None)
    if not candidate:raise HTTPException(404,'Segmentation proposal not found')
    state=await _ensure_map()
    current=_tool_data(await map_client.call('list_assets',{'map_id':state['map_id']}))
    result=_tool_data(await map_client.call('add_geojson',{'map_id':state['map_id'],
        'geojson':json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','properties':{'proposal':True},'geometry':candidate['geometry']}]}),
        'name':f'SAM proposal · {task_id} · {candidate_id}',
        'style':{'fill_color':'#72d5d0','fill_opacity':0.3,'stroke_color':'#b3fff2','stroke_width':2}}))
    for asset in current.get('assets',[]):
        if str(asset.get('name','')).startswith('SAM proposal · '):
            await map_client.call('delete_asset',{'map_id':state['map_id'],'asset_id':asset['asset_id']})
    await map_client.call('zoom',{'map_id':state['map_id'],'target':'bbox','bbox':list(geometry_bounds(candidate['geometry']))})
    return {'ok':True,'asset_id':result.get('asset_id')}


@app.post('/api/results/{run_id}/{image_index}/reviews')
async def save_manual_review(run_id: str, image_index: int, request: ManualReviewRequest):
    import rasterio
    from rasterio.features import rasterize, is_valid_geom, bounds as geometry_bounds
    from rasterio.warp import transform_geom
    from PIL import Image
    from job_runner import PALETTE
    report = await get_inference_run(run_id)
    if image_index < 0 or image_index >= len(report.get('outputs', [])):
        raise HTTPException(404, 'Image result not found')
    if request.label not in report.get('classes', []):
        raise HTTPException(422, 'Choose a class from this run')
    if request.source_model and request.source_model not in {m['id'] for m in report.get('models',[]) if m.get('status')=='complete'}:
        raise HTTPException(422, 'The selected model did not complete this run')
    segmentation_provenance=None
    if request.segmentation_id:
        task=await get_segmentation(request.segmentation_id)
        if task['run_id']!=run_id or task['image_index']!=image_index or task.get('status')!='complete':
            raise HTTPException(422,'Choose a completed segmentation for this image result')
        candidate=next((c for c in task.get('candidates',[]) if c['id']==request.segmentation_candidate),None)
        if not candidate:raise HTTPException(422,'Segmentation candidate not found')
        request=request.model_copy(update={'geometry':candidate['geometry'],'source_model':None})
        segmentation_provenance={'model':task['model'],'task_id':task['id'],'candidate':candidate['id'],
            'provenance':task['provenance'],'prompts':task['points'],'box_geometry':task.get('box_geometry'),
            'approved_by':'human','predicted_mask_quality':candidate['predicted_mask_quality']}
    try:
        if request.geometry.get('type') not in ('Point','Polygon','MultiPolygon') or not is_valid_geom(request.geometry):
            raise ValueError('Draw a valid point or polygon')
        bounds = geometry_bounds(request.geometry)
        if not np.all(np.isfinite(bounds)) or bounds[0] < -180 or bounds[2] > 180 or bounds[1] < -90 or bounds[3] > 90:
            raise ValueError('Review geometry must use longitude and latitude')
    except Exception as exc:
        raise HTTPException(422, f'Invalid review geometry: {exc}') from exc
    folder = _result_folder(run_id)
    output = report['outputs'][image_index]
    with rasterio.open(folder/f'{image_index}-consensus.tif') as source:
        original = source.read(1)
        valid = (original > 0) & (source.read_masks(1) > 0)
        profile = source.profile.copy()
    region = rasterize([(transform_geom('EPSG:4326', profile['crs'], request.geometry),1)],
                       out_shape=original.shape, transform=profile['transform'], dtype='uint8').astype(bool) & valid
    if not region.any():
        raise HTTPException(422, 'The reviewed geometry contains no valid analysis pixels')
    review_id = str(uuid4())
    record = {'id':review_id, 'run_id':run_id, 'image_index':image_index,
              'mission_id':report['mission_id'], 'job_id':report['job_id'], 'image_id':output['image_id'],
              **request.model_dump(), 'pixels':int(region.sum()), 'created_at':_now()}
    if segmentation_provenance:record['segmentation_provenance']=segmentation_provenance
    ground_truth = {**record, 'review_id':review_id}
    _init_database()
    with _database() as conn:
        # Apply the new decision last; prior decisions and original outputs remain auditable.
        existing = [json.loads(row['payload']) for row in conn.execute(
            'SELECT payload FROM manual_reviews WHERE run_id=? AND image_index=? ORDER BY created_at, rowid', (run_id,image_index))]
        corrected = original.copy(); approved = np.zeros_like(original, dtype='uint8')
        for decision in [*existing, record]:
            selected = rasterize([(transform_geom('EPSG:4326', profile['crs'], decision['geometry']),1)],
                                 out_shape=original.shape, transform=profile['transform'], dtype='uint8').astype(bool) & valid
            corrected[selected] = report['classes'].index(decision['label']) + 1
            approved[selected] = 1
        with rasterio.open(folder/f'{image_index}-reviewed.tif','w',**profile) as target:
            target.write(corrected,1)
        with rasterio.open(folder/f'{image_index}-reviewed-mask.tif','w',**{**profile,'dtype':'uint8','nodata':255}) as target:
            target.write(np.where(valid,approved,255).astype('uint8'),1)
        rgb = PALETTE[np.maximum(corrected.astype('int32')-1,0)%len(PALETTE)].copy(); rgb[~valid]=0
        with rasterio.open(folder/f'{image_index}-reviewed-rgb.tif','w',**{**profile,'count':3,'dtype':'uint8','nodata':0}) as target:
            target.write(np.moveaxis(rgb,-1,0))
        Image.fromarray(np.dstack((rgb,valid.astype('uint8')*220))).save(folder/f'{image_index}-reviewed.png')
        conn.execute('INSERT INTO manual_reviews VALUES (?,?,?,?,?)',(review_id,run_id,image_index,json.dumps(record),record['created_at']))
        conn.execute('INSERT INTO ground_truth VALUES (?,?,?,?,?)',(review_id,report['mission_id'],report['job_id'],json.dumps(ground_truth),record['created_at']))
    (folder/f'{image_index}-reviews.json').write_text(json.dumps([*existing,record],indent=2))
    if globals().get('workflow_store'):workflow_store.event('review.saved','Human classification correction saved',mission_id=report['mission_id'],run_id=run_id,image_id=output['image_id'],reference_id=review_id,role=request.role,pixels=record['pixels'],audit=True)
    automatic_training = await _check_auto_training(report['mission_id'],report['job_id'])
    return {'review':record, 'ground_truth':ground_truth, 'reviewed_pixels':int((approved==1).sum()),
            'reviewed_file':f'/api/results/{run_id}/{image_index}-reviewed.tif', 'automatic_training':automatic_training}


def _reference_scope(mission_id, job_id, image_id):
    _init_database()
    with _database() as db:
        row=db.execute('SELECT payload FROM workspace WHERE id=1').fetchone()
        missions=json.loads(row['payload']).get('missions',[]) if row else []
        mission=next((m for m in missions if m['id']==mission_id and not m.get('deletedAt')), {})
        job=next((j for j in mission.get('jobs',[]) if j['id']==job_id), {})
        image=next((i for i in job.get('images',[]) if i['id']==image_id),None)
        access=db.execute('SELECT payload FROM raster_access WHERE mission_id=? AND job_id=? AND image_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1',(mission_id,job_id,image_id)).fetchone()
    if not image or not access:raise HTTPException(409,'Connect this image in Source data before labeling it')
    return image,json.loads(access['payload'])


_vector_tasks = set()
_vector_export_lock = asyncio.Lock()


def _vector_export_folder(export_id):
    import re
    if not re.fullmatch('[a-f0-9-]{36}',export_id):raise HTTPException(422,'Invalid vector export identifier')
    return DATA_DIR/'vector-exports'/export_id


def _save_vector_export(folder, record):
    path=folder/(uuid4().hex+'.json');path.write_text(json.dumps(record,indent=2));path.replace(folder/'status.json')


@app.post('/api/results/{run_id}/{image_index}/vector-export')
async def create_vector_export(run_id: str, image_index: int):
    from starlette.concurrency import run_in_threadpool
    from result_vectors import polygon_export
    report=await get_inference_run(run_id)
    if image_index<0 or image_index>=len(report.get('outputs',[])):raise HTTPException(404,'Classification output not found')
    source=_result_folder(run_id)/f'{image_index}-consensus.tif'
    if not source.is_file():raise HTTPException(404,'Original classification raster not found')
    export_id=str(uuid4());folder=_vector_export_folder(export_id);folder.mkdir(parents=True)
    record={'id':export_id,'run_id':run_id,'image_index':image_index,'status':'queued','progress':0,'stage':'Waiting to trace polygons','created_at':_now(),'format':'GeoJSON','polygon_count':0}
    _save_vector_export(folder,record);_vector_tasks.add(export_id)
    def work():
        def progress(value,stage,count):
            record.update(status='running',progress=value,stage=stage,polygon_count=count);_save_vector_export(folder,record)
        try:
            result=polygon_export(source,folder/'classes.geojson',report['classes'],{'mission_id':report['mission_id'],'job_id':report['job_id'],'source_run':run_id,'image_id':report['outputs'][image_index]['image_id'],'layer':'original_consensus'},progress)
            record.update(result,status='complete',progress=100,stage='Class polygons ready',download_url=f'/api/vector-exports/{export_id}/file')
        except Exception as exc:record.update(status='failed',stage='Vector export failed',error=str(exc))
        finally:_save_vector_export(folder,record);_vector_tasks.discard(export_id)
    async def queued():
        async with _vector_export_lock:
            await run_in_threadpool(work)
    asyncio.create_task(queued())
    return record


@app.get('/api/vector-exports/{export_id}')
async def vector_export_status(export_id: str):
    folder=_vector_export_folder(export_id)
    if not (folder/'status.json').is_file():raise HTTPException(404,'Vector export not found')
    record=json.loads((folder/'status.json').read_text())
    if record['status'] in ('queued','running') and export_id not in _vector_tasks:
        record.update(status='failed',error='Export interrupted by service restart. Original raster preserved; start a new export.');_save_vector_export(folder,record)
    return record


@app.get('/api/vector-exports/{export_id}/file')
async def vector_export_file(export_id: str):
    record=await vector_export_status(export_id)
    if record['status']!='complete':raise HTTPException(409,'Vector export is not complete')
    return FileResponse(_vector_export_folder(export_id)/'classes.geojson',media_type='application/geo+json',filename=f'classes-{record["run_id"]}-{record["image_index"]}.geojson')


@app.get('/api/reference-images')
async def reference_image(mission_id: str, job_id: str, image_id: str):
    from reference_labels import prepare_preview
    from starlette.concurrency import run_in_threadpool
    image,record=_reference_scope(mission_id,job_id,image_id)
    try:return await run_in_threadpool(run_reserved,'Source preview',2,prepare_preview,DATA_DIR,mission_id,job_id,image,record)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@app.get('/api/reference-images/{preview_id}/source.png')
async def reference_image_png(preview_id: str):
    from reference_labels import preview_path
    try:path=preview_path(DATA_DIR,preview_id)/'source.png'
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    if not path.is_file():raise HTTPException(404,'Open the source image first')
    return FileResponse(path,media_type='image/png')


class ImageLabelRequest(BaseModel):
    label: str = Field(min_length=1,max_length=80)
    kind: str = Field(pattern='^(rectangle|polygon|point)$')
    vertices: list[list[float]] = Field(max_length=2000)
    role: str = Field(default='train',pattern='^(train|validation)$')
    confidence: int = Field(default=3,ge=1,le=5)
    notes: str = Field(default='',max_length=500)


@app.post('/api/reference-images/{preview_id}/labels')
async def save_image_label(preview_id: str, request: ImageLabelRequest):
    from reference_labels import preview_path, fingerprint, label_geometry
    if not request.label.strip():raise HTTPException(422,'Enter a land-cover class')
    try:metadata=preview_path(DATA_DIR,preview_id)/'grid.json'
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    if not metadata.is_file():raise HTTPException(404,'Open the source image first')
    preview=json.loads(metadata.read_text())
    _,record=_reference_scope(preview['mission_id'],preview['job_id'],preview['image_id'])
    if fingerprint(record)!=preview['access_sha256']:raise HTTPException(409,'Source data changed. Reopen the image before saving a reference.')
    try:geometry=label_geometry(preview,request.vertices,request.kind)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    result=await _save_ground_truth(GroundTruthRequest(mission_id=preview['mission_id'],job_id=preview['job_id'],image_id=preview['image_id'],label=request.label.strip(),geometry=geometry,role=request.role,confidence=request.confidence,notes=request.notes),source={'type':'manual_image_reference','preview_id':preview_id,'access_sha256':preview['access_sha256'],'raster_access_snapshot':preview['source_record'],'grid':{k:preview[k] for k in ('crs','transform','width','height')},'display_bands':preview['bands'],'vertices':request.vertices,'kind':request.kind})
    return result


@app.get('/api/mission-thumbnail')
async def mission_thumbnail(mission_id: str, job_id: str, image_id: str):
    """Lazy bounded RGB or single-band preview for the saved raster revision."""
    from PIL import Image
    from starlette.concurrency import run_in_threadpool
    _init_database()
    with _database() as db:
        row=db.execute('SELECT payload FROM workspace WHERE id=1').fetchone()
        missions=json.loads(row['payload']).get('missions',[]) if row else []
        mission=next((m for m in missions if m['id']==mission_id),{})
        job=next((j for j in mission.get('jobs',[]) if j['id']==job_id),{})
        image=next((i for i in job.get('images',[]) if i['id']==image_id),None)
        access=db.execute('SELECT payload FROM raster_access WHERE mission_id=? AND job_id=? AND image_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1',
                          (mission_id,job_id,image_id)).fetchone()
    if not image or not access:raise HTTPException(404,'No readable source thumbnail available')
    record=json.loads(access['payload'])
    key=hashlib.sha256(json.dumps([mission_id,job_id,image_id,record],sort_keys=True).encode()).hexdigest()
    path=DATA_DIR/'previews'/f'{key}.png'
    def render():
        if path.is_file():return
        stack=read_stack(record,image,max_size=192)
        if not stack.bands:raise ValueError('No science band is available for a preview')
        display_bands=('RED','GREEN','BLUE')
        if not all(b in stack.bands for b in display_bands):
            display_bands=(stack.bands[0],)*3
        rgb=np.zeros((*stack.valid.shape,4),dtype='uint8')
        for channel,band in enumerate(display_bands):
            values=stack.pixels[stack.bands.index(band)];valid=stack.valid&np.isfinite(values)
            if valid.any():
                low,high=np.percentile(values[valid],[2,98])
                rgb[:,:,channel][valid]=128 if high-low<1e-6 else (255*np.clip((values[valid]-low)/(high-low),0,1)).astype('uint8')
        rgb[:,:,3]=stack.valid.astype('uint8')*255
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(key+'-'+uuid4().hex+'.png')
        Image.fromarray(rgb).save(temporary);temporary.replace(path)
    try:await run_in_threadpool(render)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    return FileResponse(path,media_type='image/png')


def _source_display(report, index):
    """Display-only RGB from this run's saved source, on the result extent."""
    import rasterio
    from rasterio.warp import transform_geom
    folder=_result_folder(report['id']);destination=folder/f'{index}-source-rgb.tif'
    if destination.is_file():return destination
    image_id=report['outputs'][index]['image_id']
    with _database() as db:
        workspace=json.loads(db.execute('SELECT payload FROM workspace WHERE id=1').fetchone()['payload'])
        mission=next((m for m in workspace['missions'] if m['id']==report['mission_id']),{})
        job=next((j for j in mission.get('jobs',[]) if j['id']==report['job_id']),{})
        image=next((i for i in job.get('images',[]) if i['id']==image_id),None)
        access=db.execute('SELECT payload FROM raster_access WHERE mission_id=? AND job_id=? AND image_id=? AND created_at<=? ORDER BY created_at DESC,rowid DESC LIMIT 1',
                          (report['mission_id'],report['job_id'],image_id,report.get('created_at','9999'))).fetchone()
    record=report.get('raster_access_snapshot',{}).get(str(image_id)) or (json.loads(access['payload']) if access else None)
    if not image or not record:raise HTTPException(409,'The saved source image is unavailable for this result')
    with rasterio.open(folder/f'{index}-consensus.tif') as src:
        box=src.bounds
        geometry=transform_geom(src.crs,'EPSG:4326',{'type':'Polygon','coordinates':[[[box.left,box.bottom],[box.right,box.bottom],[box.right,box.top],[box.left,box.top],[box.left,box.bottom]]]})
        factor=min(1,1024/max(src.width,src.height));width,height=max(1,round(src.width*factor)),max(1,round(src.height*factor))
        grid={'crs':src.crs,'transform':src.transform*rasterio.Affine.scale(src.width/width,src.height/height),'width':width,'height':height}
    stack=read_stack(record,image,geometry=geometry,target_profile=grid)
    if not all(band in stack.bands for band in ('RED','GREEN','BLUE')):
        raise HTTPException(409,'Choose the red, green and blue source band identities in Raster access to display the original image')
    rgb=np.zeros((3,height,width),dtype='uint8')
    for channel,band in enumerate(('RED','GREEN','BLUE')):
        values=stack.pixels[stack.bands.index(band)];valid=stack.valid&np.isfinite(values)
        if valid.any():
            low,high=np.percentile(values[valid],[2,98]);high=max(high,low+1e-6)
            rgb[channel]=np.where(valid,1+254*np.clip((values-low)/(high-low),0,1),0).astype('uint8')
    temporary=destination.with_name(f'.{destination.stem}-{uuid4().hex}.tif')
    try:
        with rasterio.open(temporary,'w',driver='GTiff',**grid,count=3,dtype='uint8',nodata=0) as dst:
            dst.write(rgb);dst.update_tags(purpose='Display-only RGB: per-band 2–98 percentile stretch. Original science pixels unchanged.')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _image_comparison_assets(report, index, layer):
    """Bounded display copies on one grid; never modify scientific rasters."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from PIL import Image
    folder=_result_folder(report['id'])
    source=_source_display(report,index)
    result=folder/f'{index}-{layer}-rgb.tif'
    if not result.is_file():raise HTTPException(404,'Result display layer is unavailable')
    # Rebuild the small display copies atomically, including after human review.
    with rasterio.open(source) as src, rasterio.open(result) as overlay:
        rgb=src.read(); valid=src.dataset_mask()>0
        original=np.dstack((rgb.transpose(1,2,0),valid.astype('uint8')*255))
        pixels=np.zeros((3,src.height,src.width),dtype='uint8')
        for band in range(3):
            reproject(source=rasterio.band(overlay,band+1),destination=pixels[band],src_transform=overlay.transform,src_crs=overlay.crs,dst_transform=src.transform,dst_crs=src.crs,resampling=Resampling.nearest)
        mask=np.zeros((src.height,src.width),dtype='uint8')
        reproject(source=overlay.dataset_mask(),destination=mask,src_transform=overlay.transform,src_crs=overlay.crs,dst_transform=src.transform,dst_crs=src.crs,resampling=Resampling.nearest)
        top=np.dstack((pixels.transpose(1,2,0),np.where(valid,mask,0)))
        width,height=src.width,src.height
    urls=[]
    for name,array in [(f'{index}-viewer-source.png',original),(f'{index}-viewer-{layer}.png',top)]:
        dest=folder/name;temp=folder/(uuid4().hex+'.png');Image.fromarray(array).save(temp);temp.replace(dest)
        urls.append(f"/api/results/{report['id']}/{name}?v={dest.stat().st_mtime_ns}")
    return {'source':urls[0],'overlay':urls[1],'width':width,'height':height}


@app.get('/api/results/{run_id}/{image_index}/viewer')
async def image_result_viewer(run_id: str,image_index: int,layer: str='consensus'):
    if layer not in ('consensus','review','agreement','disagreement','reviewed'):raise HTTPException(422,'Unknown result layer')
    report=await get_inference_run(run_id)
    if not 0<=image_index<len(report.get('outputs',[])):raise HTTPException(404,'Image result not found')
    output=report['outputs'][image_index]
    if layer in ('review','agreement','disagreement') and output.get('agreement_summary',{}).get('models_compared',0)<2:raise HTTPException(422,'Agreement requires at least two models')
    from starlette.concurrency import run_in_threadpool
    try:display=await run_in_threadpool(run_reserved,'Comparison preview',2,_image_comparison_assets,report,image_index,layer)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    return {'output':output,'classes':report.get('classes',[]),'display':display,'layer':layer}


@app.get('/api/results/{run_id}/{image_index}/viewer-location')
async def image_viewer_location(run_id: str,image_index: int,x: float,y: float):
    import rasterio
    from rasterio.warp import transform
    if not (0<=x<=1 and 0<=y<=1):raise HTTPException(422,'Location is outside the image')
    report=await get_inference_run(run_id)
    if not 0<=image_index<len(report.get('outputs',[])):raise HTTPException(404,'Image result not found')
    with rasterio.open(_result_folder(run_id)/f'{image_index}-consensus.tif') as src:
        px,py=src.transform*(min(x*src.width,src.width-0.000001),min(y*src.height,src.height-0.000001))
        lon,lat=transform(src.crs,'EPSG:4326',[px],[py])
    return {'longitude':lon[0],'latitude':lat[0]}


@app.post('/api/results/{run_id}/{image_index}/map')
async def review_result_on_map(run_id: str, image_index: int, layer: str = 'review'):
    if layer not in ('consensus','disagreement','agreement','review','reviewed') or image_index<0:
        raise HTTPException(422, 'Choose consensus, agreement, disagreement, review, or reviewed')
    report=await get_inference_run(run_id)
    outputs=report.get('outputs',[])
    if image_index>=len(outputs):
        raise HTTPException(404, 'Image result not found')
    if layer in ('review','agreement','disagreement') and outputs[image_index].get('agreement_summary', {}).get('models_compared', 0) < 2:
        raise HTTPException(422, 'Agreement comparison requires at least two successful models')
    path=(DATA_DIR/'results'/run_id/f'{image_index}-{layer}-rgb.tif').resolve()
    if not path.is_relative_to((DATA_DIR/'results').resolve()) or not path.is_file():
        raise HTTPException(404, 'Result raster not found')
    state=await _ensure_map()
    current = _tool_data(await map_client.call('list_assets', {'map_id': state['map_id']}))
    # Only replace application-generated analysis overlays; preserve AOIs and source imagery.
    old_reviews = [asset['asset_id'] for asset in current.get('assets', [])
                   if str(asset.get('name','')).startswith(('SAM proposal · ','Anomaly result · ')) or any(str(asset.get('name', '')).startswith(f'Analysis {name} · ') for name in ('review','agreement','disagreement','consensus','reviewed'))]
    from starlette.concurrency import run_in_threadpool
    try:source_path = await run_in_threadpool(run_reserved,'Source map preview',2,_source_display, report, image_index)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    async with private_http_client(MC2_URL,timeout=180) as client:
        with source_path.open('rb') as stream:
            response=await client.post(f"{MC2_URL}/api/maps/{state['map_id']}/upload-geotiff",
                files={'file':(source_path.name,stream,'image/tiff')},
                data={'mode':'rgb','bands':'1,2,3','name':f"Original image · {outputs[image_index]['image_id']}"})
        if response.is_error:raise HTTPException(502,'Could not display the source image: '+response.text[:200])
        with path.open('rb') as stream:
            response=await client.post(f"{MC2_URL}/api/maps/{state['map_id']}/upload-geotiff",
                files={'file':(path.name,stream,'image/tiff')},
                data={'mode':'rgb','bands':'1,2,3','alpha':'0.65','name':f"Analysis {layer} · {outputs[image_index]['image_id']}"})
        if response.is_error:
            raise HTTPException(502,response.text[:300])
        overlay_id=response.json().get('asset_id')
        if not overlay_id:raise HTTPException(502,'Map service did not return the analysis layer identifier')
        _tool_data(await map_client.call('update_style',{'map_id':state['map_id'],'asset_id':overlay_id,'style':{'opacity':0.65}}))
    for asset_id in old_reviews:
        await map_client.call('delete_asset', {'map_id': state['map_id'], 'asset_id': asset_id})
    for asset in current.get('assets',[]):
        if str(asset.get('name','')).startswith('Original image · '):
            await map_client.call('delete_asset',{'map_id':state['map_id'],'asset_id':asset['asset_id']})
    await map_client.call('zoom',{'map_id':state['map_id'],'target':'bbox','bbox':outputs[image_index]['bbox']})
    output = outputs[image_index]
    if layer == 'reviewed':
        import rasterio
        with rasterio.open(DATA_DIR/'results'/run_id/f'{image_index}-reviewed.tif') as source:
            values=source.read(1);valid=(values>0)&(source.read_masks(1)>0);total=int(valid.sum())
        output['class_distribution']=[{**item,'pixels':int(((values==item['value'])&valid).sum()),
            'percent':round(100*int(((values==item['value'])&valid).sum())/total,2) if total else None} for item in output['class_legend']]
        output['classification_summary']['predicted_classes']=sum(item['pixels']>0 for item in output['class_distribution'])
    return {'ok':True,'layer':layer,'map_url':state['url'], 'output': output, 'classes': report.get('classes',[])}


@app.post('/api/map/review/clear')
async def clear_analysis_map():
    state = await _ensure_map()
    current = _tool_data(await map_client.call('list_assets', {'map_id':state['map_id']}))
    for asset in current.get('assets', []):
        if str(asset.get('name','')).startswith(('SAM proposal · ','Anomaly result · ')) or any(str(asset.get('name','')).startswith(f'Analysis {name} · ') for name in ('review','agreement','disagreement','consensus','reviewed')):
            await map_client.call('delete_asset', {'map_id':state['map_id'],'asset_id':asset['asset_id']})
    return {'ok':True}


@app.get("/api/inference/runs")
async def list_inference_runs(mission_id: str | None = None, job_id: str | None = None):
    _init_database()
    query, values = "SELECT payload FROM inference_runs", []
    where = []
    if mission_id:
        where.append("mission_id = ?"); values.append(mission_id)
    if job_id:
        where.append("job_id = ?"); values.append(job_id)
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY created_at DESC"
    with _database() as conn:
        return {"runs": [_honest_report(json.loads(row["payload"])) for row in conn.execute(query, values)]}


@app.post("/api/ground-truth")
async def add_ground_truth(request: GroundTruthRequest):
    return await _save_ground_truth(request)


async def _save_ground_truth(request: GroundTruthRequest, source=None):
    _init_database()
    if not request.geometry or request.geometry.get('type') not in ('Point','Polygon','MultiPolygon'):
        raise HTTPException(422, 'Ground truth requires a point or polygon geometry')
    with _database() as conn:
        row=conn.execute('SELECT payload FROM workspace WHERE id=1').fetchone()
    missions=json.loads(row['payload']).get('missions',[]) if row else []
    mission=next((m for m in missions if m.get('id')==request.mission_id),None)
    job=next((j for j in (mission or {}).get('jobs',[]) if j.get('id')==request.job_id),None)
    if not job or (request.image_id and request.image_id not in {image['id'] for image in job.get('images',[])}):
        raise HTTPException(404, 'Ground truth must belong to an existing job and image')
    record = {"id": str(uuid4()), **request.model_dump(), "created_at": _now(), "origin":"human_reference"}
    if source:record["source"]=source
    with _database() as conn:
        conn.execute("INSERT INTO ground_truth (id, mission_id, job_id, payload, created_at) VALUES (?, ?, ?, ?, ?)", (record["id"], record["mission_id"], record["job_id"], json.dumps(record), record["created_at"]))
    if globals().get('workflow_store'):workflow_store.event('reference.saved','Human-reviewed reference saved',mission_id=request.mission_id,reference_id=record['id'],image_id=request.image_id,role=request.role,audit=True)
    record['automatic_training']=await _check_auto_training(request.mission_id,request.job_id)
    return record


@app.get("/api/ground-truth")
async def list_ground_truth(mission_id: str, job_id: str):
    _init_database()
    with _database() as conn:
        rows = conn.execute("SELECT payload FROM ground_truth WHERE mission_id = ? AND job_id = ? ORDER BY created_at DESC", (mission_id, job_id))
        return {"labels": [json.loads(row["payload"]) for row in rows]}


@app.post('/api/vector-imports/preview')
async def preview_vector_import(files: list[UploadFile] = File(...), dataset: str = Form(''), source_crs: str = Form(''), encoding: str = Form('')):
    from starlette.concurrency import run_in_threadpool
    if len(files)>20:raise HTTPException(422,'Choose one ZIP or up to 20 matching shapefile components')
    sources=[];total=0
    for upload in files:
        chunks=[]
        while chunk:=await upload.read(1024*1024):
            total+=len(chunk)
            if total>MAX_BYTES:raise HTTPException(413,'Shapefile upload exceeds 64 MiB')
            chunks.append(chunk)
        sources.append((upload.filename or 'upload',b''.join(chunks)))
    try:
        metadata,collection=await run_in_threadpool(read_polygons,sources,dataset,source_crs,encoding)
    except Exception as exc:
        raise HTTPException(422,f'Could not read shapefile: {exc}') from exc
    import_id=str(uuid4());folder=import_folder(DATA_DIR,import_id);folder.mkdir(parents=True)
    source_files=[]
    for index,(name,contents) in enumerate(sources):
        saved=f'upload-{index}{Path(name).suffix.lower()}'
        (folder/saved).write_bytes(contents)
        source_files.append({'filename':name,'saved_as':saved,'bytes':len(contents)})
    metadata.update(id=import_id,created_at=_now(),source_files=source_files)
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
    if collection:(folder/'polygons.geojson').write_text(json.dumps(collection))
    return metadata


@app.post('/api/vector-imports/{import_id}/map')
async def preview_vector_on_map(import_id: str):
    try:folder=import_folder(DATA_DIR,import_id)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    if not (folder/'polygons.geojson').is_file():raise HTTPException(404,'Read a polygon dataset first')
    metadata=json.loads((folder/'metadata.json').read_text())
    state=await _ensure_map()
    result=_tool_data(await map_client.call('add_geojson',{'map_id':state['map_id'],
        'geojson':(folder/'polygons.geojson').read_text(),'name':f'Imported polygons · {metadata["dataset"]}',
        'style':{'fill_color':'#a58de0','fill_opacity':0.18,'stroke_color':'#c7b2ff','stroke_width':2}}))
    await map_client.call('zoom',{'map_id':state['map_id'],'target':'bbox','bbox':metadata['bbox']})
    return {'ok':True,'asset_id':result.get('asset_id'),'feature_count':metadata['feature_count']}


@app.post('/api/vector-imports/{import_id}/commit')
async def commit_vector_import(import_id: str, request: VectorImportRequest):
    try:folder=import_folder(DATA_DIR,import_id)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    if not (folder/'polygons.geojson').is_file():raise HTTPException(404,'Read a polygon dataset first')
    metadata=json.loads((folder/'metadata.json').read_text())
    collection=json.loads((folder/'polygons.geojson').read_text())
    _init_database()
    with _database() as conn:
        workspace=conn.execute('SELECT payload FROM workspace WHERE id=1').fetchone()
    missions=json.loads(workspace['payload']).get('missions',[]) if workspace else []
    mission=next((m for m in missions if m['id']==request.mission_id),{})
    job=next((j for j in mission.get('jobs',[]) if j['id']==request.job_id),None)
    if not job or (request.image_id and request.image_id not in {image['id'] for image in job.get('images',[])}):
        raise HTTPException(404,'Choose an existing job and image')
    if request.class_field and request.class_field not in metadata['fields']:raise HTTPException(422,'Choose a class attribute from the shapefile')
    if request.class_field and metadata['field_value_counts'][request.class_field]>200:raise HTTPException(422,'Choose a class field with at most 200 distinct values')
    if not request.class_field and not request.fixed_label.strip():raise HTTPException(422,'Choose a class field or enter one class for all polygons')
    signature=hashlib.sha256(json.dumps({'sha256':metadata['sha256'],**request.model_dump()},sort_keys=True).encode()).hexdigest()
    with _database() as conn:
        previous=conn.execute('SELECT payload FROM vector_import_commits WHERE mission_id=? AND job_id=? AND scope=?',(request.mission_id,request.job_id,request.image_id or '*')).fetchall()
    if any(json.loads(row['payload']).get('signature')==signature for row in previous):raise HTTPException(409,'These source polygons and class mappings have already been imported into this image scope')
    records=[];skipped=0;created=_now();class_counts={}
    for feature in collection['features']:
        raw=feature['properties'].get(request.class_field) if request.class_field else request.fixed_label
        label=request.class_mapping.get(str(raw),str(raw)) if raw is not None else ''
        label=label.strip()
        if not label:
            if request.skip_unlabelled:skipped+=1;continue
            raise HTTPException(422,f'Feature {feature["id"]} has no class. Supply a label or enable skipping unlabelled polygons.')
        if len(label)>80:raise HTTPException(422,'Class names must be at most 80 characters')
        records.append({'id':str(uuid4()),'mission_id':request.mission_id,'job_id':request.job_id,'image_id':request.image_id,
            'label':label,'geometry':feature['geometry'],'role':request.role,'confidence':request.confidence,'notes':request.notes,
            'origin':'human_reference','created_at':created,'source':{'type':'external_shapefile','import_id':import_id,
            'dataset':metadata['dataset'],'feature_id':feature['id'],'source_crs':metadata['source_crs'],
            'class_field':request.class_field,'original_properties':feature['properties'],'sha256':metadata['sha256']}})
        class_counts[label]=class_counts.get(label,0)+1
    if not records:raise HTTPException(422,'No labelled polygons remain to import')
    summary={'id':import_id,'signature':signature,'created_at':created,'labels':len(records),'class_counts':class_counts,
             'skipped_unlabelled':skipped,'skipped_invalid':metadata.get('invalid_count',0),'role':request.role,'image_id':request.image_id}
    try:
        with _database() as conn:
            conn.execute('INSERT INTO vector_import_commits VALUES(?,?,?,?,?)',(import_id,request.mission_id,request.job_id,request.image_id or '*',json.dumps(summary)))
            conn.executemany('INSERT INTO ground_truth VALUES(?,?,?,?,?)',[(r['id'],r['mission_id'],r['job_id'],json.dumps(r),r['created_at']) for r in records])
    except sqlite3.IntegrityError as exc:raise HTTPException(409,'This import is already saved for this image scope') from exc
    automatic=await _check_auto_training(request.mission_id,request.job_id)
    return {'summary':summary,'labels':records,'automatic_training':automatic}


@app.post("/api/fine-tuning/runs")
async def create_fine_tune_run(request: FineTuneRequest):
    return await _create_training_version(request)


async def _create_training_version(request: FineTuneRequest, evidence: dict | None = None, policy: dict | None = None, workflow_profile_id: str | None = None):
    _init_database()
    if request.model not in MODEL_DETAILS or not algorithm_training_rule(request.model)['enabled']:
        raise HTTPException(422,'Choose a registered trainable algorithm')
    with _database() as conn:
        rows = conn.execute("SELECT payload FROM ground_truth WHERE mission_id = ? AND job_id = ?", (request.mission_id, request.job_id)).fetchall()
    labels = [json.loads(row["payload"]) for row in rows]
    selected = [label for label in labels if not request.label_ids or label["id"] in request.label_ids]
    classes = sorted({label["label"] for label in selected if label.get('role','train')=='train'})
    if len(classes)<2:
        raise HTTPException(422, 'At least two reviewed classes are required before training')
    # The inference queue trains a new version from reviewed labels and persists
    # model checkpoints and validation metrics beside its prediction maps.
    metadata={'run_kind':'training','model':request.model,'model_name':MODEL_DETAILS[request.model]['name'],
              'training_mode':algorithm_training_rule(request.model)['mode'],
              'trigger':'correction_threshold' if evidence else 'manual',
              'candidate_version':True,'message':'Training is happening in a separate version. Watch its live progress.',
              'notification_policy':'training_started_only'}
    if evidence:
        metadata['correction_evidence']={key:value for key,value in evidence.items() if key!='label_ids'}
        metadata['training_policy']=policy
    affected=[]
    if globals().get('workflow_store'):
        affected=[p for p in workflow_store.all('workflow_models',request.mission_id) if p['job_id']==request.job_id and p['algorithm']==request.model and p['state']!='deleted' and not p.get('builtin') and (not workflow_profile_id or p['id']==workflow_profile_id)]
        for p in affected:
            if p['state']=='qualified':workflow_store.retract(p['id'],'New training version started')
    run=await _queue_inference_run(InferenceRequest(mission_id=request.mission_id,job_id=request.job_id,
        algorithms=[request.model],label_ids=request.label_ids,epochs=request.epochs,max_size=request.max_size,
        band_alignment=not MODEL_DETAILS[request.model].get('pretrained_lulc',False)), metadata, evidence)
    for p in affected:
        p.update(state='training',candidate_run=run['id']);workflow_store.put('workflow_models',p)
    return run


async def _check_auto_training(mission_id: str, job_id: str):
    from starlette.concurrency import run_in_threadpool
    if not mission_control.allowed(_database, mission_id):
        return {'status':'mission_suspended','runs':[]}
    try:
        async with auto_training_lock:
            readiness=await run_in_threadpool(retraining.evaluate,_database,DATA_DIR,mission_id,job_id)
            queued=[]
            for item in readiness['algorithms']:
                if not item['ready']:
                    continue
                try:
                    run=await _create_training_version(FineTuneRequest(mission_id=mission_id,job_id=job_id,model=item['model_id'],label_ids=item['label_ids']),item,readiness['policy'])
                    queued.append(run)
                except sqlite3.IntegrityError:
                    # Another request/process already reserved this exact evidence batch.
                    continue
            return {'status':'training_started' if queued else 'collecting_evidence','runs':queued}
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception('Automatic training evaluation failed for %s/%s',mission_id,job_id)
        with _database() as conn:
            conn.execute('INSERT INTO settings(key,payload) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload',
                         (f'retraining_error:{mission_id}:{job_id}',json.dumps({'message':str(exc),'created_at':_now()})))
        return {'status':'evaluation_deferred','runs':[]}


@app.get('/api/training/policy')
async def training_policy():
    _init_database()
    with _database() as conn:
        return retraining.get_policy(conn)


@app.get('/api/training/readiness')
async def training_readiness(mission_id: str, job_id: str):
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(retraining.evaluate,_database,DATA_DIR,mission_id,job_id)


def _phase_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[int, int, float]:
    """Estimate an integer band shift with phase correlation, robust to brightness changes."""
    size = max(reference.shape)
    stride = max(1, int(np.ceil(size / 1024)))
    ref, mov = reference[::stride, ::stride].astype(np.float32), moving[::stride, ::stride].astype(np.float32)
    ref = np.nan_to_num(ref - np.nanmean(ref)); mov = np.nan_to_num(mov - np.nanmean(mov))
    cross = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    cross /= np.maximum(np.abs(cross), 1e-9)
    surface = np.abs(np.fft.ifft2(cross))
    peak_y, peak_x = np.unravel_index(np.argmax(surface), surface.shape)
    if peak_y > ref.shape[0] // 2: peak_y -= ref.shape[0]
    if peak_x > ref.shape[1] // 2: peak_x -= ref.shape[1]
    confidence = float(surface.max() / max(surface.mean(), 1e-9))
    return int(peak_y * stride), int(peak_x * stride), confidence


def _shift_without_wrap(band: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    output = np.zeros_like(band)
    source_y = slice(max(0, -shift_y), min(band.shape[0], band.shape[0] - shift_y))
    source_x = slice(max(0, -shift_x), min(band.shape[1], band.shape[1] - shift_x))
    destination_y = slice(max(0, shift_y), min(band.shape[0], band.shape[0] + shift_y))
    destination_x = slice(max(0, shift_x), min(band.shape[1], band.shape[1] + shift_x))
    output[destination_y, destination_x] = band[source_y, source_x]
    return output


@app.post("/api/band-alignment")
async def align_bands(file: UploadFile = File(...), reference_band: int = Form(1)):
    """Align a local multiband GeoTIFF to one reference band using phase correlation."""
    if Path(file.filename or "").suffix.lower() not in (".tif", ".tiff"):
        raise HTTPException(415, "Band alignment accepts a multiband GeoTIFF (.tif or .tiff)")
    try:
        import rasterio
        from rasterio.io import MemoryFile
        content = await file.read()
        with MemoryFile(content) as memory:
            with memory.open() as source:
                if source.count < 2:
                    raise HTTPException(422, "The GeoTIFF must contain at least two bands")
                if not 1 <= reference_band <= source.count:
                    raise HTTPException(422, f"Reference band must be between 1 and {source.count}")
                stack = source.read()
                reference = stack[reference_band - 1]
                aligned = stack.copy()
                shifts = []
                for index in range(source.count):
                    if index == reference_band - 1:
                        shifts.append({"band": index + 1, "shift_y_px": 0, "shift_x_px": 0, "confidence": None, "reference": True})
                        continue
                    shift_y, shift_x, confidence = _phase_shift(reference, stack[index])
                    aligned[index] = _shift_without_wrap(stack[index], shift_y, shift_x)
                    shifts.append({"band": index + 1, "shift_y_px": shift_y, "shift_x_px": shift_x, "confidence": round(confidence, 2), "reference": False})
                ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
                artifact = f"aligned-{uuid4().hex[:12]}.tif"
                output = ARTIFACT_DIR / artifact
                profile = source.profile.copy(); profile.update(compress="lzw")
                with rasterio.open(output, "w", **profile) as destination:
                    destination.write(aligned)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Could not align GeoTIFF bands: {exc}") from exc
    return {"ok": True, "artifact": artifact, "download_url": f"/api/artifacts/{artifact}", "reference_band": reference_band, "method": "phase correlation", "shifts": shifts}


@app.get("/api/artifacts/{artifact}")
async def get_artifact(artifact: str):
    path = ARTIFACT_DIR / Path(artifact).name
    if not path.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path, filename=path.name, media_type="image/tiff")


@app.on_event("shutdown")
async def shutdown():
    await map_client.close()

# Separate output family: anomaly scores never become LULC class votes.
check_anomaly_training = anomaly_api.register(app, _database, DATA_DIR, model_executor,
    map_client, _ensure_map, _tool_data, MC2_URL)
layer_insights.register(app, _database, DATA_DIR, map_client, _ensure_map, _tool_data, MC2_URL, model_executor)
basemap_manager.register(app, _database, DATA_DIR, map_client, _ensure_map, _tool_data, MC2_URL)
search_local_imagery, attach_local_imagery = local_imagery.register(app, _database, LOCAL_RASTER_ROOT)


# Bootstrap only from immutable, explicitly reviewed Dynamic World predictions.
import pseudo_labels
pseudo_labels.register(app,_database,DATA_DIR,get_inference_run,_result_folder,_init_database,_check_auto_training)

import workflow_api
workflow_store=workflow_api.register(sys.modules[__name__])
mission_control.PROGRESS_OBSERVER=workflow_store.run_changed

import workflow_engine
workflow_engine.register(sys.modules[__name__],workflow_store)

import catalog_downloads
catalog_downloads.register(sys.modules[__name__])

import object_labels
object_labels.register(sys.modules[__name__])

# Supporting services never add a fourth mission workspace.
import agent_services
agent_services.register(sys.modules[__name__])
import distributed_queue
distributed_queue.register(sys.modules[__name__])
import distributed_execution
distributed_execution.register(sys.modules[__name__])
import catalog_connections
catalog_connections.register(sys.modules[__name__])
