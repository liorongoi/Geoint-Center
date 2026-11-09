"""File serving and GeoTIFF upload endpoints."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..config import load_config
from ..services import geotiff_service, event_service, session_service, asset_service
from ..services.geotiff_service import (
    GeoTIFFError,
    GeoTIFFResult,
    _bounds_to_geojson_polygon,
)
from ..models import AssetStyle, MapEvent, MapEventResponse
from ..websocket import manager

router = APIRouter(tags=["files"])


@router.post('/api/maps/{map_id}/basemap-image')
async def save_basemap_image(map_id: str, file: UploadFile = File(...)):
    """Store a small prepared display PNG without decoding a source raster."""
    import io
    from PIL import Image
    if await session_service.get_map(map_id) is None:
        raise HTTPException(404, 'Map not found')
    raw = bytearray()
    try:
        while chunk := await file.read(1024 * 1024):
            raw.extend(chunk)
            if len(raw) > 8 * 1024 * 1024:
                raise HTTPException(413, 'Basemap preview exceeds 8 MiB')
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != 'PNG' or image.mode not in ('RGB', 'RGBA') or max(image.size) > 1024:
                    raise ValueError('Use an RGB/RGBA PNG no larger than 1024 pixels per side')
                image.verify()
        except (ValueError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, str(exc)) from exc
        filename = 'basemap-' + uuid.uuid4().hex + '.png'
        directory = Path(load_config().storage.file_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_bytes(raw)
        return {'image_url': '/api/files/' + filename}
    finally:
        await file.close()


# ─── Serve rendered files ────────────────────────────────────────────────────

@router.get("/api/files/{filename}")
async def serve_file(filename: str):
    """Serve a rendered file (e.g., GeoTIFF PNG) from the file storage directory."""
    config = load_config()
    file_dir = Path(config.storage.file_dir)
    file_path = file_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Security: ensure the resolved path is within file_dir
    try:
        file_path.resolve().relative_to(file_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    # Determine content type
    suffix = file_path.suffix.lower()
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".json": "application/json",
    }
    media_type = content_types.get(suffix, "application/octet-stream")

    return FileResponse(
        str(file_path),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ─── GeoTIFF Upload ─────────────────────────────────────────────────────────

@router.post("/api/maps/{map_id}/upload-geotiff", status_code=201)
async def upload_geotiff(
    map_id: str,
    file: UploadFile = File(...),
    mode: str = Form("singleband"),  # "rgb" or "singleband"
    name: str | None = Form(None),
    bands: str = Form("1,2,3"),  # Comma-separated band indices for RGB
    band: int = Form(1),  # Band index for singleband
    colormap: str = Form("viridis"),
    alpha: float = Form(1.0),
    vmin: float | None = Form(None),
    vmax: float | None = Form(None),
    percentile_min: float | None = Form(None),
    percentile_max: float | None = Form(None),
    nodata: float | None = Form(None),
    user_session_id: str | None = Form(None),
):
    """Upload a GeoTIFF file and add it as a raster overlay on the map.

    Supports two modes:
    - **rgb**: 3-band composite (specify bands as comma-separated, e.g. "1,2,3")
    - **singleband**: Single band with colormap (viridis, magma, plasma, inferno, gray, etc.)
    """
    # Verify map exists
    info = await session_service.get_map(map_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Map not found")

    config = load_config()
    max_bytes = config.geotiff.max_file_size_mb * 1024 * 1024

    # Keep staged inputs inside the renderer's approved storage, outside
    # the single-file public route. The upload is deleted after processing.
    incoming = Path(config.storage.file_dir) / "_upload"
    incoming.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", dir=incoming, delete=False)
    total = 0
    try:
        while chunk := await file.read(65536):
            total += len(chunk)
            if total > max_bytes:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"This GeoTIFF is too large ({total / (1024*1024):.1f} MB so far). "
                        f"Maximum supported size is {config.geotiff.max_file_size_mb} MB for direct rendering. "
                        f"{config.geotiff.tile_server_message}"
                    ),
                )
            tmp.write(chunk)
        tmp.close()
    except HTTPException:
        raise
    except Exception:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise

    # Generate asset ID
    asset_id = str(uuid.uuid4())

    try:
        if mode == "rgb":
            band_list = [int(b.strip()) for b in bands.split(",")]
            result = await geotiff_service.process_geotiff_rgb(
                source=tmp.name,
                asset_id=asset_id,
                bands=band_list,
                alpha=alpha,
                nodata=nodata,
                config=config,
            )
        elif mode == "singleband":
            result = await geotiff_service.process_geotiff_singleband(
                source=tmp.name,
                asset_id=asset_id,
                band=band,
                colormap=colormap,
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
                percentile_min=percentile_min,
                percentile_max=percentile_max,
                nodata=nodata,
                config=config,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{mode}'. Use 'rgb' or 'singleband'.",
            )

        if isinstance(result, GeoTIFFError):
            status = 413 if result.error == "file_too_large" else 400
            raise HTTPException(status_code=status, detail=result.detail)

        # Create asset in DB (store bounds as GeoJSON polygon for bbox computation)
        # Use the same asset_id that was used for the PNG filename
        bounds_geojson = _bounds_to_geojson_polygon(result.bounds)
        asset_type = f"geotiff_{mode}"
        asset = await asset_service.create_asset(
            map_id=map_id,
            asset_type=asset_type,
            geojson=json.dumps(bounds_geojson),
            name=name or file.filename or f"GeoTIFF ({mode})",
            source_url=f"upload:{file.filename}",
            asset_id=asset_id,
        )

        # Broadcast to all sessions
        broadcast_msg = {
            "type": f"add_geotiff_{mode}",
            "event_id": str(uuid.uuid4()),
            "asset_id": asset.asset_id,
            "data": {
                "asset_id": asset.asset_id,
                "image_url": result.image_url,
                "bounds": result.bounds,
                "name": name or file.filename,
                "alpha": alpha,
                "width": result.width,
                "height": result.height,
                "crs": result.crs,
                "band_count": result.band_count,
            },
        }
        await manager.broadcast_to_map(map_id, broadcast_msg)

        return {
            "asset_id": asset.asset_id,
            "image_url": result.image_url,
            "bounds": result.bounds,
            "width": result.width,
            "height": result.height,
            "crs": result.crs,
            "band_count": result.band_count,
            "name": name or file.filename,
        }

    finally:
        # Clean up temp file
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
