"""Map-scoped, persisted custom basemaps. Source URLs are fetched by the viewer only."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
import math
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

from ..config import load_config
from ..database import get_db


class BasemapDefinition(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    kind: Literal['raster', 'vector'] = 'raster'
    url: str = Field(min_length=1, max_length=4096)
    attribution: str = Field(default='', max_length=500)
    tile_size: int = Field(default=256, ge=64, le=2048)
    min_zoom: int = Field(default=0, ge=0, le=24)
    max_zoom: int = Field(default=22, ge=0, le=24)
    scheme: Literal['xyz', 'tms'] = 'xyz'
    tilejson: bool = False
    image_bounds: list[float] | None = Field(default=None, min_length=4, max_length=4)
    opacity: float = Field(default=1, ge=0, le=1)

    @field_validator('label')
    @classmethod
    def label_not_blank(cls, value):
        if not value.strip():
            raise ValueError('Give the basemap a name')
        return value.strip()

    @field_validator('url')
    @classmethod
    def browser_url(cls, value):
        if any(char.isspace() or ord(char) < 32 for char in value) or any(char in value for char in '\\<>"\''):
            raise ValueError('Use a valid HTTP(S) URL without credentials or control characters')
        if re.fullmatch(r'/api/files/basemap-[0-9a-f]{32}\.png', value):
            return value
        try:
            parsed = urlsplit(value)
            parsed.port
        except ValueError as exc:
            raise ValueError('Malformed basemap URL') from exc
        if parsed.scheme.lower() not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Basemap URLs must use HTTP(S), with no embedded username/password')
        if '{' in parsed.netloc or '}' in parsed.netloc or parsed.fragment:
            raise ValueError('Use a concrete tile host and remove the URL fragment')
        return value

    @field_validator('attribution')
    @classmethod
    def plain_credit(cls, value):
        return escape(value, quote=True)

    @model_validator(mode='after')
    def compatible(self):
        if self.min_zoom > self.max_zoom:
            raise ValueError('Minimum zoom must not exceed maximum zoom')
        if self.tile_size & (self.tile_size - 1):
            raise ValueError('Tile size must be a power of two')
        if self.image_bounds:
            w, s, e, n = self.image_bounds
            if not all(math.isfinite(v) for v in (w, s, e, n)) or not -180 <= w < e <= 180 or not -90 <= s < n <= 90:
                raise ValueError('Image bounds must be valid WGS84 west,south,east,north coordinates')
            if self.kind != 'raster' or self.tilejson:
                raise ValueError('A georeferenced image must be a raster basemap')
        return self


async def list_basemaps(map_id: str) -> dict:
    db = await get_db()
    row = await (await db.execute('SELECT config FROM maps WHERE id = ?', (map_id,))).fetchone()
    if row is None:
        raise ValueError('Map not found')
    config = json.loads(row['config'] or '{}')
    return config.get('custom_basemaps', {})


async def configure(map_id: str, key: str, definition: dict | None, remove: bool = False) -> dict:
    if not re.fullmatch(r'custom-[0-9a-f]{32}', key):
        raise ValueError('Custom basemap keys must be generated custom UUID identifiers')
    existing = await list_basemaps(map_id)
    if not remove and key not in existing and len(existing) >= 40:
        raise ValueError('This map already has 40 custom basemaps; remove an unused one first')
    db = await get_db()
    path = '$.custom_basemaps."' + key + '"'
    now = datetime.now(timezone.utc).isoformat()
    if remove:
        await db.execute("UPDATE maps SET config = json_remove(COALESCE(NULLIF(config, ''), '{}'), ?), updated_at = ? WHERE id = ?", (path, now, map_id))
        await db.commit()
        previous = existing.get(key, {})
        # Only delete files created by the bounded basemap-image upload route.
        if re.fullmatch(r'/api/files/basemap-[0-9a-f]{32}\.png', previous.get('url', '')):
            file = Path(load_config().storage.file_dir) / previous['url'].rsplit('/', 1)[-1]
            file.unlink(missing_ok=True)
        return {'key': key, 'removed': True}
    payload = BasemapDefinition.model_validate(definition or {}).model_dump(exclude_none=True)
    payload.update(group='data', thumbnail='')
    await db.execute("UPDATE maps SET config = json_set(COALESCE(NULLIF(config, ''), '{}'), ?, json(?)), updated_at = ? WHERE id = ?", (path, json.dumps(payload), now, map_id))
    await db.commit()
    return {'key': key, 'definition': payload}
