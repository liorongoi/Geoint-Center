"""Reviewed offline runtime for data-only, fixed linear LULC classifiers."""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from algorithm_registry import PORTABLE_MODEL as MODEL, PORTABLE_DETAILS as DETAILS

SCHEMA = 'geo-intelligence-linear-lulc-v1'
MAX_BYTES = 256 * 1024
MAX_ELEMENTS = 32 * 1024**2


def read(path, expected):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Portable model exceeds its size limit or its checksum changed')
    return raw, parse(raw)


def parse(raw):
    def unique(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError('Duplicate portable model fields')
        return dict(pairs)

    try:
        spec = json.loads(raw.decode('utf-8'), object_pairs_hook=unique)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('Portable model must be a bounded UTF-8 JSON object') from exc
    if not isinstance(spec, dict) or set(spec) != {
        'schema', 'classes', 'sensors', 'processing_levels', 'bands', 'ranges', 'coefficients', 'intercepts'
    } or spec['schema'] != SCHEMA:
        raise ValueError(f'Expected the reviewed {SCHEMA} runtime contract')

    def names(key, limit, minimum, predicate):
        values = spec[key]
        if (not isinstance(values, list) or not minimum <= len(values) <= limit
                or not all(isinstance(v, str) and predicate(v) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'Invalid or duplicate portable model {key}')
        return values

    classes = names('classes', 32, 2, lambda v: 1 <= len(v) <= 80 and v.strip() == v
                    and v not in {'accuracy', 'macro avg', 'micro avg', 'weighted avg', 'samples avg'}
                    and all(ch.isprintable() for ch in v))
    bands = names('bands', 64, 1, lambda v: bool(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}', v)))
    names('sensors', 16, 1, lambda v: bool(re.fullmatch(r'[a-z0-9][a-z0-9_.-]{0,79}', v)))
    names('processing_levels', 16, 1, lambda v: bool(re.fullmatch(r'[A-Z0-9][A-Z0-9_.-]{0,39}', v)))

    def vector(values, size):
        return (isinstance(values, list) and len(values) == size
                and all(type(v) in (int, float) and -1e9 <= v <= 1e9 and math.isfinite(v) for v in values))

    weights, limits = spec['coefficients'], spec['ranges']
    if (not vector(spec['intercepts'], len(classes)) or not isinstance(weights, list)
            or len(weights) != len(classes) or not all(vector(row, len(bands)) for row in weights)):
        raise ValueError('Expected one finite coefficient row and intercept per class')
    if (not isinstance(limits, list) or len(limits) != len(bands)
            or not all(vector(pair, 2) and pair[0] < pair[1] for pair in limits)):
        raise ValueError('Declare one finite, increasing input range per band in the raster engine’s scaled units')
    return spec


def predict(spec, stack, sensor, processing_level, progress):
    if sensor not in spec['sensors']:
        raise ValueError('Input sensor is outside the portable model contract: ' + ', '.join(spec['sensors']))
    level=str(processing_level or 'UNSPECIFIED').strip().upper()
    if level not in spec['processing_levels']:
        raise ValueError('Input processing level is outside the portable model contract: ' + ', '.join(spec['processing_levels']))
    if any(stack.bands.count(name) != 1 for name in spec['bands']):
        raise ValueError('Portable model requires unique mapped bands: ' + ', '.join(spec['bands']))
    valid = np.flatnonzero(stack.valid)
    weights = np.asarray(spec['coefficients'], dtype='float64')
    bias = np.asarray(spec['intercepts'], dtype='float64')
    bands = [stack.pixels[stack.bands.index(name)].ravel() for name in spec['bands']]
    prob = np.zeros((len(bias), stack.valid.size), dtype='float32')
    for start in range(0, len(valid), 32768):
        ids = valid[start:start+32768]
        data = np.stack([band[ids] for band in bands], axis=1)
        if not np.isfinite(data).all():
            raise ValueError('Non-finite values entered the portable model valid-pixel mask')
        for index, (low, high) in enumerate(spec['ranges']):
            if data[:, index].min() < low or data[:, index].max() > high:
                raise ValueError(f"Band {spec['bands'][index]} falls outside its declared input range")
        scores = data @ weights.T + bias
        scores -= scores.max(axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=1, keepdims=True)
        prob[:, ids] = scores.T
        progress('CPU pixel classification', min(100, 100*(start+len(ids))//len(valid)))
    predicted = prob.argmax(axis=0).astype('int32').reshape(stack.valid.shape)
    predicted[~stack.valid] = -1
    return predicted, prob.reshape((len(bias), *stack.valid.shape))
