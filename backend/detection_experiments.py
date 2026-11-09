"""Local rotated detection experiments and independent object validation."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from fewshot_dinov2 import SHA256, encoder, predict_vectors
from rotated_fewshot import oriented_chip

MODEL_ID = 'rotated_dinov2_exemplars'
NAME = 'Aircraft subtypes · revised few-shot'
CLASSES = ('jet', 'propeller')
BANK_SHA256 = '9496f1e042df60e157bb3451fba1e17e367174757314681ed308fe5da22ba5e8'
RUN_KIND = 'research_detection'


def pack(data):
    configured = os.getenv('GEO_AIRCRAFT_MODELS')
    if configured:
        return Path(configured)
    workspace = Path(data) / 'components/aircraft-dinov2'
    if workspace.exists():
        return workspace
    bundled = os.getenv('SPECTRASCOPE_BUNDLED_MODELS')
    if bundled:
        installed = Path(bundled).parent / 'components/aircraft-dinov2'
        if installed.is_dir():
            return installed
    installed = Path(sys.executable).resolve().parent.parent / 'components/aircraft-dinov2'
    return installed if installed.is_dir() else workspace


def availability(data):
    from object_detection import component

    if component() is None:
        return 'Install the offline Oriented R-CNN proposal component.'
    if not all(importlib.util.find_spec(name) for name in ('torch', 'timm', 'safetensors')):
        return 'Install the offline PyTorch, timm and SafeTensors runtime.'
    if not all((pack(data) / name).is_file() for name in ('model.safetensors', 'aircraft-exemplars.json')):
        return 'Install the offline aircraft exemplar and encoder pack.'
    return ''


def load(data):
    root = pack(data)
    bank_file = root / 'aircraft-exemplars.json'
    if bank_file.is_symlink() or bank_file.stat().st_size > 1024 * 1024:
        raise ValueError('Invalid aircraft exemplar file')
    raw = bank_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BANK_SHA256:
        raise ValueError('Aircraft exemplar bank differs from the tested offline version')
    bank = json.loads(raw)
    examples = np.asarray([bank['examples'][name] for name in CLASSES], dtype='float32')
    if bank['encoder_sha256'] != SHA256 or examples.shape != (2, 10, 768) or not np.isfinite(examples).all():
        raise ValueError('Invalid aircraft exemplar bank or encoder revision')
    embed, device = encoder(root / 'model.safetensors')

    classify = make_classifier(embed, CLASSES, examples, bank['neighbors'], bank['minimum_margin'])
    return classify, {
        'encoder_sha256': SHA256, 'exemplar_sha256': BANK_SHA256, 'encoder_device': device,
        'reference_source': bank['reference_source'], 'benchmark': bank['benchmark'],
        'shots_per_class': 10, 'neighbors': bank['neighbors'], 'minimum_margin': bank['minimum_margin'],
        'proposal_threshold': bank['proposal_threshold'],
    }


def audit(run, labels, exhaustive):
    """Match independent human boxes in native pixel space and explain misses."""
    from object_detection import prepare
    from rotated_fewshot import score_oriented

    prepare()
    from oriented_det.ops.iou import polygon_iou

    grid = run['grid']
    classes = run['classes']
    targets = []
    seen = {}
    for label in labels:
        if (label['label'] not in classes or label.get('role') != 'validation' or label.get('confidence', 0) < 3
                or label.get('source', {}).get('access_sha256') != run['source_access_sha256']):
            continue
        corners = reference_corners(label, grid)
        if not np.isfinite(corners).all() or (corners < -1e-6).any() or (corners > np.array([grid['width'], grid['height']])+1e-6).any():
            continue
        key = tuple(sorted(tuple(p) for p in np.round(corners, 5)))
        if key in seen:
            if seen[key] != label['label']:
                raise ValueError('The same validation box has conflicting class labels.')
            continue
        seen[key] = label['label']
        targets.append((label, corners))
    if not targets:
        raise ValueError('Add independent rotated validation boxes for the configured classes on this exact source image.')
    if len(targets) * len(run['detections']) > 500000:
        raise ValueError('This review exceeds the matching limit. Evaluate a smaller image.')
    predictions = [(np.asarray(item['properties']['pixel_corners']), item['properties']['label'],
                    item['properties']['subtype_margin'], item['id']) for item in run['detections']]
    matched = set()
    overlaps = [[] for _ in targets]
    ordered = sorted(predictions, key=lambda row: -row[2])
    for corners, name, _, identifier in ordered:
        candidates = []
        for index, (label, target) in enumerate(targets):
            iou = polygon_iou(corners, target, intersection_backend='python')
            if iou >= .5:
                overlaps[index].append({'detection_id': identifier, 'predicted_class': name, 'iou': iou})
                if label['label'] == name and index not in matched:
                    candidates.append((iou, index))
        if candidates:
            matched.add(max(candidates)[1])
    rows = []
    for index, (label, _) in enumerate(targets):
        hits = overlaps[index]
        result = 'matched' if index in matched else 'no usable box' if not hits else (
            'competing detection' if any(h['predicted_class'] == label['label'] for h in hits) else 'wrong subtype')
        rows.append({'label_id': label['id'], 'class': label['label'], 'result': result, 'overlaps': hits})
    counts = {name: {state: sum(r['class'] == name and r['result'] == state for r in rows)
                     for state in ('matched', 'no usable box', 'wrong subtype', 'competing detection')} for name in classes}
    metrics = baseline_metrics = None
    if exhaustive:
        scene = run['image_id']
        truth = {scene: [(corners, label['label']) for label, corners in targets]}
        metrics = score_oriented(truth, [(scene, corners, name, margin) for corners, name, margin, _ in predictions], classes)
        if run.get('candidate_fit'):
            baseline = [(scene, np.asarray(p['pixel_corners']), p['baseline_class'], p['baseline_margin'])
                        for item in run['detections'] if (p := item['properties'])['baseline_margin'] >= .05]
            baseline_metrics = score_oriented(truth, baseline, classes)
    guidance = {}
    for name, row in counts.items():
        if not sum(row.values()):
            guidance[name] = 'Add validation examples for this class.'
        elif row['no usable box'] and row['no usable box'] >= row['wrong subtype']:
            guidance[name] = 'Improve the proposal detector for this class or image scale. More exemplar labels alone cannot recover missing boxes.'
        elif row['wrong subtype']:
            guidance[name] = 'Label similar-looking objects and more varied examples on separate training images, then build another candidate.'
        elif row['competing detection']:
            guidance[name] = 'Review duplicate boxes and closely spaced objects before changing the classifier.'
        else:
            guidance[name] = 'Confirm these results on additional acquisitions reserved for validation.'
    comparison = {}
    if metrics:
        for name, current in metrics.items():
            baseline = baseline_metrics[name] if baseline_metrics else None
            comparison[name] = {'precision_target': .8, 'meets_precision_target': current['precision'] >= .8,
                                'recall_delta': current['recall'] - baseline['recall'] if baseline else None}
    return {'counts': counts, 'objects': rows, 'metrics': metrics, 'baseline_metrics': baseline_metrics,
            'comparison': comparison, 'next_steps': guidance, 'exhaustive_by_operator': exhaustive,
            'scope': 'This image only, IoU ≥ 0.50. No usable box includes missing, poorly localized, or filtered boxes.'}


def reference_corners(label, grid):
    from affine import Affine
    from rasterio.warp import transform

    source = label['source']['grid']
    forward = Affine(*source['transform'][:6])
    inverse = ~Affine(*grid['transform'][:6])
    xy = [forward * (x * source['width'], y * source['height']) for x, y in label['vertices']]
    x, y = transform(source['crs'], grid['crs'], *zip(*xy))
    return np.asarray([inverse * point for point in zip(x, y)])


def fit_candidate(data, references, evaluation_access, folder, progress):
    """Fit a new bank from human-reviewed support images, without fitting the encoder."""
    from affine import Affine
    from object_detection import display_rgb, detection_stretch, native_grid
    from raster_pipeline import read_stack
    from fewshot_dinov2 import fit_vectors

    from imagery_products import checksum
    from offline_policy import local_raster
    from raster_pipeline import source_assets

    evaluation_hashes = {checksum(Path(local_raster(path))) for _, path in source_assets(evaluation_access)}
    source_hashes = {}
    benchmark_root = (Path(data) / 'benchmarks').resolve()
    for label, image, access in references:
        if any(item.get('benchmark_split') in ('validation', 'test') for item in (label,image,access)):
            raise ValueError('Benchmark validation and test objects cannot fit a local candidate')
        if any(Path(local_raster(path)).resolve().is_relative_to(benchmark_root) for _, path in source_assets(access)):
            raise ValueError('Managed benchmark imagery is reserved for benchmark protocols')
        key = label['image_id']
        if key not in source_hashes:
            source_hashes[key] = {checksum(Path(local_raster(path))) for _, path in source_assets(access)}
        if evaluation_hashes & source_hashes[key]:
            raise ValueError('Training and evaluation share a source file. Use a different acquisition for validation.')
    embed, device = encoder(pack(data) / 'model.safetensors')
    vectors, labels = [], []
    for index, (label, image, access) in enumerate(references):
        progress(16 + 7 * index / len(references), f'Embedding approved training box {index + 1} of {len(references)}')
        grid, is_byte = native_grid(access)
        corners = reference_corners(label, grid)
        x, y = np.floor(corners.min(axis=0)).astype(int)
        right, bottom = np.ceil(corners.max(axis=0)).astype(int)
        if not (0 <= x < right <= grid['width'] and 0 <= y < bottom <= grid['height']
                and max(right - x, bottom - y) <= 1024):
            raise ValueError('Training box falls outside its native image or exceeds 1024 pixels')
        if is_byte:
            stretch = [{'band': band, 'low': 0, 'high': 255} for band in ('RED','GREEN','BLUE')]
        else:
            preview = read_stack(access, image, max_size=1024)
            stretch = detection_stretch(preview, False)
        profile = {**grid, 'width': right - x, 'height': bottom - y,
                   'transform': grid['transform'] * Affine.translation(x, y)}
        stack = read_stack(access, image, target_profile=profile)
        if not stack.valid.any():
            raise ValueError('Training box has no valid source pixels')
        rgb = display_rgb(stack, stretch)
        vectors.append(embed(Image.fromarray(rgb), [corners - [x, y]])[0])
        labels.append(label['label'])
    vectors = np.asarray(vectors)
    shots = len(labels) // len(set(labels))
    classes, centroids = fit_vectors(vectors, labels, shots)
    exemplars = np.stack([vectors[np.asarray(labels) == name] for name in classes])
    bank = {'classes': classes, 'shots_per_class': shots, 'neighbors': 3, 'minimum_margin': 0,
            'baseline_minimum_margin': .05, 'proposal_threshold': .4, 'encoder_sha256': SHA256,
            'encoder_device': device, 'evaluation_source_sha256': sorted(evaluation_hashes),
            'training_label_ids': [label['id'] for label, _, _ in references]}
    bank['source_sha256'] = {key: sorted(values) for key, values in source_hashes.items()}
    path = folder / 'candidate-bank.json'
    with path.open('x') as stream:
        json.dump({**bank, 'exemplars': exemplars.tolist()}, stream, allow_nan=False)
    bank['exemplar_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return make_classifier(embed, classes, exemplars, 3, 0, centroids), classes, bank


def make_classifier(embed, classes, examples, neighbors, minimum_margin, centroids=None):
    def classify(rgb, proposals, plane_label):
        image = Image.fromarray(rgb)
        eligible = []
        for proposal in proposals:
            if plane_label is not None and proposal['label'] != plane_label:
                continue
            box = list(proposal['rbox'].corners())
            try:
                oriented_chip(image, box, 16)
            except ValueError:
                continue
            eligible.append((proposal, box))
        if not eligible:
            return []
        vectors = embed(image, [box for _, box in eligible])
        results = predict_vectors(vectors, (classes, None), exemplars=examples, neighbors=neighbors)
        baseline = predict_vectors(vectors, (classes, centroids)) if centroids is not None else [(None, 0)] * len(results)
        return [{**proposal, 'subtype': name, 'subtype_margin': margin, 'baseline_class': old, 'baseline_margin': old_margin}
                for (proposal, _), (name, margin), (old, old_margin) in zip(eligible, results, baseline)
                if margin >= minimum_margin]
    return classify


def select_support(labels, evaluation_image, evaluation_fingerprint, count):
    """Reserve validation images entirely and select a balanced reviewed support set."""
    eligible = [r for r in labels if r.get('confidence', 0) >= 3]
    validation_images = {r['image_id'] for r in labels if r['role'] == 'validation'}
    groups = {}
    seen = set()
    for row in eligible:
        key = (row['image_id'], tuple(sorted(tuple(p) for p in row['vertices'])))
        if row['role'] != 'train' or row['image_id'] in validation_images | {evaluation_image} or key in seen:
            continue
        groups.setdefault(row['label'], []).append(row)
        seen.add(key)
    if not 2 <= len(groups) <= 8:
        raise ValueError('Use 2–8 object classes on training images, separate from all validation images.')
    shortages = [name for name, rows in groups.items() if len(rows) < count]
    if shortages:
        raise ValueError(f'Add at least {count} distinct approved training boxes for: ' + ', '.join(sorted(shortages)))
    evaluation_classes = {r['label'] for r in eligible if r['role'] == 'validation'
                          and r['image_id'] == evaluation_image
                          and r['source']['access_sha256'] == evaluation_fingerprint}
    if set(groups) - evaluation_classes:
        raise ValueError('Add independent validation boxes on the selected image for: '
                         + ', '.join(sorted(set(groups) - evaluation_classes)))
    return [row for name in sorted(groups) for row in groups[name][:count]]


def load_candidate(data, bank_path, expected_hash, evaluation_access):
    """Apply a previous immutable candidate to a new validation image."""
    from imagery_products import checksum
    from offline_policy import local_raster
    from raster_pipeline import source_assets
    from fewshot_dinov2 import fit_vectors

    if bank_path.is_symlink() or bank_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Invalid saved candidate bank')
    raw = bank_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('Saved candidate bank changed. Build a new candidate from approved labels.')
    bank = json.loads(raw)
    evaluation_hashes = {checksum(Path(local_raster(path))) for _, path in source_assets(evaluation_access)}
    if evaluation_hashes & set(bank['evaluation_source_sha256']):
        raise ValueError('This source was already used to evaluate the candidate. Select a different acquisition.')
    if evaluation_hashes & {h for values in bank['source_sha256'].values() for h in values}:
        raise ValueError('Training and evaluation share a source file. Use a different acquisition for validation.')
    classes = bank['classes']
    shots = bank['shots_per_class']
    examples = np.asarray(bank.pop('exemplars'), dtype='float32')
    if bank['encoder_sha256'] != SHA256 or examples.shape != (len(classes), shots, 768):
        raise ValueError('Saved candidate is incompatible with this encoder')
    _, centroids = fit_vectors(examples.reshape(-1, 768), [name for name in classes for _ in range(shots)], shots)
    embed, device = encoder(pack(data) / 'model.safetensors')
    bank.update(exemplar_sha256=expected_hash, encoder_device=device, evaluation_source_sha256=sorted(evaluation_hashes))
    return make_classifier(embed, classes, examples, bank['neighbors'], bank['minimum_margin'], centroids), classes, bank
