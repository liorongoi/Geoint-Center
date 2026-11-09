"""Numerical contract for synchronized U-Net training across internal workers.

The coordinator owns weights and optimizer state. Workers compute sums of
gradients on disjoint fitting pixels. A round changes weights only after one
verified contribution from every partition. The worker has no network client.
"""
from pathlib import Path

import numpy as np

from model_workers import SEED, _unet

MAX_TENSOR_BYTES = 32 * 1024**2


def _torch_model(channels, classes):
    import torch

    if not 1 <= channels <= 256 or not 2 <= classes <= 256:
        raise ValueError('Unsupported U-Net channel or class count')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        return _unet(channels, classes)


def initial_state(channels, classes):
    model = _torch_model(channels, classes)
    return {name: tensor.detach().numpy().copy() for name, tensor in model.state_dict().items()}


def _matching(reference, tensors):
    if set(tensors) != set(reference):
        raise ValueError('U-Net tensor schema does not match the pinned architecture')
    for name, value in tensors.items():
        if value.dtype != np.float32 or value.shape != reference[name].shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid U-Net tensor: {name}')


def training_mask(labels, valid, held_out):
    if labels.shape != valid.shape or labels.shape != held_out.shape:
        raise ValueError('Fit and holdout masks must use the same analysis grid')
    if labels.dtype.kind not in ('i', 'u') or valid.dtype != bool or held_out.dtype != bool:
        raise ValueError('Use integer class codes and explicit Boolean validity and holdout masks')
    return np.where(valid & ~held_out, labels, -1).astype('int64')


def gradient(weights, pixels, mask, classes):
    """One forward/backward pass on already normalized science channels."""
    import torch
    import torch.nn.functional as functional

    from execution_policy import torch_device

    if pixels.ndim != 3 or pixels.dtype != np.float32 or not np.isfinite(pixels).all():
        raise ValueError('Expected finite, normalized float32 science channels')
    if (pixels.shape[1:] != mask.shape or min(mask.shape) < 4 or max(mask.shape) > 1024
            or mask.dtype != np.int64 or mask.min() < -1 or mask.max() >= classes):
        raise ValueError('Invalid U-Net training mask or analysis dimensions')
    count = int(np.count_nonzero(mask >= 0))
    if not count:
        raise ValueError('Partition contains no fitting pixels')
    model = _torch_model(pixels.shape[0], classes)
    _matching(model.state_dict(), weights)
    model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
    device = torch_device()
    model.to(device).train()
    logits = model(torch.from_numpy(pixels[None]).to(device))
    loss = functional.cross_entropy(logits, torch.from_numpy(mask[None]).to(device),
                                    ignore_index=-1, reduction='sum')
    loss.backward()
    gradients = {name: parameter.grad.detach().cpu().numpy().copy()
                 for name, parameter in model.named_parameters()}
    _matching(weights, gradients)
    total = float(loss.detach().cpu())
    if not np.isfinite(total) or total < 0:
        raise ValueError('Nonfinite U-Net loss')
    return gradients, count, total, str(device)


def apply_round(weights, velocity, partitions, contributions, *, learning_rate=.002, momentum=.9):
    """Validate every partition before producing the next weights and velocity."""
    _matching(weights, velocity)
    if len(partitions) < 2 or set(partitions) != set(contributions):
        raise ValueError('A synchronized round requires every scheduled partition')
    sums = {name: np.zeros(value.shape, dtype='float64') for name, value in weights.items()}
    pixels = 0
    loss = 0.
    for identifier, expected_count in partitions.items():
        gradients, count, total_loss = contributions[identifier]
        if count != expected_count or count <= 0 or not np.isfinite(total_loss) or total_loss < 0:
            raise ValueError(f'Unexpected fitting evidence for partition {identifier}')
        _matching(weights, gradients)
        for name, value in gradients.items():
            sums[name] += value
        pixels += count
        loss += total_loss
    next_velocity = {name: (momentum*velocity[name]+value/pixels).astype('float32') for name, value in sums.items()}
    next_weights = {name: value-learning_rate*next_velocity[name] for name, value in weights.items()}
    _matching(weights, next_velocity)
    _matching(weights, next_weights)
    return next_weights, next_velocity, loss/pixels


def write_tensors(path, tensors):
    """Store numeric tensors only; publish a new immutable filename."""
    import os
    import tempfile

    from safetensors.numpy import save_file

    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.neural-', dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        save_file(tensors, tmp)
        with tmp.open('rb') as stream:
            os.fsync(stream.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_tensors(path, reference):
    from safetensors.numpy import load

    path = Path(path)
    if path.is_symlink():
        raise ValueError('Invalid or oversized neural tensor artifact')
    with path.open('rb') as stream:
        encoded = stream.read(MAX_TENSOR_BYTES + 1)
    if len(encoded) > MAX_TENSOR_BYTES:
        raise ValueError('Invalid or oversized neural tensor artifact')
    tensors = load(encoded)
    _matching(reference, tensors)
    return tensors


def save_round(folder, round_number, weights, velocity):
    """Publish weights and optimizer state together at a completed barrier."""
    if type(round_number) is not int or round_number < 0:
        raise ValueError('Invalid neural round number')
    _matching(weights, velocity)
    path = Path(folder) / f'round-{round_number:04d}.safetensors'
    tensors = {f'weight/{name}': value for name, value in weights.items()}
    tensors.update({f'velocity/{name}': value for name, value in velocity.items()})
    write_tensors(path, tensors)
    return path


def load_round(path, reference):
    """Read a saved barrier without executable Python or partial optimizer data."""
    schema = {f'{prefix}/{name}': value
              for prefix in ('weight', 'velocity') for name, value in reference.items()}
    saved = read_tensors(path, schema)
    return tuple({name: saved[f'{prefix}/{name}'] for name in reference}
                 for prefix in ('weight', 'velocity'))
