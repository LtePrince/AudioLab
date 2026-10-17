"""Bounded, standard-library transport for the three local inference weights.

No checkpoint is deserialized here. File integrity does not certify model
compatibility or musical quality; the runtime checks the model architecture.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import zipfile

from src.data.timing import TIMING_SCHEMA

SCHEMA = 'audiolab-model-bundle-v1'
MANIFEST = 'model_bundle.json'
FILES = {
    'transcriber/transcriber_best.pt': 'audio_candidates',
    'decoder/decoder_best.pt': '4k_layout',
    'decoder_freex/decoder_best.pt': 'freex_static_layout',
}
MIB = 1024**2
GIB = 1024**3
MAX_FILE = 512*MIB
MAX_TOTAL = GIB
MAX_MANIFEST = 64*1024
RESERVE = 8*GIB


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(MIB):
            digest.update(block)
    return digest.hexdigest()


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False)+'\n').encode('utf-8')


def _check_space(directory, extra_bytes=0):
    if shutil.disk_usage(directory).free < RESERVE + extra_bytes:
        raise RuntimeError('Model bundle needs 8 GiB disk reserve after this operation')


def _exists(path):
    # A dangling symlink is also an existing destination, never an invitation
    # to replace it or write through to another directory.
    return path.exists() or path.is_symlink()


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA:
        raise ValueError('unsupported model bundle schema')
    if not isinstance(manifest.get('bundle_id'), str) or not re.fullmatch(
            r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', manifest['bundle_id']):
        raise ValueError('invalid model bundle ID')
    if (manifest.get('timing_schema') != TIMING_SCHEMA
            or manifest.get('validation') != 'file_integrity_only'):
        raise ValueError('unsupported model bundle contract')
    files = manifest.get('files')
    if not isinstance(files, dict) or set(files) != set(FILES):
        raise ValueError('model bundle must contain exactly the three inference weights')
    for name, role in FILES.items():
        entry = files[name]
        if (not isinstance(entry, dict) or entry.get('role') != role
                or type(entry.get('bytes')) is not int or not 0 < entry['bytes'] <= MAX_FILE
                or not isinstance(entry.get('sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', entry['sha256'])):
            raise ValueError(f'invalid model bundle entry: {name}')
    if sum(entry['bytes'] for entry in files.values()) > MAX_TOTAL:
        raise ValueError('model bundle exceeds 1 GiB unpacked budget')
    return manifest


def _json_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate key in model bundle manifest')
        obj[key] = value
    return obj


def _read_manifest(archive):
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(names) != 4 or set(names) != set(FILES) | {MANIFEST}:
        raise ValueError('unexpected, duplicated or missing model bundle member')
    for entry in entries:
        mode = entry.external_attr >> 16
        if (entry.compress_type != zipfile.ZIP_STORED or entry.flag_bits & 1
                or entry.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                or entry.file_size != entry.compress_size
                or not 0 < entry.file_size <= (MAX_MANIFEST if entry.filename == MANIFEST else MAX_FILE)):
            raise ValueError('unsupported model bundle member type or size')
    if sum(entry.file_size for entry in entries) > MAX_TOTAL+MAX_MANIFEST:
        raise ValueError('model bundle exceeds unpacked budget')
    manifest = _validate_manifest(json.loads(archive.read(MANIFEST), object_pairs_hook=_json_object))
    for name, expected in manifest['files'].items():
        if archive.getinfo(name).file_size != expected['bytes']:
            raise ValueError(f'model size differs from manifest: {name}')
    return manifest


def _stream_member(archive, name, expected, destination=None, directory=None):
    digest = hashlib.sha256()
    size = 0
    with archive.open(name) as source:
        while block := source.read(MIB):
            size += len(block)
            if size > expected['bytes']:
                raise ValueError(f'model exceeds declared size: {name}')
            digest.update(block)
            if destination is not None:
                _check_space(directory, len(block))
                destination.write(block)
    if size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
        raise ValueError(f'model checksum mismatch: {name}')


def _check_bundle_file(path, expected_sha256):
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_TOTAL+MIB:
        raise ValueError('missing model bundle or archive exceeds 1 GiB budget')
    digest = sha256(path)
    if expected_sha256 is not None:
        if not re.fullmatch('[0-9a-fA-F]{64}', expected_sha256):
            raise ValueError('expected SHA-256 must contain 64 hexadecimal characters')
        if digest != expected_sha256.lower():
            raise ValueError('model bundle SHA-256 does not match expected transfer checksum')
    return digest


def verify_bundle(bundle, *, expected_sha256=None):
    """Read every byte and validate ZIP CRC, exact paths, sizes and hashes."""
    bundle = Path(bundle)
    digest = _check_bundle_file(bundle, expected_sha256)
    with zipfile.ZipFile(bundle) as archive:
        manifest = _read_manifest(archive)
        for name, expected in manifest['files'].items():
            _stream_member(archive, name, expected)
    return dict(sha256=digest, bytes=bundle.stat().st_size, manifest=manifest)


def verify_installed(model_root):
    """Verify a previously installed manifest and each weight, without Torch."""
    model_root = Path(model_root)
    path = model_root/MANIFEST
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_MANIFEST:
        raise ValueError('missing or oversized installed model manifest')
    manifest = _validate_manifest(json.loads(path.read_bytes(), object_pairs_hook=_json_object))
    for name, expected in manifest['files'].items():
        path = model_root/name
        if (not path.is_file() or path.stat().st_size != expected['bytes']
                or sha256(path) != expected['sha256']):
            raise ValueError(f'installed model checksum mismatch: {name}')
    return manifest


def _member(name):
    info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def export_bundle(model_root, output, *, bundle_id):
    """Export only inference weights. Deterministic bytes, no absolute paths."""
    model_root, output = Path(model_root), Path(output)
    if _exists(output):
        raise FileExistsError(f'model bundle destination already exists: {output}')
    files = {}
    for name, role in FILES.items():
        path = model_root/name
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE:
            raise ValueError(f'missing or oversized model: {name}')
        files[name] = dict(role=role, bytes=path.stat().st_size, sha256=sha256(path))
    manifest = _validate_manifest(dict(schema=SCHEMA, bundle_id=bundle_id,
        timing_schema=TIMING_SCHEMA, validation='file_integrity_only', files=files))
    output.parent.mkdir(parents=True, exist_ok=True)
    total = sum(entry['bytes'] for entry in files.values()) + MAX_MANIFEST
    _check_space(output.parent, total)
    fd, tmp_name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.partial', dir=output.parent)
    partial = Path(tmp_name)
    os.close(fd)
    try:
        with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(_member(MANIFEST), _encoded(manifest))
            for name, expected in files.items():
                count = 0
                with (model_root/name).open('rb') as source, archive.open(_member(name), 'w') as dest:
                    while block := source.read(MIB):
                        count += len(block)
                        if count > expected['bytes']:
                            raise ValueError(f'model changed during export: {name}')
                        _check_space(output.parent, len(block))
                        dest.write(block)
        result = verify_bundle(partial)
        # Hard-link publication is atomic and refuses an existing destination.
        os.link(partial, output)
        return result
    finally:
        partial.unlink(missing_ok=True)


def install_bundle(bundle, model_root, *, expected_sha256=None):
    """Stage and validate the complete bundle before publishing a new root.

    Existing model roots are never merged or replaced. Install an update into a
    different directory and select it explicitly with the runtime's model-root.
    """
    bundle, model_root = Path(bundle), Path(model_root)
    if _exists(model_root):
        raise FileExistsError(f'model root already exists; choose a new directory: {model_root}')
    digest = _check_bundle_file(bundle, expected_sha256)
    model_root.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        manifest = _read_manifest(archive)
        total = sum(entry['bytes'] for entry in manifest['files'].values()) + MAX_MANIFEST
        _check_space(model_root.parent, total)
        lock = model_root.parent/f'.{model_root.name}.install.lock'
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        staging = None
        try:
            if _exists(model_root):
                raise FileExistsError(f'model root already exists: {model_root}')
            staging = Path(tempfile.mkdtemp(prefix=f'.{model_root.name}.install-', dir=model_root.parent))
            for name, expected in manifest['files'].items():
                path = staging/name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as destination:
                    _stream_member(archive, name, expected, destination, model_root.parent)
            (staging/MANIFEST).write_bytes(_encoded(manifest))
            _check_space(model_root.parent)
            if _exists(model_root):
                raise FileExistsError(f'model root appeared during installation: {model_root}')
            staging.rename(model_root)
            staging = None
        finally:
            if staging is not None:
                shutil.rmtree(staging)
            lock.unlink(missing_ok=True)
    return dict(model_root=str(model_root.resolve()), sha256=digest, manifest=manifest)
