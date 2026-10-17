"""Transport invariants: exact bytes, bounded writes and failure atomicity."""
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from src.product import model_bundle as mb


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    # CI disk size is unrelated to the behavior under test. Resource refusal
    # has separate tests with mocked disk readings; never fill a real disk.
    monkeypatch.setattr(mb, '_check_space', lambda *args: None)
    source = tmp_path/'source'
    data = {}
    for number, name in enumerate(mb.FILES):
        path = source/name
        path.parent.mkdir(parents=True)
        data[name] = bytes([number, 255, 3, 0])*17000
        path.write_bytes(data[name])
    output = tmp_path/'models.zip'
    result = mb.export_bundle(source, output, bundle_id='test-v1')
    return source, output, data, result


def rewrite(bundle, output, transform):
    with zipfile.ZipFile(bundle) as original, zipfile.ZipFile(output, 'w') as target:
        for info in original.infolist():
            info, content = transform(info, original.read(info.filename))
            target.writestr(info, content)


def test_roundtrip_independent_of_original_paths_and_no_overwrite(bundle, tmp_path):
    source, archive, contents, report = bundle
    destination = tmp_path/'other-PC/models'
    result = mb.install_bundle(archive, destination, expected_sha256=report['sha256'])
    assert result['manifest']['bundle_id'] == 'test-v1'
    for name, expected in contents.items():
        assert (destination/name).read_bytes() == expected
        (source/name).unlink()
    assert set(p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()) == set(contents) | {mb.MANIFEST}
    assert mb.verify_installed(destination) == report['manifest']
    assert str(source) not in (destination/mb.MANIFEST).read_text()
    with pytest.raises(FileExistsError):
        mb.install_bundle(archive, destination)
    assert mb.verify_installed(destination) == report['manifest']


def test_deterministic_export_and_existing_output_preserved(bundle, tmp_path):
    source, archive, _, report = bundle
    second = tmp_path/'second.zip'
    assert mb.export_bundle(source, second, bundle_id='test-v1')['sha256'] == report['sha256']
    assert second.read_bytes() == archive.read_bytes()
    with pytest.raises(FileExistsError):
        mb.export_bundle(source, archive, bundle_id='different')
    assert mb.sha256(archive) == report['sha256']


def test_wrong_transfer_hash_fails_before_destination_creation(bundle, tmp_path):
    _, archive, _, _ = bundle
    destination = tmp_path/'missing-parent/models'
    with pytest.raises(ValueError, match='transfer checksum'):
        mb.install_bundle(archive, destination, expected_sha256='0'*64)
    assert not destination.parent.exists()


def test_changed_member_crc_valid_but_hash_invalid_leaves_no_partial_install(bundle, tmp_path):
    _, archive, _, _ = bundle
    damaged = tmp_path/'damaged.zip'
    name = next(iter(mb.FILES))
    rewrite(archive, damaged, lambda info, data: (info, b'X'+data[1:] if info.filename == name else data))
    destination = tmp_path/'models'
    with pytest.raises(ValueError, match='checksum mismatch'):
        mb.install_bundle(damaged, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob('.models.install*'))


@pytest.mark.parametrize('alteration', ['traversal', 'symlink', 'compressed', 'size', 'role', 'schema', 'duplicate_json'])
def test_rejects_unsupported_archive_before_publishing(bundle, tmp_path, alteration):
    _, archive, _, _ = bundle
    changed = tmp_path/'changed.zip'
    first = next(iter(mb.FILES))

    def change(info, data):
        if info.filename == first:
            if alteration == 'traversal':
                info.filename = '../escape.pt'
            if alteration == 'symlink':
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            if alteration == 'compressed':
                info.compress_type = zipfile.ZIP_DEFLATED
        if info.filename == mb.MANIFEST:
            manifest = json.loads(data)
            if alteration == 'size':
                manifest['files'][first]['bytes'] = mb.MAX_FILE+1
            if alteration == 'role':
                manifest['files'][first]['role'] = 'wrong-model'
            if alteration == 'schema':
                manifest['timing_schema'] = 'legacy'
            data = json.dumps(manifest).encode()
            if alteration == 'duplicate_json':
                data = b'{"schema":"duplicate",'+data[1:]
        return info, data

    rewrite(archive, changed, change)
    with pytest.raises(ValueError):
        mb.install_bundle(changed, tmp_path/'models')
    assert not (tmp_path/'models').exists()
    assert not (tmp_path/'escape.pt').exists()


def test_missing_or_duplicate_members_rejected(bundle, tmp_path):
    _, archive, _, _ = bundle
    with zipfile.ZipFile(archive, 'a') as target:
        with pytest.warns(UserWarning, match='Duplicate name'):
            target.writestr(mb.MANIFEST, '{}')
    with pytest.raises(ValueError, match='member'):
        mb.verify_bundle(archive)


def test_install_resource_drop_cleans_staging(bundle, tmp_path, monkeypatch):
    _, archive, _, _ = bundle
    calls = 0

    def check(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('simulated disk reserve')

    monkeypatch.setattr(mb, '_check_space', check)
    with pytest.raises(RuntimeError, match='simulated'):
        mb.install_bundle(archive, tmp_path/'models')
    assert not (tmp_path/'models').exists()
    assert not list(tmp_path.glob('.models.install*'))


def test_export_resource_drop_never_publishes(bundle, tmp_path, monkeypatch):
    source, _, _, _ = bundle
    calls = 0

    def check(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('simulated disk reserve')

    monkeypatch.setattr(mb, '_check_space', check)
    with pytest.raises(RuntimeError, match='simulated'):
        mb.export_bundle(source, tmp_path/'interrupted.zip', bundle_id='test-v1')
    assert not (tmp_path/'interrupted.zip').exists()
    assert not list(tmp_path.glob('.interrupted.zip.*'))


def test_resource_estimate_reserves_space_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(mb.shutil, 'disk_usage', lambda _: SimpleNamespace(free=mb.RESERVE+100))
    mb._check_space(tmp_path, 100)
    with pytest.raises(RuntimeError, match='8 GiB'):
        mb._check_space(tmp_path, 101)


def test_corrupted_installed_weight_detected(bundle, tmp_path):
    _, archive, _, _ = bundle
    destination = tmp_path/'models'
    mb.install_bundle(archive, destination)
    (destination/next(iter(mb.FILES))).write_bytes(b'changed')
    with pytest.raises(ValueError, match='installed model checksum'):
        mb.verify_installed(destination)


def test_cli_verifies_without_any_site_packages(bundle):
    _, archive, _, report = bundle
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-S', '-m', 'script.model_bundle', 'verify',
                             '--bundle', str(archive), '--sha256', report['sha256']],
                            cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()
