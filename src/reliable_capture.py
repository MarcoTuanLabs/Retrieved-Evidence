"""Durable local research captures and bounded, byte-verified checkpoint copies.

This module never imports a model, scorer, reference answer or routing policy.
Call it only between timed requests. A failed mirror is not a successful backup.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def verify_archive(path):
    """Verify manifest coverage and member bytes without extracting any paths."""
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        assert len(names) == len(set(names)), 'Duplicate archive member'
        assert all(m.isfile() and not Path(m.name).is_absolute()
                   and '..' not in Path(m.name).parts for m in members)
        manifest = json.load(archive.extractfile('checkpoint_manifest.json'))
        assert set(names) == set(manifest['files']) | {'checkpoint_manifest.json'}
        for name, expected in manifest['files'].items():
            body = archive.extractfile(name).read()
            assert hashlib.sha256(body).hexdigest() == expected, name
    return manifest


def mirror_file(source, destination, expected):
    """Publish a copy only after read-back verification; never replace prior data."""
    source, destination = Path(source), Path(destination)
    assert digest(source) == expected, 'Local archive digest mismatch'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert digest(destination) == expected, 'Existing checkpoint differs'
        return
    tmp = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.part')
    try:
        with source.open('rb') as reader, tmp.open('xb') as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        assert digest(tmp) == expected, 'Mirrored bytes differ before publication'
        # Names are unique for each checkpoint; a previous checkpoint is untouched.
        assert not destination.exists(), 'Checkpoint appeared during copy'
        os.replace(tmp, destination)
        fsync_directory(destination.parent)
        assert digest(destination) == expected, 'Mirrored bytes differ after publication'
    finally:
        if tmp.exists():
            tmp.unlink()


def checkpoint(capture, output, tag, mirror_root=None, mirror_timeout=20):
    tick = time.perf_counter()
    capture, output = Path(capture), Path(output)
    assert tag and all(c.isalnum() or c in '_-' for c in tag)
    assert capture.resolve() not in output.resolve().parents
    assert capture.resolve() != output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / (tag + '.tar.gz')
    assert not path.exists(), 'Refuse to replace a checkpoint'
    tmp = output / (tag + '.' + uuid.uuid4().hex + '.tmp')
    hashes = {}
    try:
        with tmp.open('xb') as raw:
            with tarfile.open(fileobj=raw, mode='w:gz') as archive:
                for source in sorted(capture.rglob('*')):
                    if not source.is_file():
                        continue
                    assert not source.is_symlink(), 'No symlink capture members'
                    name = str(Path('capture') / source.relative_to(capture))
                    # Read once: member content and its hash describe the same bytes.
                    body = source.read_bytes()
                    hashes[name] = hashlib.sha256(body).hexdigest()
                    member = tarfile.TarInfo(name)
                    member.size = len(body)
                    archive.addfile(member, io.BytesIO(body))
                body = json.dumps({'files': hashes, 'tag': tag}, sort_keys=True).encode()
                member = tarfile.TarInfo('checkpoint_manifest.json')
                member.size = len(body)
                archive.addfile(member, io.BytesIO(body))
            raw.flush()
            os.fsync(raw.fileno())
        verify_archive(tmp)
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()
    expected = digest(path)
    result = {'tag': tag, 'local_archive': str(path), 'sha256': expected,
              'bytes': path.stat().st_size, 'capture_files': len(hashes),
              'local_archive_verified': True, 'mirror': {'attempted': False}}
    if mirror_root is not None:
        target = Path(mirror_root) / path.name
        start = time.perf_counter()
        result['mirror'] = {'attempted': True, 'verified': False, 'destination': str(target)}
        try:
            call = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'mirror',
                                   '--source', str(path), '--destination', str(target),
                                   '--sha256', expected], capture_output=True, text=True,
                                  timeout=mirror_timeout, check=False)
            result['mirror'].update(verified=call.returncode == 0, returncode=call.returncode,
                                    error=call.stderr[-2000:] if call.returncode else None)
        except subprocess.TimeoutExpired:
            result['mirror']['error'] = 'Mirror timeout; no verified remote copy claimed'
        result['mirror']['seconds'] = time.perf_counter() - start
    result['checkpoint_seconds_outside_online_timer'] = time.perf_counter() - tick
    atomic_json(output / (tag + '.receipt.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['mirror'])
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    mirror_file(args.source, args.destination, args.sha256)


if __name__ == '__main__':
    main()
