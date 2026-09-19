"""MIT. Research-only, single-writer, fail-closed artifact transaction store.

This does not train, evaluate semantics, or certify an evaluator's honesty. It binds
an external evaluation receipt to exact artifacts and a declared parent. Callers
must evaluate a disposable clone before constructing a candidate. Original weights
are never overwritten. A stale writer lock requires human inspection, not auto-removal.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def artifact(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path)}


def verify_artifact(ref):
    if set(ref) != {'path', 'sha256'} or not Path(ref['path']).is_absolute():
        raise ValueError('INVALID_ARTIFACT_REFERENCE')
    if digest(ref['path']) != ref['sha256']:
        raise ValueError('ARTIFACT_CHANGED')


def exclusive_json(path, value):
    with Path(path).open('xb') as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ['versions', 'decisions', 'events']:
            (self.root / name).mkdir(exist_ok=True)

    @contextmanager
    def writer(self):
        lock = self.root / 'WRITER.lock'
        with lock.open('x', encoding='utf-8') as f:
            f.write(str(os.getpid()))
        try:
            yield
        finally:
            lock.unlink()

    def version(self, version_id):
        if len(version_id) != 64 or any(c not in '0123456789abcdef' for c in version_id):
            raise ValueError('INVALID_VERSION_ID')
        p = self.root / 'versions' / (version_id + '.json')
        data = read(p)
        if hashlib.sha256(canonical(data)).hexdigest() != version_id:
            raise ValueError('VERSION_MANIFEST_CHANGED')
        for ref in data['artifacts']:
            verify_artifact(ref)
        for name in ['evaluation', 'contract']:
            if name in data:
                verify_artifact(data[name])
        return data

    def current(self):
        version_id = read(self.root / 'CURRENT.json')['version']
        self.version(version_id)
        return version_id

    def _save_version(self, value):
        version_id = hashlib.sha256(canonical(value)).hexdigest()
        p = self.root / 'versions' / (version_id + '.json')
        if p.exists():
            if p.read_bytes() != canonical(value):
                raise ValueError('VERSION_COLLISION')
        else:
            exclusive_json(p, value)
        return version_id

    def _switch(self, version_id, reason):
        self.version(version_id)
        event_id = uuid.uuid4().hex
        current = self.root / 'CURRENT.json'
        old = read(current)['version'] if current.exists() else None
        event = {'from': old, 'to': version_id, 'reason': reason, 'standing': 'research sandbox; not production promotion'}
        exclusive_json(self.root / 'events' / (event_id + '.json'), event)
        temporary = self.root / (event_id + '.pointer')
        exclusive_json(temporary, {'version': version_id, 'event': event_id})
        # Atomic single-filesystem pointer replacement. Crash before replacement
        # leaves the previous version active; an unreferenced intent is not a commit.
        os.replace(temporary, current)

    def initialize(self, artifacts, label):
        with self.writer():
            if (self.root / 'CURRENT.json').exists():
                raise ValueError('ALREADY_INITIALIZED')
            if not artifacts:
                raise ValueError('EMPTY_ARTIFACT_SET')
            for ref in artifacts:
                verify_artifact(ref)
            version_id = self._save_version({'label': label, 'parent': None, 'artifacts': artifacts})
            self._switch(version_id, 'initialize sandbox reference')
            return version_id

    def review(self, candidate, apply=False):
        """Receipt schema: parent, artifacts, contract_sha256, tests{name: bool}.

        candidate contains parent/artifacts/contract/evaluation. The contract file
        declares required_tests and evidence_standing. No scoring threshold is
        calculated here: exact binding and typed, complete external checks only.
        """
        with self.writer():
            parent = self.current()
            if candidate['parent'] != parent:
                raise ValueError('STALE_PARENT')
            if not candidate['artifacts']:
                raise ValueError('EMPTY_ARTIFACT_SET')
            for ref in [*candidate['artifacts'], candidate['contract'], candidate['evaluation']]:
                verify_artifact(ref)
            contract = read(candidate['contract']['path'])
            receipt = read(candidate['evaluation']['path'])
            required = contract['required_tests']
            if not required or len(set(required)) != len(required):
                raise ValueError('INVALID_REQUIRED_TESTS')
            if receipt['parent'] != parent or receipt['artifacts'] != candidate['artifacts'] or receipt['contract_sha256'] != candidate['contract']['sha256']:
                raise ValueError('EVALUATION_NOT_BOUND_TO_CANDIDATE')
            tests = receipt['tests']
            if any(name not in tests or type(tests[name]) is not bool for name in required):
                raise ValueError('MISSING_OR_UNTYPED_TEST')
            failures = [name for name in required if not tests[name]]
            decision = {'candidate': candidate, 'failed': failures, 'eligible': not failures, 'apply_requested': bool(apply), 'evidence_standing': contract['evidence_standing']}
            decision_id = uuid.uuid4().hex
            exclusive_json(self.root / 'decisions' / (decision_id + '.json'), decision)
            if apply and not failures:
                version_id = self._save_version({'label': decision_id, 'parent': parent, 'artifacts': candidate['artifacts'], 'evaluation': candidate['evaluation'], 'contract': candidate['contract']})
                self._switch(version_id, 'explicit apply after complete declared checks')
            if failures or not apply:
                assert self.current() == parent
            return {**decision, 'applied': bool(apply and not failures)}

    def rollback(self, ancestor):
        with self.writer():
            current = self.current()
            cursor = current
            while cursor is not None and cursor != ancestor:
                cursor = self.version(cursor)['parent']
            if cursor is None:
                raise ValueError('ROLLBACK_TARGET_NOT_ANCESTOR')
            self._switch(ancestor, 'explicit ancestor rollback')
            return self.current()
