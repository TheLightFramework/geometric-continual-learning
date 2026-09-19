"""Verify the declared source/evidence inventory, not model competence."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    for rel,entry in manifest['files'].items():
        p=(ROOT/rel).resolve()
        if not p.is_relative_to(ROOT):raise ValueError('UNSAFE_MANIFEST_PATH')
        with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
        if h!=entry['sha256'] or p.stat().st_size!=entry['bytes']:raise ValueError(f'MISMATCH: {rel}')
    print('Verified',len(manifest['files']),'declared files. This is integrity, not authentication or competence.')

if __name__=='__main__':main()
