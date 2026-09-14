"""Invoke the exact official ARC builder, and record input hashes/protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from .upstream import checkout, TRM_SHA


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', default='.external/trm')
    p.add_argument('--input-prefix', help='Default: pinned TRM kaggle/combined/arc-agi')
    p.add_argument('--output', required=True)
    p.add_argument('--arc-version', choices=['1', '2'], default='1')
    p.add_argument('--num-aug', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    upstream = checkout(a.upstream)
    prefix = Path(a.input_prefix).resolve() if a.input_prefix else upstream / 'kaggle/combined/arc-agi'
    dest = Path(a.output).resolve()
    if dest.exists():
        raise FileExistsError('Use a new output directory; do not overwrite a data manifest')
    subsets = ['training', 'evaluation', 'concept'] if a.arc_version == '1' else ['training2', 'evaluation2', 'concept']
    sources = {}
    for subset in subsets:
        for suffix in ['challenges', 'solutions']:
            path = Path(f'{prefix}_{subset}_{suffix}.json')
            if path.exists():
                sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    command = [sys.executable, '-m', 'dataset.build_arc_dataset', '--input-file-prefix', str(prefix),
               '--output-dir', str(dest), '--subsets', *subsets, '--test-set-name', subsets[1],
               '--num-aug', str(a.num_aug), '--seed', str(a.seed)]
    subprocess.run(command, cwd=upstream, check=True)
    manifest = dict(trm_commit=TRM_SHA, arc_version=a.arc_version, subsets=subsets,
                    num_aug=a.num_aug, seed=a.seed, source_sha256=sources,
                    protocol='official task-demonstration adaptation',
                    heldout_query_outputs_for_scoring_only=True,
                    warning='Do not mix ARC-2 training into an ARC-1 evaluation run.')
    (dest / 'fpsa_data_manifest.json').write_text(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
