"""Create isolated numerical runs while preserving the submitted assets and data.

Public entry points are Anisotropy_reproducibility.ipynb and code/run_all.py.
The lower-level simulation scripts write relative to their own code/ directory;
therefore these entry points always execute a copied code/data/figures tree.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def tree_hashes(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}


def protected_hashes(source):
    source = Path(source).resolve()
    return {name: tree_hashes(source/name) for name in ('figures', 'data')}


def create_workspace(source, output_parent=None, label='run'):
    source = Path(source).resolve()
    if not (source/'code/polygon.py').is_file():
        raise FileNotFoundError('Source must be the complete extracted package.')
    parent = Path(output_parent).expanduser().resolve() if output_parent else source/'reproduced'
    # Never place generated output within any submitted data or figure directory.
    for reserved in ('figures', 'data', 'code', 'Definitions'):
        target = (source/reserved).resolve()
        if parent == target or target in parent.parents:
            raise ValueError(f'Output directory cannot be inside {reserved}/.')
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    work = Path(tempfile.mkdtemp(prefix=f'{label}_{stamp}_', dir=parent))
    for name in ('code', 'data', 'figures'):
        shutil.copytree(source/name, work/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (work/'review').mkdir()
    record = {'source': str(source), 'workspace': str(work),
              'created_utc': stamp, 'protected_source_hashes': protected_hashes(source)}
    (work/'review/workspace_provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    return work


def assert_source_unchanged(source, before, report_path=None):
    after = protected_hashes(source)
    changed = {name: sorted(set(before[name]) | set(after[name])) for name in before}
    changed = {name: [p for p in paths if before[name].get(p) != after[name].get(p)]
               for name, paths in changed.items()}
    report = {'source_figures_unchanged': not changed['figures'],
              'source_data_unchanged': not changed['data'],
              'figure_assets_checked': len(before['figures']),
              'data_files_checked': len(before['data']), 'changed_paths': changed}
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2)+'\n')
    if any(changed.values()):
        raise RuntimeError('Protected source assets changed: '+json.dumps(changed))
    return report
