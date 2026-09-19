#!/usr/bin/env python3
"""Reproduce results in a new isolated directory; never replace submitted figures/data.

Default output: reproduced/<timestamped-run>/. Intermediate simulation scripts are
internal workers and are executed only from that copy. The optional expensive
historical scan regeneration is disabled unless explicitly requested.
"""
import argparse
from pathlib import Path
import subprocess
import sys
from reproduction_workspace import create_workspace, protected_hashes, assert_source_unchanged
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-parent', type=Path, help='Parent for a new isolated run directory (default: reproduced/).')
    parser.add_argument('--regenerate-tissue',action='store_true',help='Regenerate the original 200 tissue realizations inside the run copy.')
    parser.add_argument('--regenerate-original',action='store_true',help='Regenerate all legacy Monte Carlo cell scans inside the run copy; can be slow.')
    parser.add_argument('--verify-only',action='store_true',help='Check supplied data in the isolated copy, without rerunning regional experiments.')
    parser.add_argument('--build-pdfs',action='store_true',help='Also build main.pdf and SI.pdf from the untouched source figures; requires pdflatex and bibtex.')
    args=parser.parse_args()
    if args.verify_only and (args.regenerate_original or args.regenerate_tissue):
        parser.error('--verify-only cannot be combined with regeneration flags.')
    before=protected_hashes(ROOT)
    work=create_workspace(ROOT,args.output_parent)
    print(f'Isolated numerical output: {work}',flush=True)
    jobs=[]
    if not args.verify_only:
        if args.regenerate_original:
            jobs.append(['run_original_figures.py','--output-directory','reproduced_figures'])
        if args.regenerate_tissue:
            jobs.append(['run_tissue.py','--regenerate'])
        jobs.extend([['run_boundary_identity.py'],['run_regional_mechanics.py'],
                     ['run_boundary_scaling.py'],['run_driven_strain.py'],['run_polarized_tension.py'],
                     ['fit_window_controls.py'],['run_regional_mechanism_figure.py']])
    jobs.extend([['verify_package.py'],['verify_regional.py'],['verify_window_controls.py']])
    try:
        for name,*arguments in jobs:
            print(f'Running {name}',flush=True)
            subprocess.run([sys.executable,str(work/'code'/name),*arguments],cwd=work,check=True)
        if args.build_pdfs:
            # Typeset with the untouched submitted assets, never the regenerated plots.
            subprocess.run([sys.executable,str(ROOT/'code/build_documents.py')],cwd=ROOT,check=True)
    finally:
        report=assert_source_unchanged(ROOT,before,work/'review/source_integrity.json')
        print(f"Protected assets unchanged: {report['figure_assets_checked']} figures and {report['data_files_checked']} data files.",flush=True)
    print(f'Requested workflow completed. Results: {work}')


if __name__=='__main__':
    main()
