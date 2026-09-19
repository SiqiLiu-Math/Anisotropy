#!/usr/bin/env python3
"""Build both documents and resolve their reciprocal external references."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(command, directory):
    result = subprocess.run(command, cwd=directory, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout[-6000:])


def main():
    # Isolate intermediate writes; only publish complete, checked build artifacts.
    with tempfile.TemporaryDirectory(prefix="regional_tex_") as temporary:
        work = Path(temporary)
        shutil.copytree(ROOT/"figures", work/"figures")
        if (ROOT/"Definitions").exists():
            shutil.copytree(ROOT/"Definitions", work/"Definitions")
        for filename in ("main.tex", "SI.tex", "references.bib", "main.aux", "SI.aux"):
            if (ROOT/filename).exists():
                shutil.copy2(ROOT/filename, work/filename)
        for name in ("main", "SI"):
            run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name+".tex"], work)
            run(["bibtex", name], work)
        for _ in range(2):
            for name in ("main", "SI"):
                run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name+".tex"], work)
        failures = []
        for name in ("main", "SI"):
            log = (work/(name+".log")).read_text(errors="replace")
            for phrase in ("undefined references", "undefined citations", "Overfull",
                           "multiply defined", "Rerun to get cross-references right",
                           "pdfTeX warning (dest)"):
                if phrase in log:
                    failures.append(f"{name}: {phrase}")
            if not (work/(name+".pdf")).read_bytes().rstrip().endswith(b"%%EOF"):
                failures.append(f"{name}: incomplete PDF")
        if failures:
            raise RuntimeError("Check document logs: " + "; ".join(failures))
        for name in ("main", "SI"):
            for suffix in ("pdf", "aux", "bbl", "log", "out", "blg"):
                source = work/(name+"."+suffix)
                if not source.exists():
                    continue
                with tempfile.NamedTemporaryFile(dir=ROOT, prefix=".build_", delete=False) as handle:
                    handle.write(source.read_bytes())
                    staged = Path(handle.name)
                os.replace(staged, ROOT/source.name)
    print("Built main.pdf and SI.pdf with resolved references and citations.")


if __name__ == "__main__":
    main()
