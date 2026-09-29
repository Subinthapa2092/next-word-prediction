"""Sample and mix the blogs / news / twitter sources into one training corpus.

Run standalone once the raw en_US.*.txt files are in data/raw/ (download_data.py
does this automatically). Kept separate so you can re-mix at a different size
or ratio without re-downloading the 560 MB archive.

    python scripts/build_corpus.py --lines-per-source 300000
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SOURCES = ("en_US.blogs.txt", "en_US.news.txt", "en_US.twitter.txt")


def sample_lines(path: Path, n: int, seed: int) -> list[str]:
    """Reservoir-sample n lines so we never load a 200+ MB file fully into memory."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = line
    return reservoir


def build_corpus(sources: dict[str, Path], out_path: Path, lines_per_source: int, seed: int = config.RANDOM_STATE) -> Path:
    all_lines = []
    for name, path in sources.items():
        if not path.exists():
            print(f"  skipping {name}: not found at {path}")
            continue
        lines = sample_lines(path, lines_per_source, seed)
        print(f"  {name}: sampled {len(lines):,} lines")
        all_lines.extend(lines)

    if not all_lines:
        raise SystemExit(f"No source files found in {config.RAW_DIR}. Run download_data.py first.")

    random.Random(seed).shuffle(all_lines)  # interleave registers so no epoch is one-source-heavy
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"Wrote {len(all_lines):,} lines ({out_path.stat().st_size / 1e6:.1f} MB) to {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lines-per-source", type=int, default=config.CORPUS_LINES_PER_SOURCE)
    p.add_argument("--out", default=str(config.DEFAULT_CORPUS))
    p.add_argument("--sources-dir", default=str(config.RAW_DIR), help="folder containing the en_US.*.txt files")
    a = p.parse_args()
    sources = {name: Path(a.sources_dir) / name for name in SOURCES}
    build_corpus(sources, Path(a.out), a.lines_per_source)
