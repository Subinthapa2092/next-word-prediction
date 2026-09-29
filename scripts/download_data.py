"""Download the SwiftKey / HC Corpora text corpus into data/raw/.

This is the real dataset built for exactly this task: predictive typing on a
mobile keyboard. It was released by SwiftKey for Coursera's Data Science
Specialization capstone and is sourced from HC Corpora (public web text).
It ships three English registers in one archive:

    en_US.blogs.txt     long-form, informal prose      ~900k lines
    en_US.news.txt      edited, formal writing          ~1.0M lines
    en_US.twitter.txt   short, noisy, social text        ~2.4M lines

Free to use for research, learning and portfolio projects like this one; if
this ever becomes a shipped commercial product, use a dataset with clearer
commercial terms instead (e.g. WikiText-103, or your own collected text).

    python scripts/download_data.py                  # fetch + extract + build data/raw/corpus.txt
    python scripts/download_data.py --skip-download   # already have the zip, just rebuild the corpus
    python scripts/download_data.py --lines-per-source 200000
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.build_corpus import build_corpus  # noqa: E402

SWIFTKEY_URL = "https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip"
SOURCE_FILES = ("en_US.blogs.txt", "en_US.news.txt", "en_US.twitter.txt")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}  (~560 MB, this can take a few minutes)")
    req = urllib.request.Request(url, headers={"User-Agent": "next-word-prediction/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"Saved {dest.stat().st_size / 1e6:.0f} MB")


def extract(zip_path: Path, out_dir: Path) -> dict[str, Path]:
    print(f"Extracting en_US/* from {zip_path}")
    paths = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in SOURCE_FILES:
            member = f"final/en_US/{name}"
            target = out_dir / name
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            paths[name] = target
            print(f"  {name}: {target.stat().st_size / 1e6:.0f} MB")
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=SWIFTKEY_URL)
    p.add_argument("--skip-download", action="store_true", help="reuse an already-downloaded zip")
    p.add_argument("--zip-path", default=str(config.RAW_DIR / "Coursera-SwiftKey.zip"))
    p.add_argument("--lines-per-source", type=int, default=150_000,
                    help="lines sampled from each of blogs/news/twitter (default keeps training under ~20 min on CPU)")
    p.add_argument("--out", default=str(config.DEFAULT_CORPUS))
    a = p.parse_args()

    zip_path = Path(a.zip_path)
    if not a.skip_download:
        try:
            download(a.url, zip_path)
        except Exception as exc:
            raise SystemExit(
                f"Could not download the corpus automatically ({exc}).\n"
                f"Download it yourself from {a.url} and save it to {zip_path}, "
                "then re-run with --skip-download."
            )
    elif not zip_path.exists():
        raise SystemExit(f"--skip-download was set but {zip_path} does not exist.")

    sources = extract(zip_path, config.RAW_DIR)
    build_corpus(sources, Path(a.out), a.lines_per_source, seed=config.RANDOM_STATE)


if __name__ == "__main__":
    main()
