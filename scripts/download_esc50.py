"""Download a small subset of ESC-50 environmental sound clips.

ESC-50 (Piczak 2015) is a public CC-licensed collection of 2000 environmental
audio recordings spanning 50 classes. We download the GitHub source archive
(~600 MB), extract once, and let `fractalsig.datasets.audio_esc50` slice it
into seq_len windows on demand.
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ZIP = Path("data/raw/esc50.zip")
EXTRACT = Path("data/raw/esc50")
SENTINEL = EXTRACT / "ESC-50-master" / "audio"


def main() -> None:
    EXTRACT.mkdir(parents=True, exist_ok=True)
    if SENTINEL.exists() and any(SENTINEL.iterdir()):
        print(f"already extracted: {SENTINEL}")
        return
    print(f"downloading {URL} -> {ZIP}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (FractalSig)"})
    with urllib.request.urlopen(req, timeout=600) as r, ZIP.open("wb") as f:
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
    print(f"extracting {ZIP}")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(EXTRACT)
    print(f"done. audio at {SENTINEL}")


if __name__ == "__main__":
    main()
