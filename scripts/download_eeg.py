"""Download one CHB-MIT EEG recording for the rough-signal benchmark.

CHB-MIT (Children's Hospital Boston / MIT) is hosted on PhysioNet and is
licensed for research use under the PhysioNet Credentialed Health Data
License. We download a single ~42 MB EDF file (chb01_01.edf), a one-hour
multi-channel recording from a paediatric epilepsy patient.

The download is idempotent — re-running this script after the file is
present is a no-op.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf?download"
DST = Path("data/raw/chb01_01.edf")


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)
    if DST.exists() and DST.stat().st_size > 1_000_000:
        print(f"already downloaded: {DST} ({DST.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"downloading {URL} -> {DST}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (FractalSig)"})
    with urllib.request.urlopen(req, timeout=600) as r, DST.open("wb") as f:
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
    print(f"done: {DST.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
