"""One-shot script: build all 5 dataset caches.

Run once after a fresh clone. Each dataset is idempotent — if its cache
already exists the constructor is a no-op. EEG and ESC-50 require their
respective downloader scripts to have run first.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from fractalsig.datasets.audio_esc50 import AudioESC50
    from fractalsig.datasets.eeg_chbmit import EEGCHBMIT
    from fractalsig.datasets.sp500_intraday import SP500Intraday
    from fractalsig.datasets.synthetic_fbm import SyntheticFBM
    from fractalsig.datasets.turbulence_burgers import TurbulenceBurgers

    classes = (SyntheticFBM, SP500Intraday, TurbulenceBurgers, EEGCHBMIT, AudioESC50)
    for cls in classes:
        print(f"building {cls.__name__}...", flush=True)
        try:
            cls("train")
            print(f"  ok: {cls.__name__}")
        except FileNotFoundError as e:
            print(f"  skipped {cls.__name__}: {e}")
            print(
                "  hint: run scripts/download_eeg.py and/or scripts/download_esc50.py first."
            )
    print("done.")


if __name__ == "__main__":
    main()
