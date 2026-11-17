"""Download and unpack the UCI hydraulic dataset into data/raw/.

Usage:  python -m src.download_data

If the download fails (firewall, offline), download the zip manually from
https://archive.ics.uci.edu/dataset/447 and unzip it into data/raw/ so that
data/raw/PS1.txt, data/raw/profile.txt, ... exist.
"""
import sys
import urllib.request
import zipfile

from src.config import DATASET_URL, RAW_DIR, SENSORS


def already_downloaded() -> bool:
    needed = [RAW_DIR / f"{s}.txt" for s in SENSORS] + [RAW_DIR / "profile.txt"]
    return all(p.exists() for p in needed)


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        done = min(block_num * block_size, total_size)
        sys.stdout.write(f"\r  {done / 1e6:6.1f} / {total_size / 1e6:.1f} MB")
        sys.stdout.flush()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if already_downloaded():
        print(f"Dataset already present in {RAW_DIR}")
        return
    zip_path = RAW_DIR / "hydraulic.zip"
    print(f"Downloading {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, zip_path, _progress)
    print("\nExtracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()
    if not already_downloaded():
        raise SystemExit(
            "Extraction finished but expected files are missing. If the zip contained a "
            "sub-folder, move its contents up into data/raw/."
        )
    print("Done.")


if __name__ == "__main__":
    main()