"""
Download the NASA CMAPSS turbofan engine degradation dataset.
Falls back to a GitHub mirror since the NASA S3 URL returns 403.

Usage:
    python scripts/download_cmapss.py

Files will be saved to the data/raw/ directory.
"""
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# GitHub mirror of the CMAPSS dataset (edwardzjl/CMAPSSData)
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master/"
)

EXPECTED_FILES = [
    "train_FD001.txt",
    "train_FD002.txt",
    "train_FD003.txt",
    "train_FD004.txt",
    "test_FD001.txt",
    "test_FD002.txt",
    "test_FD003.txt",
    "test_FD004.txt",
    "RUL_FD001.txt",
    "RUL_FD002.txt",
    "RUL_FD003.txt",
    "RUL_FD004.txt",
]


def already_downloaded() -> bool:
    """Return True if all expected CMAPSS files are present in data/raw/."""
    return all(
        os.path.exists(os.path.join(DATA_DIR, fname)) for fname in EXPECTED_FILES
    )


def download_files() -> None:
    """Download each CMAPSS text file individually from GitHub."""
    os.makedirs(DATA_DIR, exist_ok=True)

    for fname in EXPECTED_FILES:
        dest = os.path.join(DATA_DIR, fname)
        if os.path.exists(dest):
            print(f"  Already present, skipping: {fname}")
            continue

        url = GITHUB_RAW_BASE + fname
        print(f"  Downloading {fname} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"    Saved to {dest}")
        except Exception as e:
            print(f"    ERROR downloading {fname}: {e}")
            raise

    print("\nDone. All CMAPSS data files are in the data/raw/ directory.")


def main() -> None:
    if already_downloaded():
        print("CMAPSS files already present in data/raw/ — nothing to do.")
        return
    download_files()


if __name__ == "__main__":
    main()
