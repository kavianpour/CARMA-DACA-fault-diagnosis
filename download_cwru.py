"""
Auto-downloader for the exact CWRU files CARMA-DACA needs.

The paper uses BOTH the 12 kHz drive-end set (source domains A1/B1/C1/D1) and the
48 kHz drive-end set (target domains C2/D2), plus the Normal baseline (NC class).
This script downloads all of them into the target directory (default ./CWRU),
naming each file <number>.mat so that data.py can find it. Every file is verified
by loading it with scipy and checking for a *_DE_time key; files that already
exist and validate are skipped.

Usage:
    python download_cwru.py                 # -> ./CWRU
    python download_cwru.py --root ./CWRU

If automatic download fails (e.g. the university site is unreachable from your
network), download the dataset manually instead:
    * https://engineering.case.edu/bearingdatacenter/download-data-file
    * Kaggle mirror: https://www.kaggle.com/datasets/sufian79/cwru-mat-full-dataset
and drop the *.mat files (named like 97.mat, 105.mat, 109.mat, ...) into ./CWRU/.
"""

import os
import sys
import argparse
import urllib.request

import scipy.io as sio

# Mirrors data.CWRU_12K / data.CWRU_48K (kept inline so the downloader has no
# torch / torch_geometric dependency and runs before the ML stack is installed).
CWRU_12K = {
    "NC":    [97, 98, 99, 100],
    "IF007": [105, 106, 107, 108], "IF014": [169, 170, 171, 172], "IF021": [209, 210, 211, 212],
    "OF007": [130, 131, 132, 133], "OF014": [197, 198, 199, 200], "OF021": [234, 235, 236, 237],
    "BF007": [118, 119, 120, 121], "BF014": [185, 186, 187, 188], "BF021": [222, 223, 224, 225],
}
CWRU_48K = {
    "NC":    [97, 98, 99, 100],
    "IF007": [109, 110, 111, 112], "IF014": [174, 175, 176, 177], "IF021": [213, 214, 215, 217],
    "OF007": [135, 136, 137, 138], "OF014": [201, 202, 203, 204], "OF021": [238, 239, 240, 241],
    "BF007": [122, 123, 124, 125], "BF014": [189, 190, 191, 192], "BF021": [226, 227, 228, 229],
}

URL_TEMPLATES = [
    "https://engineering.case.edu/sites/default/files/{n}.mat",
    "http://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/{n}.mat",
    "https://engineering.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/{n}.mat",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CARMA-DACA-downloader/1.0)"}


def file_numbers():
    nums = set()
    for mp in (CWRU_12K, CWRU_48K):
        for loads in mp.values():
            nums.update(loads)
    return sorted(nums)


def is_valid_mat(path):
    try:
        return any("DE_time" in k for k in sio.loadmat(path).keys())
    except Exception:
        return False


def download_one(number, root):
    dest = os.path.join(root, f"{number}.mat")
    if os.path.exists(dest) and is_valid_mat(dest):
        return True, "cached"
    for tmpl in URL_TEMPLATES:
        url = tmpl.format(n=number)
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            if is_valid_mat(dest):
                return True, url
            os.remove(dest)                       # downloaded an HTML error page, etc.
        except Exception:
            if os.path.exists(dest):
                os.remove(dest)
            continue
    return False, "all URLs failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./CWRU")
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)

    nums = file_numbers()
    print(f"Need {len(nums)} CWRU files (12 kHz + 48 kHz + baseline) -> {args.root}")
    ok, fail = [], []
    for n in nums:
        success, info = download_one(n, args.root)
        print(f"  [{'OK  ' if success else 'FAIL'}] {n}.mat   ({info})")
        (ok if success else fail).append(n)

    print(f"\nDone: {len(ok)} ok, {len(fail)} failed.")
    if fail:
        print("Failed files:", fail)
        print("Download these manually (see the note at the top of this file) and "
              "drop them into", args.root)
        sys.exit(1)
    print("All required CWRU files are present. You can now run train.py.")


if __name__ == "__main__":
    main()
