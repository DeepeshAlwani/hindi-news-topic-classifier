"""
Batch-classify all images in a folder and write results to CSV + JSON.

Usage:
    python batch_classify.py path/to/image_folder
    python batch_classify.py path/to/image_folder --out results
"""

import argparse
import csv
import json
import os
import sys
import time

from classifier import classify_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_images(folder: str) -> list:
    paths = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            paths.append(os.path.join(folder, name))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing images to classify")
    parser.add_argument("--out", default="results", help="Output file prefix (default: results)")
    parser.add_argument("--delay", type=float, default=1.0,
                         help="Seconds to wait between requests (be polite to the free tier)")
    args = parser.parse_args()

    images = find_images(args.folder)
    if not images:
        print(f"No images found in {args.folder}")
        sys.exit(1)

    print(f"Found {len(images)} images. Classifying...\n")

    results = []
    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {os.path.basename(path)} ...", end=" ", flush=True)
        result = classify_image(path)
        results.append(result)

        if result.get("_parse_error"):
            print(f"FAILED ({result.get('error', 'unknown error')[:80]})")
        else:
            print(f"{result.get('category')} ({result.get('confidence')})")

        if i < len(images):
            time.sleep(args.delay)

    # Write JSON (full detail)
    json_path = f"{args.out}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Write CSV (summary view)
    csv_path = f"{args.out}.csv"
    fieldnames = ["_image", "category", "confidence", "extracted_text",
                  "extracted_text_translation", "reasoning", "_parse_error", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    failed = sum(1 for r in results if r.get("_parse_error"))
    print(f"\nDone. {len(results) - failed}/{len(results)} succeeded.")
    print(f"Written to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()