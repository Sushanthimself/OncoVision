#!/usr/bin/env python3
"""
OncoVision — Training Image Scraper
====================================
Downloads proof-of-concept histology images for three diagnostic categories
using the icrawler library (no API keys required).

Usage:
    python scrape_training_images.py

Categories scraped:
    1. Dentigerous Cyst  (H&E stain histology)
    2. Invasive Ductal Carcinoma  (breast histology, H&E stain)
    3. Normal Healthy Epithelium  (tissue histology, H&E)
"""

import os
import sys
import time
import logging
from pathlib import Path

from icrawler.builtin import BingImageCrawler

# ---------------------------------------------------------------------------
# Configuration — search terms, target folders, naming prefixes
# ---------------------------------------------------------------------------

BASE_DIR = "/Users/amitkandari/OncoVision/backend/training_data"

CATEGORIES = [
    {
        "search_term": "Dentigerous Cyst histology H&E stain",
        "target_folder": os.path.join(BASE_DIR, "dentigerous_cyst"),
        "naming_prefix": "cyst_",
    },
    {
        "search_term": "Invasive Ductal Carcinoma breast histology H&E stain",
        "target_folder": os.path.join(BASE_DIR, "ductal_carcinoma"),
        "naming_prefix": "ductal_carcinoma_",
    },
    {
        "search_term": "Normal healthy epithelium tissue histology H&E",
        "target_folder": os.path.join(BASE_DIR, "normal_tissue"),
        "naming_prefix": "normal_",
    },
]

# Number of images to download per category
MAX_IMAGES = 20

# Allowed image extensions (standard formats only)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("oncovision_scraper")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def ensure_directory(path: str) -> None:
    """Create the target directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)
    logger.info("Directory ready: %s", path)


def rename_images(folder: str, prefix: str) -> int:
    """
    Sequentially rename every valid image in *folder* using the given *prefix*.

    Steps:
        1. List all files in the folder.
        2. Filter to allowed image extensions (.jpg, .jpeg, .png).
        3. Sort alphabetically for deterministic ordering.
        4. Rename each file to <prefix><counter>.<ext>  (e.g. cyst_1.jpg).

    Returns the number of files successfully renamed.
    """
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and Path(f).suffix.lower() in ALLOWED_EXTENSIONS
    )

    if not files:
        logger.warning("No valid image files found in %s — skipping rename.", folder)
        return 0

    renamed_count = 0
    for idx, filename in enumerate(files, start=1):
        ext = Path(filename).suffix.lower()
        # Normalize .jpeg -> .jpg for uniformity
        if ext == ".jpeg":
            ext = ".jpg"
        new_name = f"{prefix}{idx}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        try:
            os.rename(src, dst)
            renamed_count += 1
        except OSError as err:
            logger.error("Failed to rename %s -> %s: %s", src, dst, err)

    logger.info("Renamed %d image(s) in %s", renamed_count, folder)
    return renamed_count


def remove_non_images(folder: str) -> None:
    """Delete any downloaded file that is not a valid image format."""
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath) and Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            logger.info("Removing non-image file: %s", filepath)
            os.remove(filepath)


# ---------------------------------------------------------------------------
# Main scraping pipeline
# ---------------------------------------------------------------------------


def scrape_category(search_term: str, target_folder: str, prefix: str) -> None:
    """
    Download images for a single category and post-process the results.

    Pipeline:
        1. Ensure target directory exists.
        2. Use BingImageCrawler to fetch up to MAX_IMAGES images.
        3. Remove any files that aren't .jpg / .png.
        4. Sequentially rename surviving images with the given prefix.
    """
    logger.info("=" * 60)
    logger.info("CATEGORY: %s", search_term)
    logger.info("TARGET  : %s", target_folder)
    logger.info("=" * 60)

    # Step 1 — Create directory
    ensure_directory(target_folder)

    # Step 2 — Crawl images from Bing
    crawler = BingImageCrawler(
        storage={"root_dir": target_folder},
        log_level=logging.WARNING,          # Suppress noisy icrawler logs
    )

    crawler.crawl(
        keyword=search_term,
        max_num=MAX_IMAGES,
        file_idx_offset=0,
    )

    # Brief pause to let filesystem sync
    time.sleep(1)

    # Step 3 — Remove non-standard files (e.g. .gif, .bmp, .webp)
    remove_non_images(target_folder)

    # Step 4 — Sequential rename
    count = rename_images(target_folder, prefix)
    logger.info("Finished category '%s' — %d usable images.\n", search_term, count)


def main() -> None:
    """Entry point — iterate through all categories."""
    logger.info("OncoVision Training Image Scraper — Starting")
    logger.info("Downloading %d images per category.\n", MAX_IMAGES)

    total_success = 0

    for category in CATEGORIES:
        try:
            scrape_category(
                search_term=category["search_term"],
                target_folder=category["target_folder"],
                prefix=category["naming_prefix"],
            )
            total_success += 1
        except Exception as exc:
            # Gracefully handle any failure (network, timeout, parsing, etc.)
            logger.error(
                "Failed to process category '%s': %s",
                category["search_term"],
                exc,
                exc_info=True,
            )
            logger.info("Continuing to next category...\n")

    logger.info("=" * 60)
    logger.info(
        "DONE — %d / %d categories completed successfully.",
        total_success,
        len(CATEGORIES),
    )
    logger.info("Training data root: %s", BASE_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
