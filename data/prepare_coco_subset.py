"""
data/prepare_coco_subset.py

Create a reproducible COCO validation subset (images + annotations) for faster
evaluation and benchmarking.

This script provides a single function `create_coco_subset` and a command-line
entry point to produce a subset of the COCO validation dataset.

Requirements:
- Uses pathlib.Path for path handling.
- Uses json, random, shutil, argparse from the standard library.
- Graceful error handling and informative print statements.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def create_coco_subset(
    original_ann_path: str | Path,
    image_dir: str | Path,
    output_ann_path: str | Path,
    output_image_dir: str | Path,
    num_images: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Create a COCO-format annotation JSON and a directory of images containing a
    random subset of the original dataset.

    The function:
    - Loads the original COCO annotations JSON.
    - Randomly samples `num_images` image entries (or all if fewer available).
    - Filters annotations to only those referencing the sampled images.
    - Copies sampled image files from `image_dir` to `output_image_dir`.
    - Writes the subset annotations JSON to `output_ann_path`.

    Args:
        original_ann_path: Path to the original COCO annotations JSON.
        image_dir: Directory containing COCO images referenced in the JSON.
        output_ann_path: Destination path for the subset annotations JSON.
        output_image_dir: Destination directory for copied subset images.
        num_images: Number of images to sample (default: 500).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        The subset COCO dictionary that was written to output_ann_path.

    Raises:
        FileNotFoundError: If original_ann_path does not exist.
        RuntimeError: For JSON parsing errors or other unexpected failures.
    """
    orig_ann_p = Path(original_ann_path)
    img_dir_p = Path(image_dir)
    out_ann_p = Path(output_ann_path)
    out_img_dir_p = Path(output_image_dir)

    if not orig_ann_p.exists():
        raise FileNotFoundError(f"Annotations file not found: {orig_ann_p}")

    print(f"Loading annotations from: {orig_ann_p}")
    try:
        with orig_ann_p.open("r", encoding="utf-8") as fh:
            coco = json.load(fh)
    except Exception as exc:
        raise RuntimeError(f"Failed to read or parse annotations JSON: {exc}") from exc

    images: List[Dict[str, Any]] = coco.get("images", [])
    annotations: List[Dict[str, Any]] = coco.get("annotations", [])
    categories: List[Dict[str, Any]] = coco.get("categories", [])
    info: Optional[Dict[str, Any]] = coco.get("info")
    licenses: Optional[List[Dict[str, Any]]] = coco.get("licenses")

    total_images = len(images)
    if total_images == 0:
        print("Warning: no images found in the annotations file.")
    else:
        print(f"Found {total_images} images in the original annotations.")

    # Set seed for reproducibility
    random.seed(seed)

    # Sample images
    if total_images <= num_images:
        sampled_images = list(images)
        actual_sampled = total_images
    else:
        sampled_images = random.sample(images, num_images)
        actual_sampled = num_images

    print(f"Sampled {actual_sampled} images")

    sampled_image_ids = {int(img["id"]) for img in sampled_images}

    # Filter annotations
    filtered_annotations = [
        ann for ann in annotations if int(ann.get("image_id", -1)) in sampled_image_ids
    ]
    kept_annotations = len(filtered_annotations)
    print(f"Kept {kept_annotations} annotations")

    # Build subset dictionary
    subset = {
        "info": info if info is not None else {},
        "licenses": licenses if licenses is not None else [],
        "images": sampled_images,
        "annotations": filtered_annotations,
        "categories": categories,
    }

    # Ensure output annotation directory exists
    try:
        out_ann_p.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to create output annotation directory: {exc}") from exc

    # Save subset JSON
    try:
        with out_ann_p.open("w", encoding="utf-8") as fh:
            json.dump(subset, fh, indent=2)
        print(f"Wrote subset annotations to: {out_ann_p}")
    except Exception as exc:
        raise RuntimeError(f"Failed to write subset annotations JSON: {exc}") from exc

    # Ensure output image directory exists
    try:
        out_img_dir_p.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to create output image directory: {exc}") from exc

    # Copy image files
    missing_count = 0
    copied_count = 0
    for img in sampled_images:
        file_name = img.get("file_name")
        if not file_name:
            print(f"Warning: image entry missing 'file_name': {img.get('id')}")
            missing_count += 1
            continue

        src = img_dir_p / file_name
        dst = out_img_dir_p / file_name

        try:
            if not src.exists():
                print(f"Warning: image file missing: {src}")
                missing_count += 1
                continue

            # Ensure parent directory for destination exists (in case file_name has subdirs)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_count += 1
        except Exception as exc:
            print(f"Warning: failed to copy {src} -> {dst}: {exc}")
            missing_count += 1

    print(
        f"Copied {copied_count} images to {out_img_dir_p} "
        f"({missing_count} missing or failed)."
    )

    # Final summary
    num_categories = len(categories)
    print("Final summary:")
    print(f"  Images       : {len(sampled_images)}")
    print(f"  Annotations  : {len(filtered_annotations)}")
    print(f"  Categories   : {num_categories}")

    return subset


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for creating a COCO subset."""
    parser = argparse.ArgumentParser(
        description="Create a reproducible COCO validation subset (images + annotations)."
    )
    parser.add_argument(
        "--ann",
        type=str,
        default="data/coco/annotations/instances_val2017.json",
        help="Path to original COCO annotations JSON.",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="data/coco/val2017",
        help="Directory containing original COCO images.",
    )
    parser.add_argument(
        "--output-ann",
        type=str,
        default="data/coco/annotations/instances_val2017_subset500.json",
        help="Output path for subset annotations JSON.",
    )
    parser.add_argument(
        "--output-img-dir",
        type=str,
        default="data/coco/val2017_subset500",
        help="Output directory for subset images.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=500,
        help="Number of images to sample for the subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = _parse_args()
    try:
        create_coco_subset(
            original_ann_path=args.ann,
            image_dir=args.img_dir,
            output_ann_path=args.output_ann,
            output_image_dir=args.output_img_dir,
            num_images=args.num_images,
            seed=args.seed,
        )
    except FileNotFoundError as fnf:
        print(f"Error: {fnf}")
    except RuntimeError as rte:
        print(f"Error: {rte}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Unhandled error while creating COCO subset: {exc}")


if __name__ == "__main__":
    main()
