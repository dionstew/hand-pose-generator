#!/usr/bin/env python3
"""
make_pipeline_demo_v2.py

Improved synchronized pipeline-video generator.

Improvements:
- Prints every resolved input path and discovered file count.
- Supports nested folders recursively.
- Matches images by normalized frame key instead of relying only on sort order.
- Uses CSV output_stem / relative_path when available.
- Warns clearly when a folder is empty or incorrectly specified.
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(text):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", str(text))]


def normalize_stem(name):
    """
    Convert stereo input and generated output filenames into the same frame key.

    Actual naming pattern:

    Stereo input:
      amik-vid-1-S-frame-0-2026-07-06_173426.jpg

    Generated output:
      amik-vid-1-S-hand-img_2026-07-06_173426__
      amik-vid-1-S-frame-0-2026-07-06_173426_disparity_vis_<label>.png

    Shared key:
      amik-vid-1-S-frame-0-2026-07-06_173426

    Strategy:
    1. Remove the file extension.
    2. If "__" exists, keep only the part after the last "__".
    3. Remove known generated-output suffix combinations.
    4. Fall back to extracting a "...-frame-<number>-<timestamp>" pattern.
    """
    stem = Path(name).stem.lower()

    # Output names prepend a hand-image identifier before "__".
    if "__" in stem:
        stem = stem.split("__")[-1]

    # Remove the complete generated suffix first.
    # Longest patterns must be checked before shorter ones.
    suffixes = [
        "_vis_2d_landmark",
        "_left_rectified",
        "_left_raw",
        "_landmarks",
        "_landmark",
        "_axis",
        "_pose",
        "_overlay",
        "_disparity_vis",
    ]

    for suffix in sorted(suffixes, key=len, reverse=True):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break

    # Robust fallback: isolate the real video-frame identifier.
    # Supports names such as:
    # amik-vid-1-S-frame-0-2026-07-06_173426
    match = re.search(
        r"([a-z0-9_-]+-frame-\d+-\d{4}-\d{2}-\d{2}_\d{6})",
        stem,
        flags=re.IGNORECASE,
    )
    if match:
        stem = match.group(1)

    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem


def list_images(folder, label):
    if not folder:
        print(f"[WARN] {label}: no folder argument provided")
        return []

    folder = Path(folder).expanduser().resolve()
    print(f"[PATH] {label}: {folder}")

    if not folder.exists():
        print(f"[ERROR] {label}: folder does not exist")
        return []

    if not folder.is_dir():
        print(f"[ERROR] {label}: path is not a directory")
        return []

    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    ]
    files = sorted(files, key=lambda p: natural_key(p.relative_to(folder)))

    print(f"[FOUND] {label}: {len(files)} image(s)")
    if files:
        print(f"        first: {files[0]}")
        print(f"        last : {files[-1]}")
    return files


def build_index(files):
    index = {}
    for p in files:
        key = normalize_stem(p.name)
        if key not in index:
            index[key] = p
    return index


def load_records(csv_path):
    if not csv_path:
        print("[WARN] CSV: no path provided")
        return []

    path = Path(csv_path).expanduser().resolve()
    print(f"[PATH] CSV: {path}")

    if not path.exists():
        print("[ERROR] CSV file does not exist")
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    print(f"[FOUND] CSV: {len(records)} record(s)")
    return records


def record_key(record):
    for field in ("output_stem", "filename", "relative_path"):
        value = record.get(field)
        if value:
            return normalize_stem(value)
    return ""


def fit(img, w, h):
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    if img is None:
        cv2.putText(canvas, "Frame unavailable", (30, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 100), 2)
        return canvas

    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    x = (w - nw) // 2
    y = (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def add_title(panel, text):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (255, 255, 255), -1)
    cv2.putText(panel, text, (14, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (20, 55, 95), 2, cv2.LINE_AA)
    return panel


def pose_panel(record, w, h):
    panel = np.full((h, w, 3), 250, dtype=np.uint8)
    cv2.putText(panel, "Pseudo Ground Truth", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 55, 95), 2)

    if not record:
        lines = ["CSV record unavailable"]
    else:
        lines = [
            f"Status: {record.get('status', 'N/A')}",
            "T = [{}, {}, {}]".format(
                record.get("Tx", ""),
                record.get("Ty", ""),
                record.get("Tz", ""),
            ),
            "q = [{}, {}, {}, {}]".format(
                record.get("qw", ""),
                record.get("qx", ""),
                record.get("qy", ""),
                record.get("qz", ""),
            ),
        ]

    y = 92
    for line in lines:
        cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (45, 45, 45), 1, cv2.LINE_AA)
        y += 42
    return panel


def read_image(path):
    if path is None:
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stereo-dir", required=True)
    ap.add_argument("--rectified-dir", required=True)
    ap.add_argument("--disparity-dir", required=True)
    ap.add_argument("--landmark-dir", required=True)
    ap.add_argument("--axis-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output", default="hand_pipeline_demo.mp4")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--panel-width", type=int, default=640)
    ap.add_argument("--panel-height", type=int, default=360)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    stereo_files = list_images(args.stereo_dir, "Stereo")
    rectified_files = list_images(args.rectified_dir, "Rectified")
    disparity_files = list_images(args.disparity_dir, "Disparity")
    landmark_files = list_images(args.landmark_dir, "Landmark")
    axis_files = list_images(args.axis_dir, "Axis")
    records = load_records(args.csv)

    indices = {
        "Stereo Input": build_index(stereo_files),
        "Rectified Left": build_index(rectified_files),
        "Disparity": build_index(disparity_files),
        "2D Hand Landmarks": build_index(landmark_files),
        "Hand Coordinate Frame": build_index(axis_files),
    }

    record_map = {record_key(r): r for r in records if record_key(r)}

    # Prefer CSV order because output_stem was generated by the pipeline.
    keys = [record_key(r) for r in records if record_key(r)]

    # Fallback: union of all discovered image keys.
    if not keys:
        key_union = set()
        for idx in indices.values():
            key_union.update(idx.keys())
        keys = sorted(key_union, key=natural_key)

    # Remove duplicate keys while preserving order.
    seen = set()
    keys = [k for k in keys if k and not (k in seen or seen.add(k))]

    if args.max_frames > 0:
        keys = keys[:args.max_frames]

    print(f"[MATCH] Timeline contains {len(keys)} frame key(s)")
    if not keys:
        raise RuntimeError(
            "No frames were discovered. Check the printed absolute paths and folder names."
        )

    for label, idx in indices.items():
        matched = sum(1 for k in keys if k in idx)
        print(f"[MATCH] {label}: {matched}/{len(keys)}")

    W, H = args.panel_width, args.panel_height
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (3 * W, 2 * H),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not initialize MP4 writer.")

    panel_names = list(indices.keys())

    for i, key in enumerate(keys):
        panels = []
        for label in panel_names:
            img = read_image(indices[label].get(key))
            panels.append(add_title(fit(img, W, H), label))

        panels.append(pose_panel(record_map.get(key), W, H))

        frame = np.vstack([
            np.hstack(panels[:3]),
            np.hstack(panels[3:6]),
        ])

        cv2.putText(
            frame,
            f"{i + 1}/{len(keys)}  key={key}",
            (18, 2 * H - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
        print(f"Add new frames for video at: {i}th frame of {len(keys)} frames")

    writer.release()
    print(f"[DONE] Saved: {output}")
    print(f"[DONE] Frames: {len(keys)} | FPS: {args.fps}")


if __name__ == "__main__":
    main()
