"""
sync_pseudo_dataset_with_semantic.py

Menyelaraskan pseudo dataset ketika folder semantic lama memiliki jumlah file lebih sedikit
atau nama file berbeda dari output pseudo dataset baru.

Kasus yang ditangani:

pseudo_dataset_08072026/
├── disparity/       5972 file
├── img_left/        5972 file
├── landmark/        5972 file
├── semantic/        5059 file  <-- folder referensi
├── vis_wd_landmark/ atau vis_2d_landmark/ 5972 file
└── pseudo_ground_truth_summary.csv 5972 row

Output dibuat sebagai folder baru, tidak menghapus dataset asli:

pseudo_dataset_08072026_synced/
├── disparity/
├── img_left/
├── landmark/
├── semantic/
├── vis_2d_landmark/
└── pseudo_ground_truth_summary.csv

Default alignment mode adalah by-order:
- Ambil N = jumlah file semantic.
- Ambil N row pertama dari summary.csv.
- File disparity/img_left/landmark/vis dicopy berdasarkan output_stem dari summary.
- File semantic lama dicopy berdasarkan urutan natural-sort dan di-rename mengikuti output_stem.

Ini dipilih karena contoh nama semantic lama tidak memiliki stem yang sama dengan
output_stem pseudo dataset baru.

Author: ChatGPT
"""

import argparse
import csv
import os
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LANDMARK_EXTENSIONS = {".csv"}


def natural_key(value):
    text = str(value)
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", text)]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_files(folder, extensions=None):
    folder = Path(folder)
    if not folder.exists():
        return []

    files = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if extensions is not None and p.suffix.lower() not in extensions:
            continue
        files.append(p)

    return sorted(files, key=lambda p: natural_key(p.name))


def detect_existing_dir(root, names, required=True):
    root = Path(root)
    for name in names:
        candidate = root / name
        if candidate.exists() and candidate.is_dir():
            return candidate

    if required:
        raise FileNotFoundError(
            f"Tidak menemukan salah satu folder berikut di {root}: {names}"
        )

    return None


def read_summary_csv(summary_path):
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV tidak ditemukan: {summary_path}")

    with open(summary_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    return rows, fieldnames


def write_summary_csv(output_path, rows, fieldnames):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # Tambahkan kolom baru tanpa menghilangkan kolom lama.
    extra_fields = [
        "synced_index",
        "img_left_file",
        "disparity_file",
        "landmark_file",
        "semantic_file",
        "semantic_source_filename",
        "vis_2d_landmark_file",
        "sync_warning",
    ]

    final_fields = list(fieldnames)
    for field in extra_fields:
        if field not in final_fields:
            final_fields.append(field)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in final_fields})


def copy_or_link(src, dst, mode="copy", overwrite=False, dry_run=False):
    src = Path(src)
    dst = Path(dst)

    if src is None or not src.exists():
        return False

    ensure_dir(dst.parent)

    if dst.exists():
        if overwrite:
            if not dry_run:
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                else:
                    raise RuntimeError(f"Output sudah ada dan bukan file: {dst}")
        else:
            return True

    if dry_run:
        return True

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"mode tidak dikenal: {mode}")

    return True


def first_existing(patterns):
    for p in patterns:
        matches = sorted(Path(p.parent).glob(p.name), key=lambda x: natural_key(x.name))
        if matches:
            return matches[0]
    return None


def find_disparity_file(disparity_dir, stem):
    disparity_dir = Path(disparity_dir)
    patterns = [
        disparity_dir / f"{stem}_disparity_vis.png",
        disparity_dir / f"{stem}_disparity_vis.*",
        disparity_dir / f"{stem}*disparity*.png",
        disparity_dir / f"{stem}*disparity*.*",
    ]
    return first_existing(patterns)


def find_img_left_file(img_left_dir, stem, prefer_rectified=True):
    img_left_dir = Path(img_left_dir)

    if prefer_rectified:
        patterns = [
            img_left_dir / f"{stem}_left_rectified.png",
            img_left_dir / f"{stem}_left_rectified.*",
            img_left_dir / f"{stem}*left_rectified*.png",
            img_left_dir / f"{stem}*left_rectified*.*",
            img_left_dir / f"{stem}_left_raw.png",
            img_left_dir / f"{stem}*left*.png",
            img_left_dir / f"{stem}*left*.*",
        ]
    else:
        patterns = [
            img_left_dir / f"{stem}_left_raw.png",
            img_left_dir / f"{stem}*left_raw*.png",
            img_left_dir / f"{stem}_left_rectified.png",
            img_left_dir / f"{stem}*left*.png",
            img_left_dir / f"{stem}*left*.*",
        ]

    return first_existing(patterns)


def find_landmark_file(landmark_dir, stem):
    landmark_dir = Path(landmark_dir)
    patterns = [
        landmark_dir / f"{stem}_landmarks.csv",
        landmark_dir / f"{stem}_landmark.csv",
        landmark_dir / f"{stem}*landmark*.csv",
        landmark_dir / f"{stem}*.csv",
    ]
    return first_existing(patterns)


def find_vis_file(vis_dir, stem):
    vis_dir = Path(vis_dir)
    patterns = [
        vis_dir / f"{stem}_vis_2d_landmark.png",
        vis_dir / f"{stem}_vis_wd_landmark.png",
        vis_dir / f"{stem}*vis*landmark*.png",
        vis_dir / f"{stem}*landmark*.png",
    ]
    return first_existing(patterns)


def infer_output_stem(row):
    if row.get("output_stem"):
        return row["output_stem"]

    # Fallback jika summary bukan dari script sebelumnya.
    for key in ["stem", "image_id", "id", "filename"]:
        if row.get(key):
            return Path(row[key]).stem

    raise KeyError(
        "Tidak bisa menemukan output_stem di summary. "
        "Pastikan summary memiliki kolom output_stem, filename, stem, image_id, atau id."
    )


def sync_dataset(args):
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root tidak ditemukan: {dataset_root}")

    if dataset_root.resolve() == output_root.resolve():
        raise ValueError(
            "output-root tidak boleh sama dengan dataset-root. "
            "Script ini sengaja membuat folder baru agar dataset asli aman."
        )

    disparity_dir = detect_existing_dir(dataset_root, ["disparity"])
    img_left_dir = detect_existing_dir(dataset_root, ["img_left"])
    landmark_dir = detect_existing_dir(dataset_root, ["landmark", "landmarks"])
    semantic_dir = detect_existing_dir(dataset_root, ["semantic", "mask", "masks"])
    vis_dir = detect_existing_dir(
        dataset_root,
        ["vis_2d_landmark", "vis_wd_landmark", "visualization", "vis"],
        required=not args.allow_missing_vis,
    )

    summary_path = dataset_root / args.summary_name
    rows, fieldnames = read_summary_csv(summary_path)

    semantic_files = list_files(semantic_dir, IMAGE_EXTENSIONS)
    n_semantic = len(semantic_files)

    if args.limit is not None:
        n_keep = min(int(args.limit), n_semantic, len(rows))
    else:
        n_keep = min(n_semantic, len(rows))

    if args.summary_sort_by_stem:
        rows_for_sync = sorted(rows, key=lambda r: natural_key(infer_output_stem(r)))
    else:
        rows_for_sync = list(rows)

    kept_rows = rows_for_sync[:n_keep]
    semantic_kept = semantic_files[:n_keep]

    output_dirs = {
        "disparity": output_root / "disparity",
        "img_left": output_root / "img_left",
        "landmark": output_root / "landmark",
        "semantic": output_root / "semantic",
        "vis_2d_landmark": output_root / "vis_2d_landmark",
    }

    for d in output_dirs.values():
        if d.name == "vis_2d_landmark" and args.allow_missing_vis and vis_dir is None:
            continue
        ensure_dir(d)

    print("=" * 80)
    print("SYNC PSEUDO DATASET")
    print("=" * 80)
    print(f"Dataset root : {dataset_root}")
    print(f"Output root  : {output_root}")
    print(f"Summary rows : {len(rows)}")
    print(f"Semantic ref : {n_semantic}")
    print(f"Keep rows    : {n_keep}")
    print(f"Mode         : order-based alignment")
    print(f"Dry run      : {args.dry_run}")
    print("-" * 80)

    synced_rows = []
    missing_counts = {
        "img_left": 0,
        "disparity": 0,
        "landmark": 0,
        "semantic": 0,
        "vis_2d_landmark": 0,
    }

    for idx, (row, semantic_src) in enumerate(zip(kept_rows, semantic_kept)):
        row = dict(row)
        stem = infer_output_stem(row)
        warnings = []

        img_left_src = find_img_left_file(
            img_left_dir,
            stem,
            prefer_rectified=not args.prefer_raw_left,
        )
        disparity_src = find_disparity_file(disparity_dir, stem)
        landmark_src = find_landmark_file(landmark_dir, stem)
        vis_src = find_vis_file(vis_dir, stem) if vis_dir is not None else None

        # Semantic lama di-align berdasarkan urutan dan di-rename mengikuti stem summary.
        semantic_ext = semantic_src.suffix.lower()
        semantic_dst_name = f"{stem}_semantic{semantic_ext}"

        copy_specs = [
            ("img_left", img_left_src, output_dirs["img_left"] / (img_left_src.name if img_left_src else f"{stem}_left_missing")),
            ("disparity", disparity_src, output_dirs["disparity"] / (disparity_src.name if disparity_src else f"{stem}_disparity_missing")),
            ("landmark", landmark_src, output_dirs["landmark"] / (landmark_src.name if landmark_src else f"{stem}_landmark_missing")),
            ("semantic", semantic_src, output_dirs["semantic"] / semantic_dst_name),
        ]

        if vis_dir is not None:
            copy_specs.append(
                ("vis_2d_landmark", vis_src, output_dirs["vis_2d_landmark"] / (vis_src.name if vis_src else f"{stem}_vis_missing"))
            )

        copied_paths = {}

        for label, src, dst in copy_specs:
            if src is None or not Path(src).exists():
                missing_counts[label] += 1
                warnings.append(f"missing_{label}")
                copied_paths[label] = ""
                continue

            ok = copy_or_link(
                src,
                dst,
                mode=args.mode,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

            if ok:
                copied_paths[label] = str(Path(dst).relative_to(output_root))
            else:
                missing_counts[label] += 1
                warnings.append(f"failed_{label}")
                copied_paths[label] = ""

        row["synced_index"] = idx
        row["img_left_file"] = copied_paths.get("img_left", "")
        row["disparity_file"] = copied_paths.get("disparity", "")
        row["landmark_file"] = copied_paths.get("landmark", "")
        row["semantic_file"] = copied_paths.get("semantic", "")
        row["semantic_source_filename"] = semantic_src.name
        row["vis_2d_landmark_file"] = copied_paths.get("vis_2d_landmark", "")
        row["sync_warning"] = ";".join(warnings)

        synced_rows.append(row)

        if args.verbose and (idx < 5 or warnings):
            print(f"[{idx:05d}] {stem}")
            print(f"       semantic <- {semantic_src.name}")
            if warnings:
                print(f"       WARNING  : {row['sync_warning']}")

    out_summary = output_root / args.summary_name

    if not args.dry_run:
        write_summary_csv(out_summary, synced_rows, fieldnames)

    print("-" * 80)
    print("Missing counts:")
    for key, value in missing_counts.items():
        print(f"  {key:16s}: {value}")

    print("-" * 80)
    print("Output counts:")
    for name, folder in output_dirs.items():
        if folder.exists():
            if name == "landmark":
                count = len(list_files(folder, LANDMARK_EXTENSIONS))
            else:
                count = len(list_files(folder, IMAGE_EXTENSIONS))
            print(f"  {name:16s}: {count}")

    if not args.dry_run:
        print(f"  summary rows    : {len(synced_rows)}")
        print(f"  summary file    : {out_summary}")

    print("=" * 80)

    if any(v > 0 for v in missing_counts.values()):
        print(
            "Catatan: masih ada file missing. Cek kolom sync_warning di summary output. "
            "Jika semantic dan summary tidak urut, jangan pakai hasil ini sebagai final."
        )
    else:
        print("Selesai. Semua folder output sudah diselaraskan berdasarkan semantic reference.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sinkronisasi pseudo dataset dengan folder semantic sebagai referensi jumlah data."
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Folder dataset lama, misalnya pseudo_dataset_08072026."
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Folder output baru, misalnya pseudo_dataset_08072026_synced."
    )
    parser.add_argument(
        "--summary-name",
        default="pseudo_ground_truth_summary.csv",
        help="Nama file summary CSV di dataset root."
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="copy membuat salinan file; symlink membuat symbolic link agar hemat storage."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Timpa file output jika sudah ada."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulasi tanpa menulis/copy file."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batasi jumlah data untuk testing, misalnya --limit 20."
    )
    parser.add_argument(
        "--summary-sort-by-stem",
        action="store_true",
        help="Sort summary berdasarkan output_stem sebelum mengambil N row. Default memakai urutan CSV apa adanya."
    )
    parser.add_argument(
        "--prefer-raw-left",
        action="store_true",
        help="Jika img_left punya raw dan rectified, prefer raw. Default prefer rectified."
    )
    parser.add_argument(
        "--allow-missing-vis",
        action="store_true",
        help="Tetap jalan walaupun folder vis_2d_landmark/vis_wd_landmark tidak ada."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Cetak beberapa mapping awal dan semua warning."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    sync_dataset(args)


if __name__ == "__main__":
    main()
