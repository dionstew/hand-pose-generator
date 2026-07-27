"""
pseudo_dataset_extraction_pipeline.py

Single-file pipeline untuk:
1. Membaca banyak gambar stereo dari folder utama dan subfolder data
2. Split left-right stereo image
3. Rectification memakai file kalibrasi .npz
4. StereoSGBM + WLS disparity
5. MediaPipe HandLandmarker pada left rectified image
6. Landmark 2D -> landmark 3D memakai disparity + Q
7. Estimasi pseudo ground-truth:
   - translation t_cam_hand
   - rotation R_cam_hand
   - quaternion [w, x, y, z]
8. Menampilkan hasil di terminal dan menyimpan CSV.
9. Menyimpan output dalam satu folder: vis_2d_landmark, disparity, landmark, img_left.

Catatan:
- Input diasumsikan berupa stereo image side-by-side:
  kiri di setengah gambar kiri, kanan di setengah gambar kanan.
- Jika SQUARE_SIZE saat kalibrasi = 0.03 meter, maka hasil X,Y,Z/T dalam meter.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    MEDIAPIPE_IMPORT_ERROR = exc
else:
    MEDIAPIPE_IMPORT_ERROR = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]


def natural_key(path):
    """Natural sorting untuk nama file: frame-2 sebelum frame-10."""
    text = str(path)
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", text)]


def list_images(input_dir, recursive=True):
    """
    Mengambil gambar dari folder utama.

    Default recursive=True supaya struktur seperti ini bisa diproses:
        folder_utama/
            folder_data_1/gambarstereo1-1.jpg
            folder_data_2/gambarstereo2-1.jpg

    Jika recursive=False, hanya gambar langsung di input_dir yang dibaca.
    """
    input_dir = Path(input_dir)

    if recursive:
        candidates = input_dir.rglob("*")
    else:
        candidates = input_dir.iterdir()

    images = [
        p for p in candidates
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(images, key=lambda p: natural_key(p.relative_to(input_dir)))


def sanitize_name(text):
    """Membuat nama file aman dari relative path folder data."""
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "item"


def make_output_stem(image_path, input_root):
    """
    Membuat output stem unik untuk semua gambar dari banyak subfolder.

    Contoh:
        input_root/folder data 1/gambarstereo1-1.jpg
        -> folder_data_1__gambarstereo1-1
    """
    image_path = Path(image_path)
    input_root = Path(input_root)

    try:
        rel = image_path.relative_to(input_root)
    except ValueError:
        rel = image_path.name

    rel = Path(rel)
    parts = list(rel.parts)

    if len(parts) == 0:
        return sanitize_name(image_path.stem)

    safe_parts = []
    for idx, part in enumerate(parts):
        if idx == len(parts) - 1:
            safe_parts.append(sanitize_name(Path(part).stem))
        else:
            safe_parts.append(sanitize_name(part))

    return "__".join([p for p in safe_parts if p])


def load_calibration(npz_path):
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"File kalibrasi tidak ditemukan: {npz_path}")

    with np.load(str(npz_path)) as data:
        required_keys = ["K_left", "D_left", "K_right", "D_right", "R", "T"]
        missing = [k for k in required_keys if k not in data.files]

        if missing:
            raise KeyError(
                f"File kalibrasi tidak memiliki key: {missing}. "
                f"Key tersedia: {data.files}"
            )

        calib = {
            "K_left": data["K_left"],
            "D_left": data["D_left"],
            "K_right": data["K_right"],
            "D_right": data["D_right"],
            "R": data["R"],
            "T": data["T"],
        }

    return calib


def split_stereo_side_by_side(image_bgr):
    h, w = image_bgr.shape[:2]

    if w % 2 != 0:
        raise ValueError(
            f"Lebar gambar stereo harus genap agar bisa dibagi kiri-kanan. "
            f"Shape gambar: {image_bgr.shape}"
        )

    left = image_bgr[:, :w // 2]
    right = image_bgr[:, w // 2:]

    return left, right


def rectify_stereo_image(
    image_path,
    calib,
    resize_width=2560,
    resize_height=720,
    alpha=0.0
):
    """
    Membaca satu stereo image side-by-side, split, lalu rectification.

    Return:
        gray_left_rectified
        gray_right_rectified
        left_rectified_bgr
        right_rectified_bgr
        Q
        P1
        left_bgr
        right_bgr
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    if resize_width is not None and resize_height is not None:
        if resize_width > 0 and resize_height > 0:
            image = cv2.resize(
                image,
                (int(resize_width), int(resize_height)),
                interpolation=cv2.INTER_CUBIC
            )

    image = cv2.GaussianBlur(image, (3, 3), 0.5)

    left_bgr, right_bgr = split_stereo_side_by_side(image)

    image_size_left = (left_bgr.shape[1], left_bgr.shape[0])
    image_size_right = (right_bgr.shape[1], right_bgr.shape[0])

    if image_size_left != image_size_right:
        raise ValueError(
            f"Ukuran kiri dan kanan tidak sama: {image_size_left} vs {image_size_right}"
        )

    K_left = calib["K_left"]
    D_left = calib["D_left"]
    K_right = calib["K_right"]
    D_right = calib["D_right"]
    R = calib["R"]
    T = calib["T"]

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_left,
        D_left,
        K_right,
        D_right,
        image_size_left,
        R,
        T,
        alpha=alpha
    )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left,
        D_left,
        R1,
        P1,
        image_size_left,
        cv2.CV_32FC1
    )

    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right,
        D_right,
        R2,
        P2,
        image_size_right,
        cv2.CV_32FC1
    )

    left_rectified_bgr = cv2.remap(
        left_bgr,
        map_left_x,
        map_left_y,
        cv2.INTER_LINEAR
    )

    right_rectified_bgr = cv2.remap(
        right_bgr,
        map_right_x,
        map_right_y,
        cv2.INTER_LINEAR
    )

    gray_left = cv2.cvtColor(left_rectified_bgr, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_rectified_bgr, cv2.COLOR_BGR2GRAY)

    return gray_left, gray_right, left_rectified_bgr, right_rectified_bgr, Q, P1, left_bgr, right_bgr


def compute_disparity_wls(
    gray_left,
    gray_right,
    num_disparity_blocks=12,
    block_size=3,
    filter_cap=63,
    lmbda=60000,
    sigma=1.2,
    uniqueness_ratio=15,
    speckle_window_size=150,
    speckle_range=2
):
    """
    Hitung disparity memakai StereoSGBM + WLS.

    num_disparity_blocks:
        Akan dikalikan 16.
        Contoh: 12 -> actual numDisparities = 192.

    Return:
        displ_raw
        dispr_raw
        disparity_float
        disparity_vis
    """

    if not hasattr(cv2, "ximgproc"):
        raise ImportError(
            "cv2.ximgproc tidak ditemukan. Install opencv-contrib-python:\n"
            "pip install opencv-contrib-python"
        )

    actual_num_disparities = int(num_disparity_blocks) * 16

    if actual_num_disparities <= 0 or actual_num_disparities % 16 != 0:
        raise ValueError("actual numDisparities harus positif dan kelipatan 16.")

    if block_size % 2 == 0:
        raise ValueError("block_size harus bilangan ganjil, misalnya 3, 5, 7.")

    channels = 1
    p1 = 8 * channels * block_size ** 2
    p2 = 32 * channels * block_size ** 2

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=actual_num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=filter_cap,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(
        matcher_left=left_matcher
    )
    wls_filter.setLambda(float(lmbda))
    wls_filter.setSigmaColor(float(sigma))

    displ_raw = left_matcher.compute(gray_left, gray_right).astype(np.int16)
    dispr_raw = right_matcher.compute(gray_right, gray_left).astype(np.int16)

    filtered_raw = wls_filter.filter(
        displ_raw,
        gray_left,
        None,
        dispr_raw
    ).astype(np.int16)

    disparity_float = filtered_raw.astype(np.float32) / 16.0

    disparity_vis = cv2.normalize(
        disparity_float,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    return displ_raw, dispr_raw, disparity_float, disparity_vis


class HandLandmarkerWrapper:
    def __init__(
        self,
        model_path,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        if mp is None:
            raise ImportError(
                f"mediapipe gagal di-import: {MEDIAPIPE_IMPORT_ERROR}\n"
                "Install dengan: pip install mediapipe"
            )

        if not hasattr(mp, "tasks"):
            raise AttributeError(
                "mediapipe.tasks tidak ditemukan. "
                "Pastikan memakai versi MediaPipe yang mendukung Tasks API."
            )

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model MediaPipe HandLandmarker tidak ditemukan: {model_path}\n"
                "Download/simpan file hand_landmarker.task lalu masukkan path-nya "
                "ke argumen --hand-model."
            )

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=int(num_hands),
            min_hand_detection_confidence=float(min_hand_detection_confidence),
            min_hand_presence_confidence=float(min_hand_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    def detect(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        return self.landmarker.detect(mp_image)

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None


def hand_landmarker_result_to_pixels(result, image_shape):
    h, w = image_shape[:2]

    if result.hand_landmarks is None or len(result.hand_landmarks) == 0:
        return None

    hand_landmarks = result.hand_landmarks[0]

    landmarks_2d = []

    for lm in hand_landmarks:
        u = int(round(lm.x * w))
        v = int(round(lm.y * h))
        landmarks_2d.append([u, v])

    return np.array(landmarks_2d, dtype=np.int32)


def get_3d_landmarks_from_disparity(
    landmarks_2d,
    disparity_float,
    Q,
    window_size=5,
    min_disparity=0.1
):
    """
    Landmark 2D -> Landmark 3D memakai disparity map + Q.

    Output:
        landmarks_3d shape (21, 3)
        satuan mengikuti file kalibrasi.
        Jika SQUARE_SIZE=0.03 meter, maka satuannya meter.
    """

    if landmarks_2d is None:
        return None

    h, w = disparity_float.shape[:2]

    points_3d_map = cv2.reprojectImageTo3D(
        disparity_float.astype(np.float32),
        Q
    )

    half = int(window_size) // 2
    landmarks_3d = []

    for u, v in landmarks_2d:
        if u < 0 or u >= w or v < 0 or v >= h:
            landmarks_3d.append([np.nan, np.nan, np.nan])
            continue

        x1 = max(int(u) - half, 0)
        x2 = min(int(u) + half + 1, w)
        y1 = max(int(v) - half, 0)
        y2 = min(int(v) + half + 1, h)

        disparity_patch = disparity_float[y1:y2, x1:x2]
        points_patch = points_3d_map[y1:y2, x1:x2]

        valid_mask = disparity_patch > float(min_disparity)
        valid_mask &= np.isfinite(disparity_patch)
        valid_mask &= np.isfinite(points_patch).all(axis=-1)

        if np.sum(valid_mask) == 0:
            landmarks_3d.append([np.nan, np.nan, np.nan])
            continue

        valid_points = points_patch[valid_mask]
        point_3d = np.median(valid_points, axis=0)
        landmarks_3d.append(point_3d)

    return np.array(landmarks_3d, dtype=np.float32)


def normalize_vector(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)

    if norm < eps:
        return None

    return v / norm


def compute_palm_normal_from_landmarks(
    landmarks_3d,
    palm_indices=(0, 5, 9, 13, 17),
    normal_facing_camera=True
):
    palm_points = landmarks_3d[list(palm_indices)]

    valid = np.isfinite(palm_points).all(axis=1)
    palm_points = palm_points[valid]

    if len(palm_points) < 3:
        raise ValueError(
            f"Titik palm valid kurang dari 3. Valid: {len(palm_points)}"
        )

    palm_center = np.mean(palm_points, axis=0)
    centered = palm_points - palm_center

    _, _, vh = np.linalg.svd(centered)
    palm_normal = normalize_vector(vh[-1])

    if palm_normal is None:
        raise ValueError("Normal palm gagal dihitung.")

    if normal_facing_camera:
        direction_to_camera = -palm_center

        if np.dot(palm_normal, direction_to_camera) < 0:
            palm_normal = -palm_normal

    return palm_center.astype(np.float32), palm_normal.astype(np.float32)


def compute_hand_coordinate_frame(landmarks_3d):
    """
    Definisi frame lokal tangan:
        Origin = mean landmark palm [0,5,9,13,17]
        Y_hand = wrist -> middle_mcp
        Z_hand = normal palm
        X_hand = Y x Z

    Return:
        R_cam_hand:
            Kolom 0 = X_hand dalam koordinat kamera
            Kolom 1 = Y_hand dalam koordinat kamera
            Kolom 2 = Z_hand dalam koordinat kamera
        t_cam_hand:
            translation/origin tangan dalam koordinat kamera
    """

    palm_center, z_axis = compute_palm_normal_from_landmarks(
        landmarks_3d,
        palm_indices=(0, 5, 9, 13, 17),
        normal_facing_camera=True
    )

    wrist = landmarks_3d[0]
    middle_mcp = landmarks_3d[9]

    if not np.isfinite(wrist).all() or not np.isfinite(middle_mcp).all():
        raise ValueError("Landmark wrist atau middle_mcp tidak valid.")

    y_axis = middle_mcp.astype(np.float64) - wrist.astype(np.float64)
    y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
    y_axis = normalize_vector(y_axis)

    if y_axis is None:
        raise ValueError("Gagal membuat Y axis.")

    x_axis = np.cross(y_axis, z_axis)
    x_axis = normalize_vector(x_axis)

    if x_axis is None:
        raise ValueError("Gagal membuat X axis.")

    y_axis = np.cross(z_axis, x_axis)
    y_axis = normalize_vector(y_axis)

    R_cam_hand = np.stack(
        [x_axis, y_axis, z_axis],
        axis=1
    ).astype(np.float32)

    t_cam_hand = palm_center.astype(np.float32)

    return R_cam_hand, t_cam_hand


def rotation_matrix_to_quaternion_wxyz(R):
    """
    Konversi rotation matrix 3x3 ke quaternion [w, x, y, z].
    """
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q_norm = np.linalg.norm(q)

    if q_norm > 0:
        q = q / q_norm

    return q.astype(np.float32)


def matrix_to_flat_list(R):
    return [float(x) for x in np.asarray(R).reshape(-1)]


def project_3d_to_2d(points_3d, K):
    points_3d = np.asarray(points_3d, dtype=np.float32)

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    eps = 1e-8
    Z_safe = np.where(np.abs(Z) < eps, np.nan, Z)

    u = K[0, 0] * X / Z_safe + K[0, 2]
    v = K[1, 1] * Y / Z_safe + K[1, 2]

    return np.stack([u, v], axis=1)


def draw_pseudo_pose_2d(
    image_bgr,
    landmarks_2d,
    R_cam_hand,
    t_cam_hand,
    K_rect,
    axis_length=0.05,
    draw_id=True
):
    img_draw = image_bgr.copy()
    h, w = img_draw.shape[:2]

    if landmarks_2d is not None:
        for i, j in HAND_CONNECTIONS:
            u1, v1 = landmarks_2d[i]
            u2, v2 = landmarks_2d[j]

            if (
                0 <= u1 < w and 0 <= v1 < h and
                0 <= u2 < w and 0 <= v2 < h
            ):
                cv2.line(
                    img_draw,
                    (int(u1), int(v1)),
                    (int(u2), int(v2)),
                    (0, 255, 0),
                    2
                )

        for idx, (u, v) in enumerate(landmarks_2d):
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(
                    img_draw,
                    (int(u), int(v)),
                    5,
                    (0, 0, 255),
                    -1
                )

                if draw_id:
                    cv2.putText(
                        img_draw,
                        str(idx),
                        (int(u) + 5, int(v) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA
                    )

    center = np.asarray(t_cam_hand, dtype=np.float32).reshape(3)

    axis_points_3d = np.stack(
        [
            center,
            center + axis_length * R_cam_hand[:, 0],
            center + axis_length * R_cam_hand[:, 1],
            center + axis_length * R_cam_hand[:, 2],
        ],
        axis=0
    )

    axis_points_2d = project_3d_to_2d(axis_points_3d, K_rect)

    if np.isfinite(axis_points_2d).all():
        c = axis_points_2d[0].astype(int)
        x = axis_points_2d[1].astype(int)
        y = axis_points_2d[2].astype(int)
        z = axis_points_2d[3].astype(int)

        c_tuple = (int(c[0]), int(c[1]))

        # OpenCV BGR:
        # X merah, Y hijau, Z biru.
        cv2.arrowedLine(img_draw, c_tuple, (int(x[0]), int(x[1])), (0, 0, 255), 4, tipLength=0.25)
        cv2.arrowedLine(img_draw, c_tuple, (int(y[0]), int(y[1])), (0, 255, 0), 4, tipLength=0.25)
        cv2.arrowedLine(img_draw, c_tuple, (int(z[0]), int(z[1])), (255, 0, 0), 4, tipLength=0.25)

        cv2.putText(img_draw, "X", (int(x[0]), int(x[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img_draw, "Y", (int(y[0]), int(y[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_draw, "Z", (int(z[0]), int(z[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return img_draw


def process_image(image_path, calib, detector, args):
    gray_left, gray_right, left_rectified_bgr, right_rectified_bgr, Q, P1, left_bgr, right_bgr = rectify_stereo_image(
        image_path=image_path,
        calib=calib,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        alpha=args.rectify_alpha
    )

    _, _, disparity_float, disparity_vis = compute_disparity_wls(
        gray_left=gray_left,
        gray_right=gray_right,
        num_disparity_blocks=args.num_disparity_blocks,
        block_size=args.block_size,
        filter_cap=args.filter_cap,
        lmbda=args.wls_lambda,
        sigma=args.wls_sigma,
        uniqueness_ratio=args.uniqueness_ratio,
        speckle_window_size=args.speckle_window_size,
        speckle_range=args.speckle_range,
    )

    result = detector.detect(left_rectified_bgr)
    landmarks_2d = hand_landmarker_result_to_pixels(
        result,
        left_rectified_bgr.shape
    )

    if landmarks_2d is None:
        raise RuntimeError("Hand tidak terdeteksi.")

    landmarks_3d = get_3d_landmarks_from_disparity(
        landmarks_2d=landmarks_2d,
        disparity_float=disparity_float,
        Q=Q,
        window_size=args.landmark_window_size,
        min_disparity=args.landmark_min_disparity
    )

    valid_landmarks = int(np.sum(np.isfinite(landmarks_3d).all(axis=1)))

    if valid_landmarks < args.min_valid_landmarks:
        raise RuntimeError(
            f"Landmark 3D valid kurang. Valid={valid_landmarks}, "
            f"minimum={args.min_valid_landmarks}"
        )

    R_cam_hand, t_cam_hand = compute_hand_coordinate_frame(landmarks_3d)
    q_wxyz = rotation_matrix_to_quaternion_wxyz(R_cam_hand)

    K_rect = P1[:3, :3]

    stem = make_output_stem(image_path, args.input_dir)
    rel_path = str(Path(image_path).relative_to(Path(args.input_dir)))

    if args.save_img_left:
        left_dir = Path(args.output_dir) / "img_left"
        left_dir.mkdir(parents=True, exist_ok=True)

        if args.left_image_mode == "rectified":
            cv2.imwrite(str(left_dir / f"{stem}_left_rectified.png"), left_rectified_bgr)
        elif args.left_image_mode == "raw":
            cv2.imwrite(str(left_dir / f"{stem}_left_raw.png"), left_bgr)
        elif args.left_image_mode == "both":
            cv2.imwrite(str(left_dir / f"{stem}_left_rectified.png"), left_rectified_bgr)
            cv2.imwrite(str(left_dir / f"{stem}_left_raw.png"), left_bgr)
        else:
            raise ValueError(f"left_image_mode tidak dikenal: {args.left_image_mode}")

    if args.save_vis:
        vis_dir = Path(args.output_dir) / "vis_2d_landmark"
        vis_dir.mkdir(parents=True, exist_ok=True)

        img_vis = draw_pseudo_pose_2d(
            image_bgr=left_rectified_bgr,
            landmarks_2d=landmarks_2d,
            R_cam_hand=R_cam_hand,
            t_cam_hand=t_cam_hand,
            K_rect=K_rect,
            axis_length=args.axis_length,
            draw_id=True
        )

        cv2.imwrite(str(vis_dir / f"{stem}_vis_2d_landmark.png"), img_vis)

    if args.save_disparity:
        disp_dir = Path(args.output_dir) / "disparity"
        disp_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(disp_dir / f"{stem}_disparity_vis.png"), disparity_vis)

    if args.save_landmarks:
        lm_dir = Path(args.output_dir) / "landmark"
        lm_dir.mkdir(parents=True, exist_ok=True)
        lm_path = lm_dir / f"{stem}_landmarks.csv"

        with open(lm_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["landmark_id", "u", "v", "X", "Y", "Z"])

            for idx in range(21):
                u, v = landmarks_2d[idx]
                X, Y, Z = landmarks_3d[idx]
                writer.writerow([idx, int(u), int(v), float(X), float(Y), float(Z)])

    record = {
        "filename": Path(image_path).name,
        "relative_path": rel_path,
        "output_stem": stem,
        "status": "OK",
        "valid_landmarks_3d": valid_landmarks,
        "Tx": float(t_cam_hand[0]),
        "Ty": float(t_cam_hand[1]),
        "Tz": float(t_cam_hand[2]),
        "qw": float(q_wxyz[0]),
        "qx": float(q_wxyz[1]),
        "qy": float(q_wxyz[2]),
        "qz": float(q_wxyz[3]),
    }

    for i, value in enumerate(matrix_to_flat_list(R_cam_hand)):
        record[f"R{i // 3}{i % 3}"] = value

    return record, R_cam_hand, t_cam_hand, q_wxyz


def print_pose_to_terminal(record, R_cam_hand, t_cam_hand, q_wxyz):
    print("\n" + "=" * 80)
    print(f"[OK] {record['filename']}")
    print("-" * 80)
    print("Translation t_cam_hand [meter jika SQUARE_SIZE=0.03]:")
    print(f"  Tx = {t_cam_hand[0]: .6f}")
    print(f"  Ty = {t_cam_hand[1]: .6f}")
    print(f"  Tz = {t_cam_hand[2]: .6f}")

    print("\nRotation matrix R_cam_hand:")
    for row in R_cam_hand:
        print("  [" + "  ".join(f"{v: .6f}" for v in row) + "]")

    print("\nQuaternion [w, x, y, z]:")
    print(
        f"  [{q_wxyz[0]: .6f}, {q_wxyz[1]: .6f}, "
        f"{q_wxyz[2]: .6f}, {q_wxyz[3]: .6f}]"
    )

    print(f"\nValid 3D landmarks: {record['valid_landmarks_3d']}/21")


def write_summary_csv(output_csv, records):
    if len(records) == 0:
        return

    keys = [
        "filename",
        "relative_path",
        "output_stem",
        "status",
        "valid_landmarks_3d",
        "Tx", "Ty", "Tz",
        "qw", "qx", "qy", "qz",
        "R00", "R01", "R02",
        "R10", "R11", "R12",
        "R20", "R21", "R22",
        "error",
    ]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for rec in records:
            row = {k: rec.get(k, "") for k in keys}
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch pseudo-GT extraction: stereo image folder -> translation + rotation."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder utama yang berisi subfolder-subfolder data stereo image side-by-side."
    )

    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Jika dipakai, hanya memproses gambar langsung di input-dir, tidak masuk ke subfolder."
    )

    parser.add_argument(
        "--calib",
        required=True,
        help="Path file kalibrasi .npz berisi K_left, D_left, K_right, D_right, R, T."
    )

    parser.add_argument(
        "--hand-model",
        required=True,
        help="Path model MediaPipe hand_landmarker.task."
    )

    parser.add_argument(
        "--output-dir",
        default="pseudo_dataset_output",
        help="Folder output CSV dan visualisasi."
    )

    parser.add_argument(
        "--resize-width",
        type=int,
        default=2560,
        help="Resize lebar stereo image sebelum split. Pakai 0 untuk tidak resize."
    )

    parser.add_argument(
        "--resize-height",
        type=int,
        default=720,
        help="Resize tinggi stereo image sebelum split. Pakai 0 untuk tidak resize."
    )

    parser.add_argument(
        "--rectify-alpha",
        type=float,
        default=0.0,
        help="Alpha stereoRectify. 0 crop valid area, 1 full image."
    )

    parser.add_argument(
        "--num-disparity-blocks",
        type=int,
        default=12,
        help="numDisparities = nilai ini * 16. Default 12 -> 192."
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=3,
        help="StereoSGBM blockSize. Harus ganjil."
    )

    parser.add_argument(
        "--filter-cap",
        type=int,
        default=63,
        help="StereoSGBM preFilterCap."
    )

    parser.add_argument(
        "--wls-lambda",
        type=float,
        default=60000,
        help="WLS lambda."
    )

    parser.add_argument(
        "--wls-sigma",
        type=float,
        default=1.2,
        help="WLS sigmaColor."
    )

    parser.add_argument(
        "--uniqueness-ratio",
        type=int,
        default=15,
        help="StereoSGBM uniquenessRatio."
    )

    parser.add_argument(
        "--speckle-window-size",
        type=int,
        default=150,
        help="StereoSGBM speckleWindowSize."
    )

    parser.add_argument(
        "--speckle-range",
        type=int,
        default=2,
        help="StereoSGBM speckleRange."
    )

    parser.add_argument(
        "--landmark-window-size",
        type=int,
        default=5,
        help="Window patch di sekitar landmark 2D untuk mengambil median titik 3D."
    )

    parser.add_argument(
        "--landmark-min-disparity",
        type=float,
        default=0.1,
        help="Minimum disparity valid untuk landmark 3D."
    )

    parser.add_argument(
        "--min-valid-landmarks",
        type=int,
        default=8,
        help="Minimum jumlah landmark 3D valid agar frame tangan dihitung."
    )

    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.05,
        help="Panjang sumbu lokal untuk visualisasi 2D. Satuan mengikuti kalibrasi."
    )

    parser.add_argument(
        "--save-vis",
        dest="save_vis",
        action="store_true",
        default=True,
        help="Simpan visualisasi 2D landmark + sumbu pose. Default: aktif."
    )

    parser.add_argument(
        "--no-save-vis",
        dest="save_vis",
        action="store_false",
        help="Nonaktifkan penyimpanan visualisasi 2D."
    )

    parser.add_argument(
        "--save-disparity",
        dest="save_disparity",
        action="store_true",
        default=True,
        help="Simpan disparity visualization. Default: aktif."
    )

    parser.add_argument(
        "--no-save-disparity",
        dest="save_disparity",
        action="store_false",
        help="Nonaktifkan penyimpanan disparity visualization."
    )

    parser.add_argument(
        "--save-landmarks",
        dest="save_landmarks",
        action="store_true",
        default=True,
        help="Simpan landmark 2D/3D per gambar ke CSV. Default: aktif."
    )

    parser.add_argument(
        "--no-save-landmarks",
        dest="save_landmarks",
        action="store_false",
        help="Nonaktifkan penyimpanan landmark CSV."
    )

    parser.add_argument(
        "--save-img-left",
        dest="save_img_left",
        action="store_true",
        default=True,
        help="Simpan image kiri ke folder img_left. Default: aktif."
    )

    parser.add_argument(
        "--no-save-img-left",
        dest="save_img_left",
        action="store_false",
        help="Nonaktifkan penyimpanan image kiri."
    )

    parser.add_argument(
        "--left-image-mode",
        choices=["rectified", "raw", "both"],
        default="rectified",
        help="Jenis image kiri yang disimpan di img_left. Saran: rectified. Default: rectified."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.resize_width <= 0 or args.resize_height <= 0:
        args.resize_width = None
        args.resize_height = None

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Struktur output dibuat sejak awal agar konsisten.
    if args.save_vis:
        (output_dir / "vis_2d_landmark").mkdir(parents=True, exist_ok=True)
    if args.save_disparity:
        (output_dir / "disparity").mkdir(parents=True, exist_ok=True)
    if args.save_landmarks:
        (output_dir / "landmark").mkdir(parents=True, exist_ok=True)
    if args.save_img_left:
        (output_dir / "img_left").mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir, recursive=not args.non_recursive)

    if len(images) == 0:
        raise FileNotFoundError(f"Tidak ada gambar di folder: {input_dir}")

    print("Input folder :", input_dir)
    print("Recursive    :", not args.non_recursive)
    print("Total images :", len(images))
    print("Calibration  :", args.calib)
    print("Hand model   :", args.hand_model)
    print("Output folder:", output_dir)

    calib = load_calibration(args.calib)

    baseline = float(np.linalg.norm(calib["T"]))
    print(f"Stereo baseline dari T: {baseline:.6f} satuan kalibrasi")

    detector = HandLandmarkerWrapper(
        model_path=args.hand_model,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    records = []

    try:
        for idx, image_path in enumerate(images, start=1):
            print("\n" + "#" * 80)
            print(f"[{idx}/{len(images)}] Processing: {image_path.name}")

            try:
                record, R_cam_hand, t_cam_hand, q_wxyz = process_image(
                    image_path=image_path,
                    calib=calib,
                    detector=detector,
                    args=args
                )

                print_pose_to_terminal(
                    record=record,
                    R_cam_hand=R_cam_hand,
                    t_cam_hand=t_cam_hand,
                    q_wxyz=q_wxyz
                )

            except Exception as exc:
                print(f"[FAILED] {image_path.name}")
                print(f"Reason: {exc}")

                record = {
                    "filename": image_path.name,
                    "relative_path": str(image_path.relative_to(input_dir)),
                    "output_stem": make_output_stem(image_path, input_dir),
                    "status": "FAILED",
                    "error": str(exc),
                }

            records.append(record)

    finally:
        detector.close()

    summary_csv = output_dir / "pseudo_ground_truth_summary.csv"
    write_summary_csv(summary_csv, records)

    ok_count = sum(1 for r in records if r.get("status") == "OK")
    failed_count = len(records) - ok_count

    print("\n" + "=" * 80)
    print("FINISHED")
    print(f"OK     : {ok_count}")
    print(f"FAILED : {failed_count}")
    print(f"Summary: {summary_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
