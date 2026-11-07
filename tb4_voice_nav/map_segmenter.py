"""OpenCV Occupancy Grid map segmentation and semantic labeling for tb4_voice_nav."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml

REGION_METADATA = [
    {
        'id': 1,
        'code': 'A',
        'name': 'Khu vực A',
        'semantic_name': 'Phòng học',
        'aliases': [
            'khu vực a',
            'phòng a',
            'vùng a',
            'điểm a',
            'phòng học',
            'khu a',
            'a',
        ],
        'color_rgb': [66, 135, 245],
    },
    {
        'id': 2,
        'code': 'B',
        'name': 'Khu vực B',
        'semantic_name': 'Phòng thí nghiệm',
        'aliases': [
            'khu vực b',
            'phòng b',
            'vùng b',
            'điểm b',
            'phòng thí nghiệm',
            'phòng lab',
            'khu b',
            'b',
        ],
        'color_rgb': [52, 199, 89],
    },
    {
        'id': 3,
        'code': 'C',
        'name': 'Khu vực C',
        'semantic_name': 'Hành lang',
        'aliases': [
            'khu vực c',
            'phòng c',
            'vùng c',
            'điểm c',
            'hành lang c',
            'hành lang',
            'khu c',
            'c',
        ],
        'color_rgb': [255, 159, 10],
    },
    {
        'id': 4,
        'code': 'D',
        'name': 'Khu vực D',
        'semantic_name': 'Văn phòng',
        'aliases': [
            'khu vực d',
            'phòng d',
            'vùng d',
            'điểm d',
            'văn phòng',
            'phòng làm việc',
            'khu d',
            'd',
        ],
        'color_rgb': [191, 90, 242],
    },
]


def pixel_to_world(
    px: int,
    py: int,
    height: int,
    resolution: float,
    origin: Sequence[float],
) -> tuple[float, float]:
    """Convert image pixel (px, py) to ROS map frame coordinates (wx, wy)."""
    ox, oy = float(origin[0]), float(origin[1])
    wx = ox + (float(px) + 0.5) * float(resolution)
    wy = oy + (float(height - py) - 0.5) * float(resolution)
    return round(wx, 4), round(wy, 4)


def world_to_pixel(
    wx: float,
    wy: float,
    height: int,
    resolution: float,
    origin: Sequence[float],
) -> tuple[int, int]:
    """Convert ROS map frame coordinates (wx, wy) to image pixel (px, py)."""
    ox, oy = float(origin[0]), float(origin[1])
    px = int(round((float(wx) - ox) / float(resolution) - 0.5))
    py = int(round(float(height) - ((float(wy) - oy) / float(resolution) + 0.5)))
    return px, py


def segment_occupancy_map(
    pgm_path: str | Path,
    yaml_path: str | Path,
    output_yaml_path: str | Path | None = None,
    output_image_path: str | Path | None = None,
) -> dict:
    """Segment 2D Occupancy Grid map (.pgm + .yaml) into labeled regions using Watershed."""
    pgm_file = Path(pgm_path)
    yaml_file = Path(yaml_path)

    with yaml_file.open('r', encoding='utf-8') as f:
        map_meta = yaml.safe_load(f)

    resolution = float(map_meta['resolution'])
    origin = [float(v) for v in map_meta['origin']]

    img = cv2.imread(str(pgm_file), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Cannot read occupancy map image: {pgm_file}')

    height, width = img.shape
    free = (img >= 250).astype(np.uint8) * 255

    # Erode by robot inflation radius (0.30m = 6px) and keep the component connected to (0, 0)
    spawn_px, spawn_py = world_to_pixel(0.0, 0.0, height, resolution, origin)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    eroded = cv2.erode(free, erode_kernel)
    _, cc_labels = cv2.connectedComponents(eroded)
    robot_cc = int(cc_labels[spawn_py, spawn_px])

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    reachable_free = (
        cv2.dilate((cc_labels == robot_cc).astype(np.uint8) * 255, dilate_kernel) & free
    )

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(reachable_free, cv2.MORPH_OPEN, open_kernel)
    clean[0, :] = 0
    clean[-1, :] = 0
    clean[:, 0] = 0
    clean[:, -1] = 0

    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.45 * float(dist.max()), 255, 0)
    sure_fg = np.uint8(sure_fg)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sure_fg)
    candidate_seeds = [
        i for i in range(1, num_labels) if int(stats[i, cv2.CC_STAT_AREA]) >= 300
    ]
    candidate_seeds.sort(key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    selected_seeds = candidate_seeds[: len(REGION_METADATA)]

    # Order seeds consistently so Region A, B, C, D match fixed semantic regions
    seed_centers = []
    for seed_label in selected_seeds:
        seed_mask = (labels == seed_label) & (cc_labels == robot_cc)
        masked_dist = np.where(seed_mask, dist, 0)
        py, px = np.unravel_index(int(np.argmax(masked_dist)), masked_dist.shape)
        seed_centers.append((seed_label, int(px), int(py)))

    # Desired ordering:
    # A: central-upper-left (-3.12, 3.03) -> py ~ 139, px ~ 139
    # B: lower-left (-6.02, -3.82) -> py ~ 276, px ~ 81
    # C: upper-left corridor (-7.57, 7.53) -> py ~ 49, px ~ 50
    # D: right wing (6.22, -2.07) -> py ~ 241, px ~ 326
    target_pixels = [(139, 139), (81, 276), (50, 49), (326, 241)]
    ordered_seeds: list[int] = []
    remaining = list(seed_centers)
    for tx, ty in target_pixels:
        if not remaining:
            break
        best = min(remaining, key=lambda item: (item[1] - tx) ** 2 + (item[2] - ty) ** 2)
        ordered_seeds.append(best[0])
        remaining.remove(best)

    markers = np.zeros_like(img, dtype=np.int32)
    markers[clean == 0] = 1
    for marker_idx, seed_label in enumerate(ordered_seeds, start=2):
        markers[labels == seed_label] = marker_idx

    vis_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.watershed(vis_bgr.copy(), markers)

    overlay = vis_bgr.copy()
    regions_out: list[dict] = []

    for offset, meta in enumerate(REGION_METADATA[: len(ordered_seeds)]):
        marker_id = offset + 2
        region_mask = (markers == marker_id) & (clean > 0)
        safe_mask = region_mask & (cc_labels == robot_cc)
        if not np.any(safe_mask):
            safe_mask = region_mask

        masked_dist = np.where(safe_mask, dist, 0)
        py, px = np.unravel_index(int(np.argmax(masked_dist)), masked_dist.shape)
        wx, wy = pixel_to_world(int(px), int(py), height, resolution, origin)
        clearance_m = round(float(masked_dist[py, px]) * resolution, 3)
        area_px = int(np.sum(region_mask))

        r, g, b = meta['color_rgb']
        overlay[region_mask] = (b, g, r)

        regions_out.append(
            {
                'id': int(meta['id']),
                'code': str(meta['code']),
                'name': str(meta['name']),
                'semantic_name': str(meta['semantic_name']),
                'aliases': list(meta['aliases']),
                'pixel_center': [int(px), int(py)],
                'goal_pose': {
                    'x': float(wx),
                    'y': float(wy),
                    'yaw': 0.0,
                },
                'clearance_m': clearance_m,
                'area_px': area_px,
                'color_rgb': list(meta['color_rgb']),
            }
        )

    blended = cv2.addWeighted(vis_bgr, 0.45, overlay, 0.55, 0)
    blended[markers == -1] = (0, 0, 255)

    for reg in regions_out:
        px, py = reg['pixel_center']
        r, g, b = reg['color_rgb']
        cv2.circle(blended, (px, py), 6, (0, 0, 0), -1)
        cv2.circle(blended, (px, py), 4, (b, g, r), -1)
        label_str = f"{reg['code']}:{reg['semantic_name']} ({reg['goal_pose']['x']:.1f},{reg['goal_pose']['y']:.1f})"
        tx = max(5, min(width - 170, px - 45))
        ty = max(15, min(height - 10, py - 10))
        cv2.putText(
            blended,
            label_str,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            blended,
            label_str,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    payload = {
        'map': {
            'image': pgm_file.name,
            'width': int(width),
            'height': int(height),
            'resolution': resolution,
            'origin': origin,
        },
        'regions_count': len(regions_out),
        'regions': regions_out,
    }

    if output_yaml_path is not None:
        out_yaml = Path(output_yaml_path)
        out_yaml.parent.mkdir(parents=True, exist_ok=True)
        with out_yaml.open('w', encoding='utf-8') as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    if output_image_path is not None:
        out_img = Path(output_image_path)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), blended)

    return payload


def load_regions(regions_yaml_path: str | Path) -> list[dict]:
    """Load the segmented regions list from regions.yaml."""
    path = Path(regions_yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f'Regions YAML not found: {path}')
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get('regions'), list):
        raise ValueError(f'Invalid regions YAML structure in {path}')
    return data['regions']


def _default_data_root() -> Path:
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / 'maps' / 'map.yaml').is_file():
        return candidate
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory('tb4_voice_nav'))
    except Exception:
        return candidate


def main(args: Sequence[str] | None = None) -> None:
    pkg_root = _default_data_root()
    parser = argparse.ArgumentParser(description='Segment 2D Occupancy Grid map into regions.')
    parser.add_argument('--pgm', default=str(pkg_root / 'maps' / 'map.pgm'))
    parser.add_argument('--yaml', default=str(pkg_root / 'maps' / 'map.yaml'))
    parser.add_argument('--out-yaml', default=str(pkg_root / 'config' / 'regions.yaml'))
    parser.add_argument('--out-img', default=str(pkg_root / 'maps' / 'map_segmented.png'))
    parsed = parser.parse_args(args=args)

    res = segment_occupancy_map(
        pgm_path=parsed.pgm,
        yaml_path=parsed.yaml,
        output_yaml_path=parsed.out_yaml,
        output_image_path=parsed.out_img,
    )
    print(f"Segmented {res['regions_count']} regions -> {parsed.out_yaml} & {parsed.out_img}")
    for reg in res['regions']:
        gp = reg['goal_pose']
        print(
            f"  [{reg['code']}] {reg['name']} ({reg['semantic_name']}): "
            f"px={reg['pixel_center']} -> world=({gp['x']:.2f}, {gp['y']:.2f}) "
            f"clearance={reg['clearance_m']:.2f}m"
        )


if __name__ == '__main__':
    main()
