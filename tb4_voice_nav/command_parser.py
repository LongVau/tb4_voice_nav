"""Vietnamese voice command parser and waypoint resolver for tb4_voice_nav."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Sequence
import unicodedata


@dataclass
class ParsedCommand:
    status: str  # 'OK', 'UNKNOWN_REGION', 'NO_INTENT', 'EMPTY'
    mode: str  # 'SINGLE', 'FROM_TO', 'MULTI', 'NONE'
    origin: dict | None = None
    waypoints: list[dict] = field(default_factory=list)
    unknown_names: list[str] = field(default_factory=list)
    raw_text: str = ''
    message: str = ''


def normalize_vi_text(text: str) -> str:
    """Normalize Vietnamese text to lowercase NFC with clean single spaces."""
    norm = unicodedata.normalize('NFC', text or '').lower()
    norm = re.sub(r'[.,!?;:"\'()\[\]{}\-_/\\]+', ' ', norm)
    norm = re.sub(r'\s+', ' ', norm).strip()
    return norm


def _split_clauses(norm_text: str) -> list[tuple[str, str]]:
    """Split normalized command into directional clauses: (connector, clause_body)."""
    pattern = re.compile(
        r'\b(di chuyển từ|đi từ|xuất phát từ|từ|rồi đi đến|rồi đến|rồi tới|sau đó đến|tiếp theo đến|đi qua|qua|ngang qua|đi đến|đi tới|di chuyển đến|di chuyển tới|đến|tới|sang|vào|về)\b'
    )
    matches = list(pattern.finditer(norm_text))
    if not matches:
        return [('', norm_text)]

    clauses: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        connector = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(norm_text)
        body = norm_text[start:end].strip()
        if body:
            clauses.append((connector, body))
    return clauses


def _match_region_in_clause(clause_body: str, regions: Sequence[dict]) -> dict | None:
    """Match a clause snippet against known region aliases (longest multi-word alias first)."""
    candidates: list[tuple[int, int, dict]] = []
    for region in regions:
        for alias in region.get('aliases', []):
            alias_norm = normalize_vi_text(alias)
            if not alias_norm:
                continue
            # Single-character aliases ('a', 'b', 'c', 'd') only match if clause_body is
            # that exact token or preceded by region prefix word
            if len(alias_norm) == 1:
                m = re.search(
                    rf'(?:^|\b(?:khu vực|khu|phòng|vùng|điểm|hành lang)\s+)({re.escape(alias_norm)})\b',
                    clause_body,
                )
            else:
                m = re.search(rf'\b({re.escape(alias_norm)})\b', clause_body)
            if m:
                candidates.append((m.start(1), -len(alias_norm), region))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def parse_voice_command(text: str, regions: Sequence[dict]) -> ParsedCommand:
    """Parse a Vietnamese voice command into structured navigation waypoints."""
    norm = normalize_vi_text(text)
    if not norm:
        return ParsedCommand(
            status='EMPTY',
            mode='NONE',
            raw_text=text,
            message='Câu lệnh trống.',
        )

    nav_keywords = (
        'đi',
        'di chuyển',
        'đến',
        'tới',
        'từ',
        'qua',
        'sang',
        'về',
        'khu vực',
        'phòng',
        'hành lang',
        'văn phòng',
        'vùng',
        'điểm',
    )
    if not any(re.search(rf'\b{re.escape(kw)}\b', norm) for kw in nav_keywords):
        return ParsedCommand(
            status='NO_INTENT',
            mode='NONE',
            raw_text=text,
            message=f"Không nhận diện được ý định điều hướng trong câu: '{text}'",
        )

    clauses = _split_clauses(norm)
    matched_waypoints: list[dict] = []
    unknown_names: list[str] = []
    has_from_origin = False

    for idx, (connector, body) in enumerate(clauses):
        region = _match_region_in_clause(body, regions)
        if region is not None:
            if idx == 0 and 'từ' in connector:
                has_from_origin = True
            if not matched_waypoints or matched_waypoints[-1]['code'] != region['code']:
                matched_waypoints.append(region)
        else:
            # Clause had a directional or place keyword, so this clause names an unknown place
            cleaned_unknown = re.sub(
                r'\b(rồi|sau đó|tiếp theo|nhé|nha|đi|di chuyển|hãy|cho tôi|giúp tôi)\b',
                '',
                body,
            ).strip()
            if cleaned_unknown:
                unknown_names.append(cleaned_unknown)

    if unknown_names or not matched_waypoints:
        unk_str = ', '.join(unknown_names) if unknown_names else norm
        valid_str = ', '.join(f"{r['code']} ({r['semantic_name']})" for r in regions)
        return ParsedCommand(
            status='UNKNOWN_REGION',
            mode='NONE',
            unknown_names=unknown_names or [norm],
            raw_text=text,
            message=(
                f"Khu vực '{unk_str}' không tồn tại trên bản đồ! "
                f"Các khu vực hợp lệ: {valid_str}."
            ),
        )

    if len(matched_waypoints) == 1:
        target = matched_waypoints[0]
        return ParsedCommand(
            status='OK',
            mode='SINGLE',
            origin=None,
            waypoints=[target],
            raw_text=text,
            message=f"Điều hướng đến {target['name']} ({target['semantic_name']}).",
        )

    if len(matched_waypoints) == 2 and has_from_origin:
        origin_reg, dest_reg = matched_waypoints[0], matched_waypoints[1]
        return ParsedCommand(
            status='OK',
            mode='FROM_TO',
            origin=origin_reg,
            waypoints=[origin_reg, dest_reg],
            raw_text=text,
            message=(
                f"Di chuyển từ {origin_reg['name']} ({origin_reg['semantic_name']}) "
                f"đến {dest_reg['name']} ({dest_reg['semantic_name']})."
            ),
        )

    origin_reg = matched_waypoints[0] if has_from_origin else None
    route_desc = ' -> '.join(f"{w['name']} ({w['semantic_name']})" for w in matched_waypoints)
    return ParsedCommand(
        status='OK',
        mode='MULTI',
        origin=origin_reg,
        waypoints=matched_waypoints,
        raw_text=text,
        message=f'Điều hướng đa điểm: {route_desc}.',
    )


def resolve_navigation_waypoints(
    parsed: ParsedCommand,
    robot_xy: tuple[float, float] | None,
    at_region_radius: float = 1.5,
) -> list[dict]:
    """Resolve the active waypoint sequence given the robot's current map position.

    If the command specifies an origin ("từ A đến B") and the robot is already within
    `at_region_radius` meters of A's goal pose, skip A and navigate straight to B.
    Otherwise navigate to A first, then B (and any subsequent waypoints).
    """
    if parsed.status != 'OK' or not parsed.waypoints:
        return []

    waypoints = list(parsed.waypoints)
    if parsed.origin is not None and len(waypoints) >= 2 and robot_xy is not None:
        ox = float(parsed.origin['goal_pose']['x'])
        oy = float(parsed.origin['goal_pose']['y'])
        dist = math.hypot(float(robot_xy[0]) - ox, float(robot_xy[1]) - oy)
        if dist <= float(at_region_radius):
            return waypoints[1:]
    return waypoints

