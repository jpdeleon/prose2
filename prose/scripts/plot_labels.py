"""Overlap-free placement of point labels on a matplotlib axis.

Used for the SIMBAD markers on the ``_ref`` and ``_cutouts`` plots, where many
catalog entries share (or nearly share) a position and fixed-offset labels pile
on top of each other. Placement is deterministic and greedy:

1. Points closer than ``merge_pt`` on screen are merged into one marker whose
   label lists every distinct name once (``"EB*/X/IR"``).
2. Each label tries candidate offsets on rings of growing radius around its
   point and keeps the first one that overlaps neither an earlier label, a
   marker, an existing text on the axis, nor the axis edge. When none is free,
   the candidate with the least overlap wins.

Positions are evaluated after a layout pass, and placed labels are excluded
from later layout passes, so constrained layout cannot move them back into
each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.transforms import Bbox

MERGE_PT = 4.0  # points closer than this on screen share one label [pt]
# Label offsets tried, inner first [pt]. The inner ring clears the marker's own
# box (sqrt(40)/2 + pad ~ 4.2 pt) even on the diagonals.
RING_RADII_PT = (6.0, 11.0, 17.0, 25.0)
# Directions tried on each ring: NE first (the historical +5,+5 placement).
RING_ANGLES_DEG = (45, 315, 135, 225, 0, 180, 90, 270)
MARKER_PAD_PT = 1.0  # extra clearance kept around each marker [pt]


@dataclass(frozen=True)
class LabelPoint:
    x: float
    y: float
    label: str
    color: str


@dataclass(frozen=True)
class MergedPoint:
    x: float
    y: float
    labels: tuple[str, ...]
    colors: tuple[str, ...]

    @property
    def text(self) -> str:
        return "/".join(self.labels)


def merge_colocated(points, to_display, merge_pt_px: float) -> list[MergedPoint]:
    """Group points whose display positions are within ``merge_pt_px``.

    Grouping is single-linkage in input order, so the result is deterministic.
    Each group keeps the first point's position and the distinct labels and
    colours in first-seen order.
    """
    points = list(points)
    if not points:
        return []
    xy = to_display(np.array([[p.x, p.y] for p in points], dtype=float))
    group_of = list(range(len(points)))

    def root(i):
        while group_of[i] != i:
            group_of[i] = group_of[group_of[i]]
            i = group_of[i]
        return i

    for i in range(len(points)):
        close = np.hypot(*(xy[i + 1 :] - xy[i]).T) < merge_pt_px
        for j in np.nonzero(close)[0] + i + 1:
            ri, rj = root(i), root(int(j))
            if ri != rj:
                group_of[max(ri, rj)] = min(ri, rj)

    members: dict[int, list[int]] = {}
    for i in range(len(points)):
        members.setdefault(root(i), []).append(i)
    merged = []
    for first, idx in members.items():
        labels = tuple(dict.fromkeys(str(points[i].label) for i in idx))
        colors = tuple(dict.fromkeys(points[i].color for i in idx))
        merged.append(MergedPoint(points[first].x, points[first].y, labels, colors))
    return merged


def candidate_offsets():
    """``(dx_pt, dy_pt, ha, va)`` candidates, inner ring first."""
    out = []
    for radius in RING_RADII_PT:
        for angle in RING_ANGLES_DEG:
            dx = radius * np.cos(np.radians(angle))
            dy = radius * np.sin(np.radians(angle))
            ha = "left" if dx > 0.5 else "right" if dx < -0.5 else "center"
            va = "bottom" if dy > 0.5 else "top" if dy < -0.5 else "center"
            out.append((float(dx), float(dy), ha, va))
    return out


def _overlap_area(bbox: Bbox, obstacles, frame: Bbox) -> float:
    area = 0.0
    for other in obstacles:
        inter = Bbox.intersection(bbox, other)
        if inter is not None:
            area += inter.width * inter.height
    # Any part of the label outside the axis counts as overlap.
    inside = Bbox.intersection(bbox, frame)
    inside_area = inside.width * inside.height if inside is not None else 0.0
    return area + (bbox.width * bbox.height - inside_area)


def place_labels(ax, points, fontsize=5, marker_size=40, zorder=10, color_for=None):
    """Draw a hollow-diamond marker and a non-overlapping label per point.

    *points* is an iterable of :class:`LabelPoint` in data coordinates.
    *color_for* maps a merged point's colours to the one used for its marker
    and label (default: the first). Returns the annotations, in placement
    order.
    """
    fig = ax.figure
    fig.draw_without_rendering()  # settle layout so display coords are final
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    merged = merge_colocated(points, ax.transData.transform, MERGE_PT * px_per_pt)
    if not merged:
        return []
    color_for = color_for or (lambda colors: colors[0])

    frame = ax.get_window_extent(renderer)
    half = (np.sqrt(marker_size) / 2 + MARKER_PAD_PT) * px_per_pt
    obstacles = [t.get_window_extent(renderer) for t in ax.texts if t.get_visible()]
    for p in merged:
        cx, cy = ax.transData.transform((p.x, p.y))
        obstacles.append(Bbox.from_extents(cx - half, cy - half, cx + half, cy + half))

    annotations = []
    for p in merged:
        color = color_for(p.colors)
        ax.scatter(
            [p.x],
            [p.y],
            marker="D",
            s=marker_size,
            ec=color,
            fc="none",
            lw=1,
            zorder=zorder,
        )
        ann = ax.annotate(
            p.text,
            (p.x, p.y),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=fontsize,
            color=color,
            zorder=zorder,
        )
        ann.set_in_layout(False)
        best = None
        for dx, dy, ha, va in candidate_offsets():
            ann.xyann = (dx, dy)
            ann.set_horizontalalignment(ha)
            ann.set_verticalalignment(va)
            bbox = ann.get_window_extent(renderer)
            cost = _overlap_area(bbox, obstacles, frame)
            if best is None or cost < best[0]:
                best = (cost, dx, dy, ha, va, bbox)
            if cost == 0:
                break
        _, dx, dy, ha, va, bbox = best
        ann.xyann = (dx, dy)
        ann.set_horizontalalignment(ha)
        ann.set_verticalalignment(va)
        obstacles.append(bbox)
        annotations.append(ann)
    return annotations
