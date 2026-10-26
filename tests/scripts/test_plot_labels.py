"""Tests for overlap-free SIMBAD label placement (``prose.scripts.plot_labels``)."""

import itertools

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.transforms import Bbox

from prose.scripts import plot_labels as pl
from prose.scripts import run_photometry as rp


def _identity(xy):
    return np.asarray(xy, dtype=float)


def _overlapping_pairs(annotations, renderer):
    boxes = [a.get_window_extent(renderer) for a in annotations]
    return [
        (annotations[i].get_text(), annotations[j].get_text())
        for i, j in itertools.combinations(range(len(boxes)), 2)
        if Bbox.intersection(boxes[i], boxes[j]) is not None
    ]


def _crowded_points():
    """Ten distinct objects 4 px apart (just beyond the merge distance on a
    4-inch axis showing 100 px), plus duplicate entries on the first one."""
    pts = [
        pl.LabelPoint(40.0 + 4 * (k % 4), 44.0 + 4 * (k // 4), f"Obj{k}", "teal")
        for k in range(10)
    ]
    # Two catalog entries on top of the first object, one repeating its type.
    pts.append(pl.LabelPoint(pts[0].x, pts[0].y, "X", "teal"))
    pts.append(pl.LabelPoint(pts[0].x + 0.01, pts[0].y, "Obj0", "teal"))
    return pts


def test_merge_colocated_groups_close_points_and_dedupes_labels():
    pts = [
        pl.LabelPoint(0, 0, "EB*", "orange"),
        pl.LabelPoint(1, 1, "X", "teal"),
        pl.LabelPoint(2, 0, "EB*", "orange"),  # chained to the first via (1, 1)
        pl.LabelPoint(50, 50, "IR", "teal"),
    ]

    merged = pl.merge_colocated(pts, _identity, merge_pt_px=2.0)

    assert [m.text for m in merged] == ["EB*/X", "IR"]
    assert merged[0].colors == ("orange", "teal")
    assert (merged[0].x, merged[0].y) == (0, 0)


def test_merge_colocated_empty():
    assert pl.merge_colocated([], _identity, 1.0) == []


def test_candidate_offsets_start_north_east_and_clear_the_marker():
    offsets = pl.candidate_offsets()
    dx, dy, ha, va = offsets[0]
    assert dx > 0 and dy > 0 and (ha, va) == ("left", "bottom")
    marker_half_pt = np.sqrt(40) / 2 + pl.MARKER_PAD_PT
    assert all(max(abs(dx), abs(dy)) > marker_half_pt for dx, dy, *_ in offsets)
    assert len(offsets) == len(set(offsets))


def test_fixed_offset_labels_overlap_in_crowded_field():
    """Guards the fixture: the old fixed (+5, +5) labels do collide on it."""
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.imshow(np.zeros((100, 100)), origin="lower")
    fig.draw_without_rendering()
    anns = [
        ax.annotate(
            p.label, (p.x, p.y), xytext=(5, 5), textcoords="offset points", fontsize=5
        )
        for p in _crowded_points()
    ]
    assert _overlapping_pairs(anns, fig.canvas.get_renderer())
    plt.close(fig)


def test_place_labels_avoids_overlap_in_crowded_field_after_saving(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.imshow(np.zeros((100, 100)), origin="lower")

    annotations = pl.place_labels(ax, _crowded_points())

    # The duplicates collapse onto the first object's single label.
    assert len(annotations) == 10
    assert "Obj0/X" in [a.get_text() for a in annotations]
    # Same check _savefig's output would face: dpi change + tight bbox.
    fig.savefig(tmp_path / "labels.png", dpi=120, bbox_inches="tight")
    fig.draw_without_rendering()
    assert _overlapping_pairs(annotations, fig.canvas.get_renderer()) == []
    plt.close(fig)


def test_place_labels_avoids_existing_axis_text():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    existing = ax.text(52, 52, "STAR 12", fontsize=8)

    (ann,) = pl.place_labels(ax, [pl.LabelPoint(50, 50, "EB*", "teal")])

    renderer = fig.canvas.get_renderer()
    assert (
        Bbox.intersection(
            ann.get_window_extent(renderer), existing.get_window_extent(renderer)
        )
        is None
    )
    plt.close(fig)


def test_draw_simbad_objects_merged_marker_takes_flagged_colour():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    colors = rp.get_plot_colors()

    rp._draw_simbad_objects(
        ax, [(50, 50, "X"), (50, 50, "EclBin"), (10, 90, "IR")], colors, True
    )

    by_text = {t.get_text(): t.get_color() for t in ax.texts}
    assert by_text == {
        "X/EclBin": colors["simbad_eclbin"],
        "IR": colors["simbad_default"],
    }
    plt.close(fig)


@pytest.mark.parametrize(
    "flag_types, expected_key", [(True, "simbad_eclbin"), (False, "simbad_default")]
)
def test_simbad_color_flags_only_when_enabled(flag_types, expected_key):
    colors = rp.get_plot_colors()
    assert rp._simbad_color("EclBin", colors, flag_types) == colors[expected_key]
