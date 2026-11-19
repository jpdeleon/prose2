"""run_photometry --min_axis_ratio: parsing and wiring to every detection."""

import ast
import inspect
from pathlib import Path

import pytest

from prose.blocks import PointSourceDetection
from prose.scripts import run_photometry as rp

BASE_ARGS = ["--target_name", "T1", "--data_dir", ".", "--results_dir", "."]
SOURCE = Path(inspect.getsourcefile(rp)).read_text()


def _detection_blocks(sequence):
    return [b for b in sequence.blocks if isinstance(b, PointSourceDetection)]


def test_default_is_off():
    assert rp.parse_args(BASE_ARGS).min_axis_ratio == 0.0


@pytest.mark.parametrize("flag", ["--min_axis_ratio", "--min-axis-ratio"])
def test_accepts_value_in_unit_interval(flag):
    assert rp.parse_args(BASE_ARGS + [flag, "0.2"]).min_axis_ratio == 0.2


@pytest.mark.parametrize("value", ["-0.1", "1.5", "nan", "abc"])
def test_rejects_value_outside_unit_interval(value):
    with pytest.raises(SystemExit):
        rp.parse_args(BASE_ARGS + ["--min_axis_ratio", value])


def test_reference_sequence_forwards_ratio():
    (block,) = _detection_blocks(rp.reference_sequence(min_axis_ratio=0.2))
    assert block.min_axis_ratio == 0.2


def test_frame_alignment_sequence_forwards_ratio():
    sequence = rp._frame_alignment_sequence(
        ccd_trim_size_yx=(0, 0),
        max_num_stars=20,
        min_star_separation=10,
        min_area=20,
        bad_pixel_map=None,
        min_axis_ratio=0.2,
    )
    (block,) = _detection_blocks(sequence)
    assert block.min_axis_ratio == 0.2


def test_every_function_taking_min_area_takes_min_axis_ratio():
    # Detection settings travel together: a helper that forwards min_area
    # must also forward min_axis_ratio, or the flag silently stops applying.
    tree = ast.parse(SOURCE)
    missing = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and "min_area" in {a.arg for a in node.args.args + node.args.kwonlyargs}
        and "min_axis_ratio"
        not in {a.arg for a in node.args.args + node.args.kwonlyargs}
    ]
    assert missing == []


def test_every_call_passing_min_area_passes_min_axis_ratio():
    tree = ast.parse(SOURCE)
    missing = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and "min_area" in {kw.arg for kw in node.keywords}
        and "min_axis_ratio" not in {kw.arg for kw in node.keywords}
    ]
    assert missing == []
