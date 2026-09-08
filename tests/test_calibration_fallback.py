"""Tests for ``prose.scripts.calibration_fallback``.

Builds a tiny multi-night obslog + FITS layout on-the-fly (mirroring the
pattern in ``tests/test_calibrate_muscat.py``) so tests are deterministic and
need no network or real observation data.
"""

from pathlib import Path

import pytest

from prose import utils
from prose.scripts.calibration_fallback import find_frames_in_other_nights

BAND_FROM = {"r": "rp", "g": "gp"}.get


def _write_night(
    root: Path,
    obslog_root: Path,
    instrument: str,
    night: str,
    frames: list[tuple[str, str, float, str]],
) -> None:
    """Create *frames* as ``(frame, object, exptime, filter)`` on ccd1's obslog.

    Also touches an empty ``.fits`` file per frame, since ``frames_from_obslog``
    only returns frames whose file exists on disk.
    """
    data_dir = root / instrument / night
    data_dir.mkdir(parents=True, exist_ok=True)
    obslog_dir = obslog_root / instrument / night
    obslog_dir.mkdir(parents=True, exist_ok=True)

    rows = ["FRAME,OBJECT,EXPTIME (s),FILTER"]
    for frame, obj, exptime, filt in frames:
        (data_dir / f"{frame}.fits").touch()
        rows.append(f"{frame},{obj},{exptime},{filt}")
    (obslog_dir / f"obslog-{instrument}-{night}-ccd1.csv").write_text(
        "\n".join(rows) + "\n"
    )


@pytest.fixture
def obslog_root(tmp_path, monkeypatch):
    root = tmp_path / "obslog"
    monkeypatch.setattr(utils, "OBSLOG_ROOT", str(root))
    return root


class TestFindFramesInOtherNights:
    def test_finds_nearest_night_with_matching_exposure(self, tmp_path, obslog_root):
        data_root = tmp_path / "data" / "muscat2"
        target = data_root / "260804"
        target.mkdir(parents=True)

        # Far night has a match; near night has a closer match -> near wins.
        _write_night(
            tmp_path / "data",
            obslog_root,
            "muscat2",
            "260616",
            [("F1", "DARK", 10.0, "r")],
        )
        _write_night(
            tmp_path / "data",
            obslog_root,
            "muscat2",
            "260818",
            [("F2", "DARK", 10.0, "r")],
        )

        paths, source = find_frames_in_other_nights(
            target, "rp", "DARK", BAND_FROM, exposure=10.0, max_days=90
        )
        assert source == "260818"
        assert paths == [str(data_root / "260818" / "F2.fits")]

    def test_respects_max_days(self, tmp_path, obslog_root):
        data_root = tmp_path / "data" / "muscat2"
        target = data_root / "260804"
        target.mkdir(parents=True)
        _write_night(
            tmp_path / "data",
            obslog_root,
            "muscat2",
            "260616",  # 49 days before
            [("F1", "DARK", 10.0, "r")],
        )

        paths, source = find_frames_in_other_nights(
            target, "rp", "DARK", BAND_FROM, exposure=10.0, max_days=10
        )
        assert paths == [] and source is None

    def test_exposure_mismatch_excluded(self, tmp_path, obslog_root):
        data_root = tmp_path / "data" / "muscat2"
        target = data_root / "260804"
        target.mkdir(parents=True)
        _write_night(
            tmp_path / "data",
            obslog_root,
            "muscat2",
            "260818",
            [("F1", "DARK", 4.7, "r")],
        )

        paths, source = find_frames_in_other_nights(
            target, "rp", "DARK", BAND_FROM, exposure=10.0, max_days=90
        )
        assert paths == [] and source is None

    def test_wrong_band_excluded(self, tmp_path, obslog_root):
        data_root = tmp_path / "data" / "muscat2"
        target = data_root / "260804"
        target.mkdir(parents=True)
        _write_night(
            tmp_path / "data",
            obslog_root,
            "muscat2",
            "260818",
            [("F1", "DARK", 10.0, "g")],  # gp, not rp
        )

        paths, source = find_frames_in_other_nights(
            target, "rp", "DARK", BAND_FROM, exposure=10.0, max_days=90
        )
        assert paths == [] and source is None

    def test_wrong_kind_excluded(self, tmp_path, obslog_root):
        data_root = tmp_path / "data" / "muscat2"
        target = data_root / "260804"
        target.mkdir(parents=True)
        _write_night(
            tmp_path / "data",
            obslog_root,
            "muscat2",
            "260818",
            [("F1", "FLAT", 10.0, "r")],
        )

        paths, source = find_frames_in_other_nights(
            target, "rp", "DARK", BAND_FROM, exposure=10.0, max_days=90
        )
        assert paths == [] and source is None

    def test_no_exposure_matches_any(self, tmp_path, obslog_root):
        """exposure=None (e.g. missing flats) accepts any exposure value."""
        data_root = tmp_path / "data" / "muscat2"
        target = data_root / "260804"
        target.mkdir(parents=True)
        _write_night(
            tmp_path / "data",
            obslog_root,
            "muscat2",
            "260818",
            [("F1", "FLAT", 3.2, "r")],
        )

        paths, source = find_frames_in_other_nights(
            target, "rp", "FLAT", BAND_FROM, exposure=None, max_days=90
        )
        assert source == "260818"
        assert len(paths) == 1

    def test_excludes_target_night_itself(self, tmp_path, obslog_root):
        data_root = tmp_path / "data" / "muscat2"
        _write_night(
            tmp_path / "data",
            obslog_root,
            "muscat2",
            "260804",
            [("F1", "DARK", 10.0, "r")],
        )
        target = data_root / "260804"

        paths, source = find_frames_in_other_nights(
            target, "rp", "DARK", BAND_FROM, exposure=10.0, max_days=90
        )
        assert paths == [] and source is None

    def test_night_without_obslog_is_skipped(self, tmp_path, obslog_root):
        data_root = tmp_path / "data" / "muscat2"
        target = data_root / "260804"
        target.mkdir(parents=True)
        # A night dir exists on disk but has no obslog entry at all.
        (data_root / "260818").mkdir(parents=True)

        paths, source = find_frames_in_other_nights(
            target, "rp", "DARK", BAND_FROM, exposure=10.0, max_days=90
        )
        assert paths == [] and source is None

    def test_no_candidate_nights_returns_empty(self, tmp_path, obslog_root):
        data_root = tmp_path / "data" / "muscat2"
        target = data_root / "260804"
        target.mkdir(parents=True)

        paths, source = find_frames_in_other_nights(
            target, "rp", "DARK", BAND_FROM, exposure=10.0, max_days=90
        )
        assert paths == [] and source is None
