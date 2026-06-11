"""Settings/CLI tests for the ``prose.scripts.run_photometry`` script.

These exercise ``parse_args`` across the different run configurations the
script supports (bands, references, aperture geometry, GIF, test-run, edge
trim, ...) plus the ``parse_trim`` helper. They are kept separate from the
prose library's own test-suite (``tests/test_*.py``) under ``tests/scripts/``.

``argparse`` raises ``SystemExit`` for both missing required arguments and
``parser.error(...)`` validation failures, so invalid-configuration tests
assert on ``SystemExit``.
"""

from pathlib import Path

import numpy as np
import pytest

from prose.scripts import run_photometry as rp

REQUIRED = [
    "--target_name",
    "TOI-6715",
    "--data_dir",
    "/data/x",
    "--results_dir",
    "/tmp/o",
]


def _argv(*extra):
    """Required args plus any extra tokens, as an argv list for parse_args."""
    return [*REQUIRED, *extra]


# --------------------------- defaults ---------------------------


def test_parse_args_defaults_match_documented_behaviour():
    args = rp.parse_args(_argv())
    assert args.target_name == "TOI-6715"
    assert isinstance(args.data_dir, Path) and isinstance(args.results_dir, Path)
    assert args.bands == rp.DEFAULT_BANDS
    assert args.ref_band is None and args.refid is None  # per-band self-reference
    assert args.make_gif is False  # GIF is opt-in
    assert args.gif_stride == rp.DEFAULT_GIF_STRIDE
    assert args.aper_radii is None and args.annulus is None
    assert args.aper_unit == "pix"
    assert args.glob == "*.fits"
    assert args.test_run is False
    assert args.test_run_frames == rp.TEST_RUN_FRAMES
    assert args.max_num_stars == rp.MAX_NUM_STARS
    assert args.min_star_separation == rp.MIN_STAR_SEPARATION
    assert args.n_stars_align is None
    assert args.cutout_size == rp.CUTOUT_SIZE
    assert args.ccd_trim_size_yx == rp.CCD_TRIM_SIZE_YX
    assert args.use_barycorrpy is False
    assert args.verbose is False
    assert args.plot_gaia_sources is False
    assert args.overwrite is False
    assert args.bin_size_minutes == pytest.approx(rp.BIN_SIZE_DAYS * 24 * 60)


@pytest.mark.parametrize("arg", ["--target_name", "--data_dir", "--results_dir"])
def test_parse_args_required_arguments(arg):
    """Dropping any required argument aborts with SystemExit."""
    argv = _argv()
    i = argv.index(arg)
    del argv[i : i + 2]
    with pytest.raises(SystemExit):
        rp.parse_args(argv)


# --------------------------- GIF (opt-in) ---------------------------


def test_parse_args_gif_flag_enables_gif():
    assert rp.parse_args(_argv("--gif")).make_gif is True


def test_parse_args_legacy_no_gif_flag_is_rejected():
    """--no_gif was removed when GIF became opt-in; it must no longer parse."""
    with pytest.raises(SystemExit):
        rp.parse_args(_argv("--no_gif"))


def test_parse_args_custom_gif_stride():
    assert rp.parse_args(_argv("--gif", "--gif_stride", "25")).gif_stride == 25


# --------------------------- bands / reference ---------------------------


def test_parse_args_custom_bands_list():
    args = rp.parse_args(_argv("--bands", "g_narrow", "Na_D", "i_narrow"))
    assert args.bands == ["g_narrow", "Na_D", "i_narrow"]


def test_parse_args_ref_band_and_refid():
    args = rp.parse_args(_argv("--ref_band", "gp", "--refid", "3"))
    assert args.ref_band == "gp"
    assert args.refid == 3


# --------------------------- test-run / detection knobs ---------------------------


def test_parse_args_test_run_with_custom_frame_count():
    args = rp.parse_args(_argv("--test_run", "--test_run_frames", "5"))
    assert args.test_run is True
    assert args.test_run_frames == 5


def test_parse_args_detection_and_alignment_knobs():
    args = rp.parse_args(
        _argv(
            "--max_num_stars",
            "15",
            "--min_star_separation",
            "7.5",
            "--n_stars_align",
            "6",
            "--cutout_size",
            "41",
        )
    )
    assert args.max_num_stars == 15
    assert args.min_star_separation == pytest.approx(7.5)
    assert args.n_stars_align == 6
    assert args.cutout_size == 41


def test_parse_args_bin_size_minutes_override():
    assert rp.parse_args(_argv("--bin_size_minutes", "20")).bin_size_minutes == 20.0


@pytest.mark.parametrize(
    "flag, attr",
    [
        ("--use_barycorrpy", "use_barycorrpy"),
        ("--verbose", "verbose"),
        ("--plot_gaia_sources", "plot_gaia_sources"),
        ("--overwrite", "overwrite"),
    ],
)
def test_parse_args_boolean_flags(flag, attr):
    assert getattr(rp.parse_args(_argv(flag)), attr) is True


# --------------------------- aperture geometry ---------------------------


def test_parse_args_custom_aperture_grid_with_annulus():
    args = rp.parse_args(
        _argv("--aper_radii", "10,20,2", "--annulus", "24,30", "--aper_unit", "fwhm")
    )
    np.testing.assert_allclose(args.aper_radii, [10, 12, 14, 16, 18, 20])
    assert args.annulus == (24.0, 30.0)
    assert args.aper_unit == "fwhm"


def test_parse_args_aper_radii_requires_annulus():
    with pytest.raises(SystemExit):
        rp.parse_args(_argv("--aper_radii", "10,20,2"))


def test_parse_args_annulus_without_aper_radii_is_rejected():
    with pytest.raises(SystemExit):
        rp.parse_args(_argv("--annulus", "24,30"))


def test_parse_args_aper_unit_without_aper_radii_is_rejected():
    with pytest.raises(SystemExit):
        rp.parse_args(_argv("--aper_unit", "fwhm"))


def test_parse_args_max_aperture_must_not_exceed_inner_annulus():
    # max radius 30 > inner annulus radius 24 -> invalid
    with pytest.raises(SystemExit):
        rp.parse_args(_argv("--aper_radii", "10,30,2", "--annulus", "24,40"))


def test_parse_args_max_aperture_equal_to_inner_annulus_is_allowed():
    args = rp.parse_args(_argv("--aper_radii", "10,24,2", "--annulus", "24,40"))
    assert args.aper_radii.max() == pytest.approx(24.0)
    assert args.annulus == (24.0, 40.0)


# --------------------------- ccd trim parsing ---------------------------


def test_parse_args_ccd_trim_override():
    assert rp.parse_args(_argv("--ccd_trim", "5,8")).ccd_trim_size_yx == (5, 8)


def test_parse_trim_valid():
    assert rp.parse_trim("5,8") == (5, 8)


@pytest.mark.parametrize("bad", ["5", "5,8,9", "a,b", "-1,2", "2,-1"])
def test_parse_trim_rejects_bad_input(bad):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        rp.parse_trim(bad)
