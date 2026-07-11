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
    assert args.ref_select == "position"  # opt-in; unchanged legacy behavior by default
    assert args.ref_select_top_k == rp.REF_SELECT_DEFAULT_TOP_K
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
    assert args.annulus_pix is None  # sky annulus uses default
    assert args.cutout_size == rp.CUTOUT_SIZE
    assert args.ccd_trim_size_yx == rp.CCD_TRIM_SIZE_YX
    assert args.use_barycorrpy is False
    assert args.verbose is False
    assert args.plot_gaia_sources is False
    assert args.overwrite is False
    assert args.bin_size_minutes == pytest.approx(rp.BIN_SIZE_DAYS * 24 * 60)
    assert args.site is None
    assert args.mode is None


def test_parse_args_custom_site():
    assert rp.parse_args(_argv("--site", "lsc")).site == "lsc"


def test_parse_args_custom_mode():
    assert rp.parse_args(_argv("--mode", "central_2k_2x2")).mode == "central_2k_2x2"
    assert rp.parse_args(_argv("--mode", "full_frame")).mode == "full_frame"
    with pytest.raises(SystemExit):
        rp.parse_args(_argv("--mode", "abc"))


def test_parse_args_choices_depend_on_sinistro_data(tmp_path):
    from tests.scripts.test_run_photometry import _write_sinistro_fits

    _write_sinistro_fits(tmp_path, "test1.fits", "lsc", confmode="central_2k_2x2")

    argv = [
        "--target_name",
        "TOI-6715",
        "--data_dir",
        str(tmp_path),
        "--results_dir",
        str(tmp_path / "results"),
        "--mode",
        "central_2k_2x2",
    ]
    args = rp.parse_args(argv)
    assert args.mode == "central_2k_2x2"

    argv_invalid = [
        "--target_name",
        "TOI-6715",
        "--data_dir",
        str(tmp_path),
        "--results_dir",
        str(tmp_path / "results"),
        "--mode",
        "full_frame",
    ]
    with pytest.raises(SystemExit):
        rp.parse_args(argv_invalid)


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


def test_calibration_args_forward_requested_bands():
    args = rp.parse_args(_argv("--bands", "zs", "--test_run", "--verbose"))
    calib_args = rp._calibration_args(args, Path("/tmp/cal"), ["zs"])

    assert calib_args[
        calib_args.index("--bands") + 1 : calib_args.index("--solve_wcs")
    ] == ["zs"]
    assert "--test_run" in calib_args
    assert "--verbose" in calib_args


def test_calibration_args_can_omit_band_filter():
    args = rp.parse_args(_argv("--bands", "g_narrow"))
    calib_args = rp._calibration_args(args, Path("/tmp/cal"), None)

    assert "--bands" not in calib_args
    assert "--solve_wcs" in calib_args


def test_parse_args_ref_band_and_refid():
    args = rp.parse_args(_argv("--ref_band", "gp", "--refid", "3"))
    assert args.ref_band == "gp"
    assert args.refid == 3


def test_parse_args_ref_select_quality_and_top_k():
    args = rp.parse_args(_argv("--ref_select", "quality", "--ref_select_top_k", "3"))
    assert args.ref_select == "quality"
    assert args.ref_select_top_k == 3


def test_parse_args_ref_select_rejects_invalid_choice():
    with pytest.raises(SystemExit):
        rp.parse_args(_argv("--ref_select", "best"))


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
            "--centroid_method",
            "com",
        )
    )
    assert args.max_num_stars == 15
    assert args.min_star_separation == pytest.approx(7.5)
    assert args.n_stars_align == 6
    assert args.cutout_size == 41
    assert args.centroid_method == "com"


@pytest.mark.parametrize(
    "method, expected",
    [
        ("auto", "AdaptiveCentroid"),
        ("quad", "CentroidQuadratic"),
        ("com", "CentroidCOM"),
    ],
)
def test_centroid_method_selects_reference_and_photometry_blocks(method, expected):
    reference = rp.reference_sequence(centroid_method=method)
    assert any(block.__class__.__name__ == expected for block in reference.blocks)

    from prose.core.source import Sources

    class FakeReference:
        sources = Sources(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]]))
        epsf = type("FakeEPSF", (), {"params": {}})()

    photometry = rp.photometry_sequence(
        ref=FakeReference(),
        aper_radii=np.array([3.0]),
        rin=6.0,
        rout=9.0,
        n_stars_align=4,
        centroid_method=method,
    )
    assert any(block.__class__.__name__ == expected for block in photometry.blocks)


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


# --------------- detection: faint-target inclusion ---------------


def test_reference_sequence_detects_more_than_max_num_stars():
    """reference_sequence detects DETECT_NUM_STARS_FACTOR times max_num_stars
    so faint targets are captured before truncation back to max_num_stars."""
    seq = rp.reference_sequence(max_num_stars=10)
    detect_block = next(
        b for b in seq.blocks if b.__class__.__name__ == "PointSourceDetection"
    )
    expected_min = max(int(10 * rp.DETECT_NUM_STARS_FACTOR), 10 + 5)
    assert detect_block.n >= expected_min
    assert detect_block.n > 10


def test_photometry_sequence_uses_cross_filter_alignment_tolerance():
    from prose.core.source import Sources

    class FakeReference:
        sources = Sources(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]]))
        epsf = type("FakeEPSF", (), {"params": {}})()

    seq = rp.photometry_sequence(
        ref=FakeReference(),
        aper_radii=np.array([3.0]),
        rin=6.0,
        rout=9.0,
        n_stars_align=4,
    )
    align_blocks = [
        block
        for block in seq.blocks
        if block.__class__.__name__ == "AlignReferenceSources"
    ]

    assert len(align_blocks) == 1
    assert align_blocks[0].discard_tolerance == rp.ALIGN_DISCARD_TOLERANCE


# --------------- aperture-radii / sky-annulus contamination geometry ---------------
#
# These exercise the pure helpers behind gaia_aperture_radii without any FITS
# data or network: a contaminant is a neighbour contributing >=10% of the target
# flux (delta-mag < CONTAM_DMAG = 2.5); the sky annulus is nominally 6-10*FWHM,
# shifted inward to exclude a contaminant, with rout clamped to 100 px on defocus;
# the aperture runs from FWHM up to the inner annulus.


def test_contaminant_seps_keeps_only_bright_enough_neighbours_sorted():
    # target G = 12; neighbours at increasing mag-difference
    seps = np.array([40.0, 10.0, 20.0, 30.0, 50.0])
    mags = np.array([13.4, 11.0, 13.0, 14.5, np.nan])  # dmag: 1.4, -1, 1, 2.5, nan
    contam = rp._contaminant_seps(seps, mags, target_mag=12.0)
    # dmag 2.5 is NOT < 2.5 (excluded); NaN excluded; result sorted ascending
    np.testing.assert_array_equal(contam, [10.0, 20.0, 40.0])


def test_contaminant_seps_excludes_faint_neighbour():
    contam = rp._contaminant_seps(
        np.array([15.0]), np.array([12.0 + 3.0]), target_mag=12.0
    )
    assert contam.size == 0  # dmag 3 -> ~6% flux, below the 10% threshold


def test_sky_annulus_nominal_twenty_to_thirty_pix_without_contaminants():
    rin, rout = rp._sky_annulus_pix(fwhm=5.0, contam_seps=np.array([]))
    assert (rin, rout) == (20.0, 30.0)


def test_sky_annulus_respects_explicit_override():
    rin, rout = rp._sky_annulus_pix(
        fwhm=5.0, contam_seps=np.array([]), annulus_pix=(30, 50)
    )
    assert (rin, rout) == (30.0, 50.0)


def test_sky_annulus_shifts_inward_to_exclude_contaminant():
    # contaminant at 25 px sits inside the nominal 20-30 px ring
    rin, rout = rp._sky_annulus_pix(fwhm=5.0, contam_seps=np.array([25.0]))
    assert rout < 25.0  # ring pulled inside the contaminant
    assert rout == 25.0 - rp.CONTAM_MARGIN_PIX
    assert rin < rout


def test_sky_annulus_ignores_contaminant_beyond_nominal_ring():
    rin, rout = rp._sky_annulus_pix(fwhm=5.0, contam_seps=np.array([80.0]))
    assert (rin, rout) == (20.0, 30.0)  # distant contaminant leaves ring nominal


def test_aperture_radii_span_fwhm_to_inner_annulus():
    radii = rp._aperture_radii_pix(fwhm=5.0, rin=30.0)
    assert radii[0] == 5.0  # minimum aperture is the target FWHM
    assert radii.max() < 30.0  # all radii stay inside the inner annulus
    assert np.all(np.diff(radii) == rp.APER_STEP_PIX)


def test_aperture_radii_never_empty_when_no_room():
    np.testing.assert_array_equal(rp._aperture_radii_pix(fwhm=5.0, rin=5.0), [5.0])


def test_bright_contaminant_yields_smaller_aperture_than_faint_neighbour():
    """End-to-end via the helpers: a >=10%-flux neighbour shrinks the aperture."""
    fwhm = 5.0
    # contaminant at 25 px sits inside the default 20-30 px annulus
    bright = rp._contaminant_seps(np.array([25.0]), np.array([13.0]), 12.0)  # dmag 1
    faint = rp._contaminant_seps(np.array([25.0]), np.array([17.0]), 12.0)  # dmag 5
    rin_bright, _ = rp._sky_annulus_pix(fwhm, bright)
    rin_faint, _ = rp._sky_annulus_pix(fwhm, faint)
    aper_bright = rp._aperture_radii_pix(fwhm, rin_bright)
    aper_faint = rp._aperture_radii_pix(fwhm, rin_faint)
    assert aper_bright.max() < aper_faint.max()
