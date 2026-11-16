"""kSIRIUS (Kagoshima University 1m) instrument support and WCS preparation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import astropy.units as u
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from prose.scripts import ksirius_wcs as kw
from prose.scripts import run_photometry as rp

SHAPE = (400, 420)  # (ny, nx)
CENTER_RADEC = (83.4075, -6.7985)
OPTICAL_SCALE = 0.62  # arcsec/px


def _true_wcs(scale_arcsec: float = OPTICAL_SCALE, rot_deg: float = 4.0) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(CENTER_RADEC)
    w.wcs.crpix = [SHAPE[1] / 2, SHAPE[0] / 2]
    s = scale_arcsec / 3600.0
    t = np.deg2rad(rot_deg)
    w.wcs.cd = np.array(
        [[-s * np.cos(t), s * np.sin(t)], [s * np.sin(t), s * np.cos(t)]]
    )
    return w


def _sky_catalog(wcs: WCS, n: int = 40, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Random stars inside the frame: (radec sorted bright-first, xy)."""
    rng = np.random.default_rng(seed)
    margin = 25
    x = rng.uniform(margin, SHAPE[1] - margin, n)
    y = rng.uniform(margin, SHAPE[0] - margin, n)
    ra, dec = wcs.pixel_to_world_values(x, y)
    return np.column_stack([ra, dec]), np.column_stack([x, y])


def _render(xy: np.ndarray, seed: int = 0) -> np.ndarray:
    """Gaussian stars, brightest first, on a noisy sky-subtracted background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    image = rng.normal(0.0, 5.0, SHAPE)
    for k, (x, y) in enumerate(xy):
        amp = 5000.0 / (1 + 0.15 * k)
        image += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0**2))
    return image.astype(np.float32)


def _ksirius_header(filt: str = "i", instrmen: str = "optcam") -> fits.Header:
    h = fits.Header()
    h["TELESCOP"] = "KU 1m telescope"
    h["INSTRMEN"] = instrmen
    h["OBJECT"] = "tic427460223"
    h["FILTER"] = filt
    h["EXPTIME"] = 40.0
    h["JD"] = 2461348.145228
    h["RA"] = "05:33:37.8"
    h["DEC"] = "-06:47:54.6"
    return h


def _write_frames(
    directory: Path, n: int, xy: np.ndarray, filt: str = "i"
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(n):
        path = directory / f"{filt}261103-{k:04d}skysub.fits"
        fits.writeto(path, _render(xy, seed=k), _ksirius_header(filt), overwrite=True)
        paths.append(str(path))
    return paths


# ---------------------------------------------------------------- detection


@pytest.mark.parametrize("instrmen", ["optcam", "kSIRIUS ", "KSIRIUS"])
def test_ksirius_header_detected_for_both_cameras(instrmen):
    header = _ksirius_header(instrmen=instrmen)

    assert kw.is_ksirius_header(header)
    assert rp.get_instrument(header) == "ksirius"


def test_ksirius_header_requires_the_kagoshima_telescope():
    header = _ksirius_header()
    header["TELESCOP"] = "some other 1m"

    assert not kw.is_ksirius_header(header)
    assert rp.get_instrument(header) == "unknown"


def test_muscat_header_is_not_ksirius():
    header = fits.Header({"INSTRUME": "MuSCAT2", "TELESCOP": "TCS"})

    assert not kw.is_ksirius_header(header)
    assert rp.get_instrument(header) == "muscat2"


@pytest.mark.parametrize(
    ("raw", "band"), [("g", "gp"), ("i", "ip"), ("J", "J"), ("H", "H"), ("Ks", "Ks")]
)
def test_ksirius_filters_resolve_to_canonical_bands(raw, band):
    assert rp._resolve_band(raw, "ksirius", list(kw.KSIRIUS_BANDS)) == band


def test_ksirius_bands_sort_after_optical_in_wavelength_order():
    assert rp.sort_bands_canonical(["Ks", "J", "ip", "H", "gp"]) == list(
        kw.KSIRIUS_BANDS
    )


def test_ksirius_bands_have_distinct_colors():
    colors = [rp.band_color(b) for b in kw.KSIRIUS_BANDS]

    assert len(set(colors)) == len(colors)
    assert "k" not in colors


# ----------------------------------------------------------- default bands


def test_default_bands_switch_to_ksirius_bands():
    # argparse stores DEFAULT_BANDS itself when --bands is not given
    assert rp._instrument_default_bands("ksirius", rp.DEFAULT_BANDS) == list(
        kw.KSIRIUS_BANDS
    )


def test_explicit_default_equal_bands_are_kept_for_ksirius():
    explicit = rp.parse_args(
        [
            "--target_name",
            "t",
            "--data_dir",
            ".",
            "--results_dir",
            ".",
            "--bands",
            *rp.DEFAULT_BANDS,
        ]
    ).bands

    assert rp._instrument_default_bands("ksirius", explicit) == list(rp.DEFAULT_BANDS)


def test_omitted_bands_are_the_default_object():
    args = rp.parse_args(
        ["--target_name", "t", "--data_dir", ".", "--results_dir", "."]
    )

    assert rp._instrument_default_bands("ksirius", args.bands) == list(kw.KSIRIUS_BANDS)


def test_probe_instrument_identifies_nir_only_frames(tmp_path):
    path = tmp_path / "j261103-0001skysub.fits"
    fits.writeto(path, np.zeros((4, 4), np.float32), _ksirius_header("J", "kSIRIUS"))

    instrument = rp._probe_instrument(path)

    assert instrument == "ksirius"
    assert "J" in rp._instrument_default_bands(instrument, rp.DEFAULT_BANDS)


def test_probe_instrument_unreadable_file_is_unknown(tmp_path):
    path = tmp_path / "broken.fits"
    path.write_bytes(b"not a fits file")

    assert rp._probe_instrument(path) == "unknown"


def test_explicit_bands_are_kept_for_ksirius():
    assert rp._instrument_default_bands("ksirius", ["J", "H"]) == ["J", "H"]


def test_default_bands_unchanged_for_other_instruments():
    bands = list(rp.DEFAULT_BANDS)

    assert rp._instrument_default_bands("muscat3", bands) == bands


# ---------------------------------------------------------------- site / BJD


def test_ksirius_observatory_location_is_iriki():
    loc = rp._instrument_location("ksirius")

    assert loc is not None
    assert loc.lat.deg == pytest.approx(31.748, abs=0.01)
    assert loc.lon.deg == pytest.approx(130.44, abs=0.01)


def test_location_falls_back_to_site_registry_for_muscat2(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "astropy.coordinates.EarthLocation.of_site",
        classmethod(lambda cls, name: calls.append(name) or "loc"),
    )

    assert rp._instrument_location("muscat2") == "loc"
    assert calls == ["teide"]


def test_location_is_none_for_unknown_instrument():
    assert rp._instrument_location("unknown") is None


def test_ksirius_telescope_file_saturation_is_honoured():
    class _Tel:
        name = "kSIRIUS_J"
        _default = False
        saturation = 12345.0

    value, source = rp.resolve_saturation(fits.Header(), "ksirius", _Tel())

    assert value == 12345.0
    assert "kSIRIUS_J" in source


# --------------------------------------------------------------- WCS solving


def test_solve_wcs_recovers_known_solution():
    truth = _true_wcs()
    radec, xy = _sky_catalog(truth)

    solved = kw.solve_wcs(xy[:20], radec, OPTICAL_SCALE)

    assert solved is not None
    projected = np.column_stack(solved.world_to_pixel_values(radec[:, 0], radec[:, 1]))
    assert np.max(np.hypot(*(projected - xy).T)) < 0.5
    assert kw.wcs_pixel_scale(solved) == pytest.approx(OPTICAL_SCALE, rel=0.01)


def test_solve_wcs_rejects_scale_far_from_expected():
    truth = _true_wcs()
    radec, xy = _sky_catalog(truth)

    assert kw.solve_wcs(xy[:20], radec, 2 * OPTICAL_SCALE) is None


def test_solve_wcs_returns_none_with_too_few_stars():
    truth = _true_wcs()
    radec, xy = _sky_catalog(truth)

    assert kw.solve_wcs(xy[:2], radec, OPTICAL_SCALE) is None


def test_detect_stars_finds_bright_sources_brightest_first():
    _, xy = _sky_catalog(_true_wcs(), n=10)
    found = kw.detect_stars(_render(xy), trim=0)

    assert len(found) >= 8
    assert np.hypot(*(found[0] - xy[0])) < 1.0


def test_detect_stars_ignores_the_trimmed_border():
    image = _render(np.array([[200.0, 200.0]]))
    image[:5, :] = 1e8  # garbage border like the real optcam frames

    found = kw.detect_stars(image, trim=16)

    assert len(found) == 1
    assert np.hypot(*(found[0] - [200.0, 200.0])) < 1.0


def test_median_stack_uses_frames_around_the_middle(tmp_path):
    paths = []
    for k in range(9):
        path = tmp_path / f"f{k}.fits"
        fits.writeto(path, np.full((4, 4), float(k), dtype=np.float32))
        paths.append(str(path))

    stack = kw.median_stack(paths, nframes=3)

    assert np.all(stack == 4.0)


# ------------------------------------------------------------ WCS copies


def test_prepare_frames_writes_wcs_copies_and_leaves_originals(tmp_path):
    truth = _true_wcs()
    radec, xy = _sky_catalog(truth, n=25)
    originals = _write_frames(tmp_path / "raw", 3, xy)
    out_dir = tmp_path / "wcs"

    result = kw.prepare_ksirius_frames(
        {"ip": originals}, out_dir, gaia_fn=lambda center, radius: radec
    )

    copies = result["ip"]
    assert [Path(p).parent for p in copies] == [out_dir] * 3
    for original, copy in zip(originals, copies):
        header = fits.getheader(copy)
        assert WCS(header).has_celestial
        assert header["PIXSCALE"] == pytest.approx(OPTICAL_SCALE, rel=0.01)
        assert header["WCSMTHD"] == kw.WCS_METHOD
        np.testing.assert_array_equal(fits.getdata(copy), fits.getdata(original))
        assert not WCS(fits.getheader(original)).has_celestial


def test_prepare_frames_reuses_cached_copies(tmp_path):
    truth = _true_wcs()
    radec, xy = _sky_catalog(truth, n=25)
    originals = _write_frames(tmp_path / "raw", 2, xy)
    out_dir = tmp_path / "wcs"
    kw.prepare_ksirius_frames({"ip": originals}, out_dir, gaia_fn=lambda c, r: radec)
    stamps = {p: os.stat(p).st_mtime_ns for p in out_dir.glob("*.fits")}

    def _fail(center, radius):
        raise AssertionError("cached band must not be re-solved")

    result = kw.prepare_ksirius_frames({"ip": originals}, out_dir, gaia_fn=_fail)

    assert len(result["ip"]) == 2
    assert {p: os.stat(p).st_mtime_ns for p in out_dir.glob("*.fits")} == stamps


def test_prepare_frames_keeps_originals_when_solve_fails(tmp_path, caplog):
    _, xy = _sky_catalog(_true_wcs(), n=25)
    originals = _write_frames(tmp_path / "raw", 2, xy)
    rng = np.random.default_rng(11)  # scattered (collinear points stall twirl)
    unrelated_sky = np.column_stack(
        [10.0 + rng.uniform(0, 0.05, 30), -5.0 + rng.uniform(0, 0.05, 30)]
    )

    result = kw.prepare_ksirius_frames(
        {"ip": originals}, tmp_path / "wcs", gaia_fn=lambda c, r: unrelated_sky
    )

    assert result["ip"] == originals
    assert "WCS" in caplog.text


def test_prepare_frames_keeps_originals_when_gaia_fails(tmp_path, caplog):
    _, xy = _sky_catalog(_true_wcs(), n=25)
    originals = _write_frames(tmp_path / "raw", 2, xy)

    def _offline(center, radius):
        raise OSError("network down")

    result = kw.prepare_ksirius_frames(
        {"ip": originals}, tmp_path / "wcs", gaia_fn=_offline
    )

    assert result["ip"] == originals
    assert "network down" in caplog.text


def test_telescope_files_agree_with_camera_pixel_scales():
    import yaml

    data_dir = Path(__file__).resolve().parents[2] / "data"
    camera_for_band = {
        "g": "optcam",
        "i": "optcam",
        "J": "ksirius",
        "H": "ksirius",
        "Ks": "ksirius",
    }
    for band, camera in camera_for_band.items():
        spec = yaml.safe_load((data_dir / f"ksirius_{band}.telescope").read_text())
        assert spec["name"] == f"kSIRIUS_{band}"
        assert "KU 1m telescope" in spec["names"]
        assert spec["pixel_scale"] == kw.KSIRIUS_PIXEL_SCALES[camera]
        assert spec["time_reference"] == "start"


# ----------------------------------------------------- known-scale solver


NIR_SHAPE = (255, 320)
NIR_SCALE = 0.694


def _nir_field(parity: int, angle: float, n_field: int = 6, n_outside: int = 30):
    """Sparse NIR-like field: a few stars on chip, Gaia distractors around it."""
    from astropy.coordinates import SkyCoord

    center = SkyCoord(*CENTER_RADEC, unit="deg")
    origin = np.array([NIR_SHAPE[1] / 2 + 12.0, NIR_SHAPE[0] / 2 - 30.0])
    truth = kw._known_scale_wcs(center, NIR_SCALE, parity, angle, origin)
    rng = np.random.default_rng(7)
    xy = np.column_stack(
        [
            rng.uniform(20, NIR_SHAPE[1] - 20, n_field),
            rng.uniform(20, NIR_SHAPE[0] - 20, n_field),
        ]
    )
    on_chip = np.column_stack(truth.pixel_to_world_values(xy[:, 0], xy[:, 1]))
    ring = rng.uniform(-220, 540, (n_outside, 2))  # mostly off-chip pixels
    off_chip = np.column_stack(truth.pixel_to_world_values(ring[:, 0], ring[:, 1]))
    # interleave so on-chip stars are not simply the brightest Gaia entries
    radec = np.vstack([off_chip[:10], on_chip, off_chip[10:]])
    return center, truth, xy, radec


@pytest.mark.parametrize(("parity", "angle"), [(-1, 355.5), (1, 37.0), (-1, 180.0)])
def test_known_scale_solve_recovers_sparse_field(parity, angle):
    center, truth, xy, radec = _nir_field(parity, angle)

    solved = kw.solve_wcs_known_scale(xy, radec, center, NIR_SCALE, NIR_SHAPE)

    assert solved is not None
    projected = np.column_stack(solved.world_to_pixel_values(*radec[10:16].T))
    assert np.max(np.hypot(*(projected - xy).T)) < 2.0
    assert kw.wcs_pixel_scale(solved) == pytest.approx(NIR_SCALE, rel=1e-6)


def test_known_scale_solve_rejects_field_without_counterparts():
    center, _, xy, _ = _nir_field(-1, 10.0)
    rng = np.random.default_rng(3)
    unrelated = np.column_stack(
        [
            CENTER_RADEC[0] + rng.uniform(-0.05, 0.05, 40),
            CENTER_RADEC[1] + rng.uniform(-0.05, 0.05, 40),
        ]
    )

    assert kw.solve_wcs_known_scale(xy, unrelated, center, NIR_SCALE, NIR_SHAPE) is None


def test_known_scale_solve_needs_minimum_detections():
    center, _, xy, radec = _nir_field(-1, 10.0)

    assert kw.solve_wcs_known_scale(xy[:3], radec, center, NIR_SCALE, NIR_SHAPE) is None


def test_solve_band_falls_back_to_known_scale(tmp_path, monkeypatch):
    center, truth, xy, radec = _nir_field(-1, 355.5)
    monkeypatch.setattr(kw, "solve_wcs", lambda *a, **k: None)  # twirl fails
    monkeypatch.setattr(kw, "detect_stars", lambda image, trim=0: xy)
    path = tmp_path / "j261103-0001skysub.fits"
    header = _ksirius_header("J", "kSIRIUS")
    header["RA"], header["DEC"] = center.ra.deg, center.dec.deg
    fits.writeto(path, np.zeros(NIR_SHAPE, dtype=np.float32), header)

    solved = kw.solve_band([str(path)], NIR_SCALE, gaia_fn=lambda c, r: radec)

    assert solved is not None
    assert kw._count_matches(solved, xy, radec) == len(xy)


# ------------------------------------------------- caching / solve frames


def test_prepare_frames_solves_on_solve_sciences(tmp_path, monkeypatch):
    truth = _true_wcs()
    radec, xy = _sky_catalog(truth, n=25)
    window = _write_frames(tmp_path / "raw", 2, xy)
    full = window + _write_frames(tmp_path / "raw_more", 3, xy)
    stacked = []
    real_stack = kw.median_stack

    def _spy(files, nframes=kw.STACK_NFRAMES):
        stacked.append(list(files))
        return real_stack(files, nframes)

    monkeypatch.setattr(kw, "median_stack", _spy)

    result = kw.prepare_ksirius_frames(
        {"ip": window},
        tmp_path / "wcs",
        gaia_fn=lambda c, r: radec,
        solve_sciences={"ip": full},
    )

    assert stacked == [full]
    assert len(result["ip"]) == 2


def test_cached_sidecar_wcs_is_reused_for_new_frames(tmp_path):
    truth = _true_wcs()
    radec, xy = _sky_catalog(truth, n=25)
    first = _write_frames(tmp_path / "raw", 2, xy)
    out_dir = tmp_path / "wcs"
    kw.prepare_ksirius_frames({"ip": first}, out_dir, gaia_fn=lambda c, r: radec)
    assert kw.sidecar_path(out_dir, "ip", "tic427460223").exists()
    more = _write_frames(tmp_path / "raw2", 3, xy)

    def _fail(center, radius):
        raise AssertionError("cached WCS must not be re-solved")

    result = kw.prepare_ksirius_frames({"ip": more}, out_dir, gaia_fn=_fail)

    assert all(WCS(fits.getheader(p)).has_celestial for p in result["ip"])


# ------------------------------------------------ run_photometry defaults


def test_default_trim_for_ksirius():
    assert rp._instrument_default_trim("ksirius", rp.CCD_TRIM_SIZE_YX) == (16, 16)


def test_explicit_zero_trim_is_kept_for_ksirius():
    explicit = rp.parse_args(
        [
            "--target_name",
            "t",
            "--data_dir",
            ".",
            "--results_dir",
            ".",
            "--ccd_trim",
            "0,0",
        ]
    ).ccd_trim_size_yx

    assert rp._instrument_default_trim("ksirius", explicit) == (0, 0)


def test_explicit_trim_is_kept_for_ksirius():
    assert rp._instrument_default_trim("ksirius", (4, 2)) == (4, 2)


def test_default_trim_unchanged_for_other_instruments():
    assert rp._instrument_default_trim("muscat3", rp.CCD_TRIM_SIZE_YX) == tuple(
        rp.CCD_TRIM_SIZE_YX
    )


def test_pixel_scale_falls_back_to_matched_telescope_file():
    class _Tel:
        name = "kSIRIUS_J"
        _default = False
        pixel_scale = 0.694

    assert rp.resolve_pixel_scale({}, "ksirius", _Tel()) == (
        0.694,
        "kSIRIUS_J telescope file",
    )


def test_pixel_scale_ignores_default_telescope():
    class _Tel:
        name = "default"
        _default = True
        pixel_scale = 0.5

    _, source = rp.resolve_pixel_scale({}, "unknown", _Tel())

    assert source.startswith("guess")


def test_sidecar_is_not_reused_for_another_target(tmp_path):
    truth = _true_wcs()
    radec, xy = _sky_catalog(truth, n=25)
    out_dir = tmp_path / "wcs"
    kw.prepare_ksirius_frames(
        {"ip": _write_frames(tmp_path / "a", 2, xy)},
        out_dir,
        gaia_fn=lambda c, r: radec,
    )
    other = _write_frames(tmp_path / "b", 2, xy)
    for path in other:
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["OBJECT"] = "another_target"
    solved_again = []

    def _gaia(center, radius):
        solved_again.append(center)
        return radec

    kw.prepare_ksirius_frames({"ip": other}, out_dir, gaia_fn=_gaia)

    assert len(solved_again) == 1


def test_sidecar_is_not_reused_for_a_different_pointing(tmp_path):
    truth = _true_wcs()
    radec, xy = _sky_catalog(truth, n=25)
    out_dir = tmp_path / "wcs"
    first = _write_frames(tmp_path / "a", 2, xy)
    kw.prepare_ksirius_frames({"ip": first}, out_dir, gaia_fn=lambda c, r: radec)
    for path in first:  # same target name, pointed 10' away on another night
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["DEC"] = "-06:37:54.6"
    solved_again = []

    def _gaia(center, radius):
        solved_again.append(center)
        return radec

    kw.prepare_ksirius_frames({"ip": first}, out_dir, gaia_fn=_gaia)

    assert len(solved_again) == 1


def test_stale_copy_is_rewritten_with_the_current_wcs(tmp_path):
    truth = _true_wcs()
    radec, xy = _sky_catalog(truth, n=25)
    originals = _write_frames(tmp_path / "raw", 1, xy)
    out_dir = tmp_path / "wcs"
    stale = _true_wcs(rot_deg=30.0)
    kw.write_wcs_copies(originals, stale, out_dir)

    copies = kw.write_wcs_copies(originals, truth, out_dir)

    assert kw._same_wcs(WCS(fits.getheader(copies[0])), truth)


def test_unwritable_output_falls_back_to_original_frames(tmp_path, caplog):
    truth = _true_wcs()
    radec, xy = _sky_catalog(truth, n=25)
    originals = _write_frames(tmp_path / "raw", 2, xy)
    blocker = tmp_path / "wcs"
    blocker.write_text("a file where the output directory should go")

    result = kw.prepare_ksirius_frames(
        {"ip": originals}, blocker, gaia_fn=lambda c, r: radec
    )

    assert result["ip"] == originals
    assert "WCS preparation failed" in caplog.text


def test_twirl_solution_needs_most_detections_matched(monkeypatch):
    truth = _true_wcs()
    radec, xy = _sky_catalog(truth, n=10)
    rng = np.random.default_rng(5)
    spurious = rng.uniform(30, 370, (12, 2))  # detections with no Gaia counterpart
    detections = np.vstack([xy[:5], spurious])

    assert kw.solve_wcs(detections, radec, OPTICAL_SCALE) is None


def test_known_scale_solve_rejects_offset_beyond_pointing_margin():
    center, truth, xy, radec = _nir_field(-1, 355.5)
    far = kw.POINTING_MARGIN_ARCSEC / NIR_SCALE + 40  # px
    shifted = xy + [far, 0.0]

    assert (
        kw.solve_wcs_known_scale(shifted, radec, center, NIR_SCALE, NIR_SHAPE) is None
    )


def test_known_scale_wcs_follows_the_standard_sky_orientation():
    """Independent of the solver: north up, east left for parity -1, angle 0."""
    from astropy.coordinates import SkyCoord

    center = SkyCoord(*CENTER_RADEC, unit="deg")
    origin = np.array([100.0, 80.0])
    wcs = kw._known_scale_wcs(center, NIR_SCALE, -1, 0.0, origin)
    east = center.spherical_offsets_by(36 * u.arcsec, 0 * u.arcsec)
    north = center.spherical_offsets_by(0 * u.arcsec, 36 * u.arcsec)

    ex, ey = wcs.world_to_pixel(east)
    nx, ny = wcs.world_to_pixel(north)

    assert (float(ex), float(ey)) == pytest.approx((100 - 36 / NIR_SCALE, 80), abs=0.05)
    assert (float(nx), float(ny)) == pytest.approx((100, 80 + 36 / NIR_SCALE), abs=0.05)


@pytest.mark.parametrize("flip_x", [True, False])
@pytest.mark.parametrize("angle", [0.0, 23.0, 200.0])
def test_known_scale_solve_recovers_independently_built_wcs(flip_x, angle):
    """Truth from astropy CDELT/PC keywords, not from the solver's own helper."""
    from astropy.coordinates import SkyCoord

    center = SkyCoord(*CENTER_RADEC, unit="deg")
    truth = WCS(naxis=2)
    truth.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    truth.wcs.crval = list(CENTER_RADEC)
    truth.wcs.crpix = [NIR_SHAPE[1] / 2 + 10.5, NIR_SHAPE[0] / 2 - 20.5]
    s = NIR_SCALE / 3600
    truth.wcs.cdelt = [-s if flip_x else s, s]
    t = np.deg2rad(angle)
    truth.wcs.pc = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    rng = np.random.default_rng(int(angle) + flip_x)
    xy = np.column_stack(
        [rng.uniform(20, NIR_SHAPE[1] - 20, 6), rng.uniform(20, NIR_SHAPE[0] - 20, 6)]
    )
    on_chip = np.column_stack(truth.pixel_to_world_values(xy[:, 0], xy[:, 1]))
    ring = rng.uniform(-220, 540, (30, 2))
    off_chip = np.column_stack(truth.pixel_to_world_values(ring[:, 0], ring[:, 1]))
    radec = np.vstack([off_chip[:10], on_chip, off_chip[10:]])

    solved = kw.solve_wcs_known_scale(xy, radec, center, NIR_SCALE, NIR_SHAPE)

    assert solved is not None
    projected = np.column_stack(solved.world_to_pixel_values(*on_chip.T))
    assert np.max(np.hypot(*(projected - xy).T)) < 2.0
