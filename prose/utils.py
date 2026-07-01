import csv
import inspect
import urllib
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor
from rich.progress import track

import astropy.constants as c
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from scipy import ndimage
from astroquery.simbad import Simbad

earth2sun = (c.R_earth / c.R_sun).value

# Default wall-clock timeout (seconds) for external network queries (Gaia,
# SIMBAD, MAST, BJD service). Bounds otherwise-unbounded calls so an
# unreachable service cannot hang the pipeline indefinitely.
NETWORK_TIMEOUT_S = 30

FOV_IN_ARCMIN = {
    'sinistro_full': 26.5, # CONFMODE= 'full_frame'
    'sinistro_2x2': 13,    # CONFMODE= 'central_2k_2x2'
    'muscat': 6.1,
    'muscat2': 7.4,
    'muscat3': 9.1,
    'muscat4': 9.1
}

PIXSCALES = {
    'sinistro_full': 0.389, #CONFMODE= 'full_frame'
    'sinistro_2x2': 0.778, #CONFMODE= 'central_2k_2x2'
    'muscat': 0.358,
    'muscat2': 0.44,
    'muscat3': 0.267,
    'muscat4': 0.267,
}

LCO_SITES = {
    # astroplan observatory site codes
    # LCO-1m
    "LCOGT node at SAAO": "saao",
    "LCOGT node at Tenerife": "teide",
    "LCOGT node at McDonald Observatory": "McDonald",
    "LCOGT node at Cerro Tololo Inter-American Observatory": "cerro tololo interamerican observatory",
    # LCO-2m
    "LCOGT node at Haleakala Observatory": "Haleakala",
    "LCOGT node at Siding Spring Observatory": "Siding Spring Observatory",
}

LCO_CODES = {
    #lco observatory site codes
    "LCOGT node at Siding Spring Observatory": "coj",
    "LCOGT node at Cerro Tololo Inter-American Observatory": "lsc",
    "LCOGT node at SAAO": "cpt",
    "LCOGT node at Tenerife": "tfn",
}

# persistent, coordinate-keyed catalog cache shared by the Gaia and SIMBAD
# queries: results are written under CACHE_DIR/<subdir> and reused when a live
# query is unavailable (offline fallback for future reruns).
CACHE_DIR = Path.home() / ".cache" / "prose_photometry"


def coord_cache_path(subdir, target_coord, *key_parts, ext="csv"):
    """Path to a coordinate-keyed cache file under ``CACHE_DIR/subdir``.

    The filename encodes RA/Dec (5 dp ~ 0.4") plus any extra ``key_parts``
    (e.g. cutout size, instrument, field of view) so distinct queries of the
    same target do not collide.
    """
    ra = round(float(target_coord.ra.deg), 5)
    dec = round(float(target_coord.dec.deg), 5)
    name = f"ra{ra:.5f}_dec{dec:+.5f}"
    suffix = "_".join(str(k) for k in key_parts)
    if suffix:
        name += f"_{suffix}"
    return CACHE_DIR / subdir / f"{name}.{ext}"


def load_cached_df(path):
    """Load a cached DataFrame, or ``None`` if the file is absent/unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001 - a bad cache must not be fatal
        return None


def save_cached_df(path, df):
    """Persist a DataFrame to the cache (best effort). Returns success."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return True
    except Exception:  # noqa: BLE001 - caching is best effort
        return False


_simbad_cache: dict = {}

def get_simbad_data(target_coord, inst, fov_arcmin=None):
    """
    Query SIMBAD sources around a sky coordinate.

    The result is cached in memory (per process) and on disk
    (``CACHE_DIR/simbad``). When the live query fails the cached result from a
    previous run is reused; a genuine "no sources" answer is not cached to disk.

    Parameters:
        target_coord : SkyCoord
            Target coordinates
        inst : str
            Instrument name
        fov_arcmin : float or None
            Field of view in arcminutes

    Returns:
        pandas.DataFrame : filtered SIMBAD table with objects in radius
    """
    url = "https://simbad.cds.unistra.fr/Pages/guide/otypes.htx"
    Simbad.add_votable_fields("otype")

    if inst.lower() == 'sinistro':
        inst += '_2x2'
    fov_default = FOV_IN_ARCMIN.get(inst, 5)
    fov = fov_arcmin if fov_arcmin else fov_default

    cache_key = (round(target_coord.ra.deg, 5), round(target_coord.dec.deg, 5), inst, fov)
    if cache_key in _simbad_cache:
        return _simbad_cache[cache_key]

    cache_path = coord_cache_path("simbad", target_coord, inst, f"{fov:g}")
    try:
        print(f"Querying SIMBAD sources within {fov:.2f} arcmin of ({target_coord.to_string('decimal')})")
        result = _run_with_timeout(
            lambda: Simbad.query_region(target_coord, radius=fov * u.arcmin),
            NETWORK_TIMEOUT_S,
        )
        if result is None:
            print("No sources found.")
            result_df = None
        else:
            df = result.to_pandas()
            coords = SkyCoord(ra=df.RA, dec=df.DEC, unit=(u.hourangle, u.deg))
            result_df = df[coords.separation(target_coord) < fov * u.arcmin]
            save_cached_df(cache_path, result_df)  # refresh cache whenever online
            print(f"For description of Simbad object types, see\n{url}")
    except Exception as exc:  # noqa: BLE001 - degrade to cached result when offline
        result_df = load_cached_df(cache_path)
        if result_df is not None:
            print(f"SIMBAD query unavailable ({exc}); using cached result {cache_path}")
        else:
            print(f"SIMBAD query unavailable ({exc}); no cache available")

    _simbad_cache[cache_key] = result_df
    return result_df

def get_saturation_from_header(h) -> dict:
    """Only works for LCO data header"""
    gain = h['GAIN']
    # 2m0a telescopes (MuSCAT)
    if h['TELID'] == '2m0a':
        is_narrow = h['filter'].endswith('_narrow')
        nb_names = ['g_narrow','Na_D','i_narrow','z_narrow']
        bb_names = ['gp','rp','ip','zs']
        sat_m3 = [120_000/1.9, int(120_000/1.88), int(82_000/1.8), int(100_000/2.0)]
        sat_m4 = [64_000, 64_000, 46_000, 64_000]
        url = "https://lco.global/observatory/instruments/muscat/"
        
        if h['SITEID'] == 'coj':  # MuSCAT4
            if is_narrow:
                saturation_limits = {b: s for b,s in zip(nb_names,sat_m4)}  
            else: 
                saturation_limits = {b: s for b,s in zip(bb_names,sat_m4)}
        elif h['SITEID'] == 'ogg':  # MuSCAT3
            if is_narrow:
                saturation_limits = {b: s for b,s in zip(nb_names,sat_m3)}  
            else: 
                saturation_limits = {b: s for b,s in zip(bb_names,sat_m3)}
        else:
            raise ValueError("Site ID must be 'coj' or 'ogg' for 2m0a telescopes")
    
    # 1m0a telescopes (Sinistro)
    elif h['TELID'] == '1m0a':
        sinistro_sites = ['lsc', 'cpt', 'coj', 'tfn', 'elp']
        if h['SITEID'] not in sinistro_sites:
            raise ValueError(f"Site ID must be one of {sinistro_sites} for 1m0a telescopes")
        
        url = "https://lco.global/observatory/instruments/sinistro/"
        if float(h.get('GAIN', 1.0)) == 1.0 and 'SATURATE' in h:
            base_limit = h['SATURATE']
            gain = 1.0
        else:
            gain = 6.6 if h.get('CONFMODE') == 'central_2k_2x2' else 1.0
            base_limit = 340_000 / gain
        saturation_limits = {'gp': base_limit, 'rp': base_limit, 'ip': base_limit, 'zs': base_limit}
    
    else:
        raise ValueError("This doesn't look like LCO data. Function only works with LCO telescopes.")
    
    # Print reference info
    unit = 'e-' if gain==1 else 'ADU'
    print(f"Header saturation: {h['SATURATE']:,} [{unit}], Max linearity: {h['MAXLIN']:,} [{unit}]")
    print(f"Reference: {url}")
    
    return saturation_limits

# common FITS header FILTER value -> canonical band name aliases
_FILTER_ALIASES: dict[str, str] = {
    "gp": "gp",
    "g": "gp",
    "g_narrow": "g_narrow",
    "g_wide": "g_wide",
    "rp": "rp",
    "r": "rp",
    "rp*diffuser": "rp",
    "r_narrow": "r_narrow",
    "ip": "ip",
    "i": "ip",
    "i_narrow": "i_narrow",
    "zs": "zs",
    "z": "zs",
    "zp": "zs",
    "z_s": "zs",
    "z_narrow": "z_narrow",
    "zp*diffuser": "zs",
    "Na_D": "Na_D",
}

# Canonical band ordering used for display and output products.
# Broadband first (Sloan g/r/i/z), then narrowband extras.
# This order is mirrored by muscat-db's band_utils.py (web-layer copy).
DEFAULT_BROAD_BANDS: list[str] = ["gp", "rp", "ip", "zs"]
DEFAULT_NARROW_BANDS: list[str] = ["g_narrow", "Na_D", "i_narrow", "z_narrow"]


def bands_from_filters(
    filters: list[str],
    aliases: dict[str, str] | None = None,
) -> list[str]:
    """Map raw FITS FILTER header values to ordered, de-duplicated band tokens.

    Each raw filter is normalised via *aliases* (defaults to
    :data:`_FILTER_ALIASES`); unknown values (e.g. Johnson ``R``/``V``/``B``)
    pass through unchanged. The result is sorted canonically —
    :data:`DEFAULT_BROAD_BANDS`, then :data:`DEFAULT_NARROW_BANDS`, then any
    extras in first-seen order — so callers always get a stable, familiar layout.

    Returns ``[]`` for empty input.

    Note
    ----
    Do NOT case-fold the raw filter value: Johnson ``R``/``V`` must not
    collapse into Sloan ``rp``/etc.
    """
    _aliases = aliases if aliases is not None else _FILTER_ALIASES
    seen: set[str] = set()
    tokens: list[str] = []
    for f in filters or []:
        if not f:
            continue
        token = _aliases.get(f, f)
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    order = {b: i for i, b in enumerate([*DEFAULT_BROAD_BANDS, *DEFAULT_NARROW_BANDS])}
    return sorted(tokens, key=lambda b: (order.get(b, len(order)), tokens.index(b)))


OBSLOG_ROOT = "/ut2/muscat/obslog"


def frames_from_obslog(data_dir, instrument: str | None = None) -> list[dict] | None:
    """Resolve frame metadata from MuSCAT obslog CSVs without opening FITS files.

    The obslog lives at ``<OBSLOG_ROOT>/<instrument>/<date>/`` and holds one
    ``obslog-*-ccd<N>.csv`` per CCD with ``FRAME``, ``OBJECT`` and ``FILTER``
    columns. *instrument* defaults to ``data_dir.parent.name`` and ``<date>`` to
    ``data_dir.name`` (instrument is lowercased to match the obslog layout).

    Returns a list of ``{"frame", "object", "filter", "exposure", "ccd", "path"}``
    dicts for every logged frame whose ``.fits`` file exists in *data_dir*, or
    ``None`` when no obslog directory is present, so callers can fall back to a
    header scan. ``exposure`` is the ``EXPTIME (s)`` column as a float, or ``None``
    when absent/unparseable.
    """
    data_dir = Path(data_dir)
    instrument = (instrument or data_dir.parent.name).lower()
    obslog_dir = Path(OBSLOG_ROOT) / instrument / data_dir.name
    if not obslog_dir.is_dir():
        return None

    # One directory listing instead of a per-frame ``is_file()`` stat (thousands
    # of stats on slow/networked storage otherwise).
    on_disk = {p.name for p in data_dir.glob("*.fits")}

    records: list[dict] = []
    for ccd_csv in sorted(obslog_dir.glob("obslog-*-ccd?.csv")):
        try:
            ccd = int(ccd_csv.stem.rsplit("ccd", 1)[1])
        except (IndexError, ValueError):
            ccd = None
        with open(ccd_csv) as f:
            reader = csv.DictReader(f)
            # Exposure column carries a unit, e.g. "EXPTIME (s)"; match defensively.
            exp_col = next(
                (
                    name
                    for name in (reader.fieldnames or [])
                    if name and name.strip().upper().startswith("EXPTIME")
                ),
                None,
            )
            mode_col = next(
                (
                    name
                    for name in (reader.fieldnames or [])
                    if name and name.strip().upper() in ("CONFMODE", "MODE", "CONF_MODE")
                ),
                None,
            )
            for row in reader:
                frame = (row.get("FRAME") or "").strip()
                fname = f"{frame}.fits"
                if not frame or fname not in on_disk:
                    continue
                exposure: float | None = None
                if exp_col is not None:
                    try:
                        exposure = float((row.get(exp_col) or "").strip())
                    except (TypeError, ValueError):
                        exposure = None
                confmode = None
                if mode_col is not None:
                    confmode = row.get(mode_col)
                records.append(
                    {
                        "frame": frame,
                        "object": (row.get("OBJECT") or "").strip(),
                        "filter": (row.get("FILTER") or "").strip(),
                        "exposure": exposure,
                        "ccd": ccd,
                        "path": str(data_dir / fname),
                        "confmode": (str(confmode).strip() if confmode else None),
                    }
                )
    return records


def scan_fits_headers(
    files: list,
    keys=("OBJECT",),
    ext: int = 0,
    description: str = "Scanning files",
) -> list[tuple[str, dict]]:
    """Read selected header keywords from many FITS files in parallel.

    Header reads are I/O-bound, so they are fanned out across a thread pool — a
    large win on slow/networked storage versus a serial loop. Returns a list of
    ``(path, {key: value})`` in the same order as *files*; unreadable files yield
    an empty mapping.
    """

    def _read(fp):
        try:
            header = fits.getheader(fp, memmap=True, ext=ext)
            return str(fp), {k: str(header.get(k, "")).strip() for k in keys}
        except Exception:
            return str(fp), {}

    with ThreadPoolExecutor() as executor:
        return list(
            track(
                executor.map(_read, files),
                total=len(files),
                description=description,
            )
        )


def read_filename_per_band(sciences: list, bands: list, target_name: str, ext: int = 0, filter_aliases: dict[str, str] | None = None) -> dict:
    """
    Collect FITS files by filter band for a specific target.

    Parameters:
        sciences (list): List of file paths to FITS files.
        bands (list): List of filter names to include.
        target_name (str): Name of the target object to match.

    Returns:
        dict: A dictionary {band: [file_paths]} of matching FITS files.
    """

    def _band_for(raw: str) -> str | None:
        aliases = filter_aliases or _FILTER_ALIASES
        band = aliases.get(raw)
        if band is not None and band in bands:
            return band
        band = _FILTER_ALIASES.get(raw)
        if band is not None and band in bands:
            return band
        if raw in bands:
            return raw
        return None

    data = {b: [] for b in bands}
    for fp, header in scan_fits_headers(
        sciences,
        keys=("OBJECT", "FILTER"),
        ext=ext,
        description="Scanning science files",
    ):
        if header.get("OBJECT") == target_name:
            band = _band_for(header.get("FILTER", ""))
            if band is not None:
                data[band].append(fp)
    return data

def remove_sip(dict_like):
    for kw in [
        "A_ORDER",
        "A_0_2",
        "A_1_1",
        "A_2_0",
        "B_ORDER",
        "B_0_2",
        "B_1_1",
        "B_2_0",
        "AP_ORDER",
        "AP_0_0",
        "AP_0_1",
        "AP_0_2",
        "AP_1_0",
        "AP_1_1",
        "AP_2_0",
        "BP_ORDER",
        "BP_0_0",
        "BP_0_1",
        "BP_0_2",
        "BP_1_0",
        "BP_1_1",
        "BP_2_0",
    ]:
        if kw in dict_like:
            del dict_like[kw]


def format_iso_date(date, night_date=True):
    """
    Return a datetime.date corresponding to the day 12 hours before given datetime.
    Used as a reference day, e.g. if a target is observed the 24/10 at 02:30, observation date
    is 23/10, day when night begin.

    Parameters
    ----------
    date : str or datetime
        if str: "fits" fromated date and time
        if datetime: datetime
    night_date : bool, optional
        return day 12 hours before given date and time, by default True

    Returns
    -------
    datetime.date
        formatted date
    """
    if isinstance(date, str):
        date = Time(date, format="fits").datetime
    elif isinstance(date, datetime):
        date = Time(date, format="datetime").datetime

    if night_date:
        return (
            date - timedelta(hours=15)
        ).date()  # If obs goes up to 15pm it still belongs to day before
    else:
        return date


def std_diff_metric(fluxes):
    k = len(list(np.shape(fluxes)))
    return np.std(np.diff(fluxes, axis=k - 1), axis=k - 1)


def stability_aperture(fluxes):
    lc_c = np.abs(np.diff(fluxes, axis=0))
    return np.mean(lc_c, axis=1)


def index_binning(x, size):
    if isinstance(size, float):
        bins = np.arange(np.min(x), np.max(x), size)
    else:
        x = np.arange(0, len(x))
        bins = np.arange(0.0, len(x), size)

    d = np.digitize(x, bins)
    n = np.max(d) + 2
    indexes = []

    for i in range(0, n):
        s = np.where(d == i)
        if len(s[0]) > 0:
            s = s[0]
            indexes.append(s)

    return indexes


def binning(time, flux, bins, error=None, std=True):
    """Bin a time series and return binned time, flux, and uncertainty arrays."""
    time = np.asarray(time)
    flux = np.asarray(flux)
    if time.shape[0] != flux.shape[0]:
        raise ValueError("time and flux must have the same length")
    if error is not None:
        error = np.asarray(error)
        if error.shape[0] != flux.shape[0]:
            raise ValueError("error and flux must have the same length")

    idxs = index_binning(time, bins)
    binned_time = np.array([np.nanmean(time[i]) for i in idxs])
    binned_flux = np.array([np.nanmean(flux[i]) for i in idxs])
    if std or error is None:
        binned_error = np.array([np.nanstd(flux[i]) / np.sqrt(len(i)) for i in idxs])
    else:
        binned_error = np.array(
            [np.sqrt(np.nansum(error[i] ** 2)) / len(i) for i in idxs]
        )
    return binned_time, binned_flux, binned_error


def z_scale(data, c=0.05):
    interval = ZScaleInterval(contrast=c)
    return interval(data.copy())


def rescale(y):
    ry = y - np.mean(y)
    return ry / np.std(ry)


def check_class(_class, base, default):
    if _class is None:
        return default
    elif isinstance(_class, base):
        return _class
    else:
        raise TypeError("subclass of {} expected".format(base.__name__))


def divisors(n):
    _divisors = []
    i = 1
    while i <= n:
        if n % i == 0:
            _divisors.append(i)
        i = i + 1
    return np.array(_divisors)


def fold(t, t0, p):
    return (t - t0 + 0.5 * p) % p - 0.5 * p


def header_to_cdf4_dict(header):
    header_dict = {}

    for key, value in header.items():
        if isinstance(value, str):
            if len(key) > 0 and len(value) > 0:
                header_dict[key] = value
        elif isinstance(value, (float, np.ndarray, np.number)):
            header_dict[key] = float(value)
        elif isinstance(value, (int, bool)):
            header_dict[key] = int(value)
        else:
            pass

    return header_dict


def years_to_datetime(years):
    """
    https://stackoverflow.com/questions/19305991/convert-fractional-years-to-a-real-date-in-python
    Convert atime (a float) to DT.datetime
    This is the inverse of dt2t.
    assert dt2t(t2dt(atime)) == atime
    """
    year = int(years)
    remainder = years - year
    boy = datetime(year, 1, 1)
    eoy = datetime(year + 1, 1, 1)
    seconds = remainder * (eoy - boy).total_seconds()
    return boy + timedelta(seconds=seconds)


def datetime_to_years(adatetime):
    """
    https://stackoverflow.com/questions/19305991/convert-fractional-years-to-a-real-date-in-python
    Convert adatetime into a float. The integer part of the float should
    represent the year.
    Order should be preserved. If adate<bdate, then d2t(adate)<d2t(bdate)
    time distances should be preserved: If bdate-adate=ddate-cdate then
    dt2t(bdate)-dt2t(adate) = dt2t(ddate)-dt2t(cdate)
    """
    year = adatetime.year
    boy = datetime(year, 1, 1)
    eoy = datetime(year + 1, 1, 1)
    return year + ((adatetime - boy).total_seconds() / ((eoy - boy).total_seconds()))


def split(x, dt, fill=None):
    splits = np.argwhere(np.diff(x) > dt).flatten() + 1
    xs = np.split(x, splits)
    if fill is None:
        return xs
    else:
        ones = np.ones_like(x)
        filled_xs = [np.split(ones * fill, splits) for _ in xs]
        for i in range(len(xs)):
            filled_xs[i][i] = xs[i]
        for i in range(len(xs)):
            filled_xs[i] = np.hstack(filled_xs[i])
        return [np.hstack(fx) for fx in filled_xs]


def jd_to_bjd(jd, ra, dec, timeout=NETWORK_TIMEOUT_S):
    """
    Convert JD to BJD using http://astroutils.astronomy.ohio-state.edu (Eastman et al. 2010)
    """
    bjd = urllib.request.urlopen(
        f"http://astroutils.astronomy.ohio-state.edu/time/convert.php?JDS={','.join(jd.astype(str))}&RA={ra}&DEC={dec}&FUNCTION=utc2bjd",
        timeout=timeout,
    ).read()
    bjd = bjd.decode("utf-8")
    return np.array(bjd.split("\n"))[0:-1].astype(float)


def remove_arrays(d):
    copy = d.copy()
    for name, value in d.items():
        if isinstance(value, (list, np.ndarray)):
            del copy[name]
    return copy


def sigma_clip(y, sigma=5.0, return_mask=False, x=None):
    mask = np.abs(y - np.nanmedian(y)) < sigma * np.nanstd(y)

    if return_mask:
        return mask

    else:
        if x is None:
            return y[mask]
        else:
            return x[mask], y[mask]


def args_kwargs(f):
    s = inspect.signature(f)
    args = []
    kwargs = {}
    for p in s.parameters.values():
        if p.default != inspect._empty:
            kwargs[p.name] = p.default
        else:
            args.append(p.name)
    return args, kwargs


# todo: adapt to work with positional parameters like register
def register_args(f):
    """
    When used within a class, saves args and kwargs passed to a function
    (mostly used to record __init__ inputs)
    """

    @wraps(f)
    def inner(*_args, **_kwargs):
        self = _args[0]
        args, kwargs = args_kwargs(f)
        args = dict(zip(args[1::], _args[1::]))
        kwargs.update(_kwargs)
        self.args = args
        self.kwargs = kwargs
        return f(self, *args.values(), **kwargs)

    return inner


def nan_gaussian_filter(data, sigma=1.0, truncate=4.0):
    """https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python

    Parameters
    ----------
    U : _type_
        _description_
    sigma : _type_, optional
        _description_, by default 1.
    truncate : _type_, optional
        _description_, by default 4.
    """

    V = data.copy()
    V[np.isnan(data)] = 0
    VV = ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

    W = 0 * data.copy() + 1
    W[np.isnan(data)] = 0
    WW = ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)

    return VV / WW


def clean_header(header_dict):
    return {
        key: value
        for key, value in header_dict.items()
        if not isinstance(value, (list, tuple)) and key.isupper()
    }


def easy_median(images):
    # To avoid memory errors, we split the median computation in 50
    images = np.array(images)
    shape_divisors = divisors(images.shape[1])
    n = shape_divisors[np.argmin(np.abs(50 - shape_divisors))]
    return np.concatenate(
        [np.nanmedian(im, axis=0) for im in np.split(images, n, axis=1)]
    )


def image_in_xarray(image, xarr, name="stack", stars=False):
    xarr.attrs.update(header_to_cdf4_dict(image.header))
    xarr.attrs.update(
        dict(
            telescope=image.telescope.name,
            filter=image.header.get(image.telescope.keyword_filter, ""),
            exptime=image.header.get(image.telescope.keyword_exposure_time, ""),
            name=image.header.get(image.telescope.keyword_object, ""),
        )
    )

    if image.telescope.keyword_observation_date in image.header:
        date = image.header[image.telescope.keyword_observation_date]
    else:
        date = Time(image.header[image.telescope.keyword_jd], format="jd").datetime

    xarr.attrs.update(dict(date=format_iso_date(date).isoformat()))
    xarr.coords[name] = (("w", "h"), image.data)

    xarr = xarr.assign_coords(time=xarr.jd_utc)
    xarr = xarr.sortby("time")
    xarr.attrs["time_format"] = "jd_utc"

    if stars:
        xarr = xarr.assign_coords(stars=(("star", "n"), image.stars_coords))

    return xarr


def check_skycoord(skycoord):
    """
    Check that skycoord is either:
    - a list of int (interpreted as deg)
    - a str (interpreted as houranlgle, deg)
    - a SkyCoord object

    and return a SkyCoord object

    Parameters
    ----------
    skycoord : list, tuple or SkyCoord
        coordinate of the image center

    Raise
    -----
    Raise an error if skycoord cannot be interpreted

    """
    if isinstance(skycoord, (tuple, list)):
        if isinstance(skycoord[0], (float, int)):
            skycoord = SkyCoord(*skycoord, unit=(u.deg, u.deg))
        elif isinstance(skycoord[0], str):
            skycoord = SkyCoord(*skycoord, unit=("hourangle", "deg"))
        else:
            if not isinstance(skycoord, SkyCoord):
                assert "'skycoord' must be a list of int (interpreted as deg), str (interpreted as houranlgle, deg) or SkyCoord object"

    return skycoord


def _run_with_timeout(func, timeout):
    """Run *func* (blocking I/O) under a hard wall-clock *timeout* in seconds.

    astroquery's synchronous TAP calls have no timeout and hang indefinitely
    when the remote service stalls, blocking the whole pipeline. We run the
    call in a worker thread and abandon it if it overruns. ``socket`` default
    timeout is set so the abandoned thread's connection actually closes instead
    of lingering. Raises ``concurrent.futures.TimeoutError`` on overrun.
    """
    import concurrent.futures
    import socket

    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return executor.submit(func).result(timeout=timeout)
    finally:
        socket.setdefaulttimeout(previous_timeout)
        executor.shutdown(wait=False)


def gaia_query(center, fov, *args, limit=10000, circular=True, timeout=NETWORK_TIMEOUT_S):
    """
    https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
    """

    from astroquery.gaia import Gaia

    if isinstance(center, SkyCoord):
        ra = center.ra.to(u.deg).value
        dec = center.dec.to(u.deg).value
    elif isinstance(center, (tuple, list)):
        ra, dec = center
        if isinstance(ra, u.Quantity):
            ra = ra.to(u.deg).value
        if isinstance(dec, u.Quantity):
            dec = dec.to(u.deg).value

    if not isinstance(fov, u.Quantity):
        fov = fov * u.deg

    if fov.ndim == 1:
        ra_fov, dec_fov = fov.to(u.deg).value
    else:
        ra_fov = dec_fov = fov.to(u.deg).value

    radius = np.min([ra_fov, dec_fov]) / 2

    fields = ",".join(args) if isinstance(args, (tuple, list)) else args

    if circular:
        official_query = (
            f"select top {limit} {fields} from gaiadr2.gaia_source where "
            "1=CONTAINS("
            f"POINT('ICRS', {ra}, {dec}), "
            f"CIRCLE('ICRS',ra, dec, {radius}))"
            "order by phot_g_mean_mag"
        )
        vizier_query = (
            f"select top {limit} {fields} from \"I/345/gaia2\" where "
            "1=CONTAINS("
            f"POINT('ICRS', {ra}, {dec}), "
            f"CIRCLE('ICRS',ra, dec, {radius}))"
            "order by phot_g_mean_mag"
        )
    else:
        official_query = (
            f"select top {limit} {fields} from gaiadr2.gaia_source where "
            f"ra BETWEEN {ra-ra_fov/2} AND {ra+ra_fov/2} AND "
            f"dec BETWEEN {dec-dec_fov/2} AND {dec+dec_fov/2} "
            "order by phot_g_mean_mag"
        )
        vizier_query = (
            f"select top {limit} {fields} from \"I/345/gaia2\" where "
            f"ra BETWEEN {ra-ra_fov/2} AND {ra+ra_fov/2} AND "
            f"dec BETWEEN {dec-dec_fov/2} AND {dec+dec_fov/2} "
            "order by phot_g_mean_mag"
        )

    def _official():
        return Gaia.launch_job(official_query).get_results()

    def _vizier():
        from astroquery.utils.tap import TapPlus

        vizier_tap = TapPlus(url="https://tapvizier.u-strasbg.fr/TAPVizieR/tap")
        return vizier_tap.launch_job(vizier_query).get_results()

    try:
        return _run_with_timeout(_official, timeout)
    except Exception as e:
        import logging

        logging.getLogger("prose").warning(
            f"Official Gaia TAP query failed ({e}); trying VizieR mirror"
        )
        return _run_with_timeout(_vizier, timeout)


def full_class_name(o):
    # https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


def binn2D(arr, factor):
    new_shape = np.array(arr.shape) // factor
    shape = (new_shape[0], factor, new_shape[1], factor)
    return np.mean(arr.reshape(shape).mean(-1), 1)


def distance(p1, p2):
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))


def distances(coords, coord):
    return [
        np.sqrt(((coord[0] - x) ** 2 + (coord[1] - y) ** 2))
        for x, y in zip(coords[0].flatten(), coords[1].flatten())
    ]


def cross_match(S1, S2, tolerance=10, return_idxs=False, none=True):
    # cleaning
    s1 = S1.copy()
    s2 = S2.copy()

    s1[np.any(np.isnan(s1), 1)] = (1e15, 1e15)
    s2[np.any(np.isnan(s2), 1)] = (1e15, 1e15)

    # matching
    matches = []

    for i, s in enumerate(s1):
        distances = np.linalg.norm(s - s2, axis=1)
        closest = np.argmin(distances)
        if distances[closest] < tolerance:
            matches.append([i, closest])
        else:
            if none:
                matches.append([i, np.nan])

    matches = np.array(matches)
    matches = matches[np.all(~np.isnan(matches), 1)]
    matches = matches.astype(int)

    if return_idxs:
        return matches
    else:
        if len(matches) > 0:
            return s1[matches[:, 0]], s2[matches[:, 1]]
        else:
            return np.array([]), np.array([])


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments"""
    height = data.max()
    background = data.min()
    data = data - np.min(data)
    total = data.sum()
    x, y = np.indices(data.shape)
    x = (x * data).sum() / total
    y = (y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    width_x /= gaussian_sigma_to_fwhm
    width_y /= gaussian_sigma_to_fwhm
    return {
        "amplitude": height,
        "x": x,
        "y": y,
        "sigma_x": width_x,
        "sigma_y": width_y,
        "background": background,
        "theta": 0.0,
    }


def get_all_blocks():
    """Returns a list of all block names from prose (exposed in blocks.__init__.py)

    Returns
    -------
    list
        List of all block names
    """
    from prose import blocks

    blocks = [
        getattr(blocks, b)
        for b in dir(blocks)
        if isinstance(getattr(blocks, b), type)
        and issubclass(getattr(blocks, b), blocks.Block)
    ]

    return blocks


def binned_nanstd(x, bins: int = 12):
    # set binning idxs for white noise evaluation
    bins = np.min([x.shape[-1], bins])
    n = x.shape[-1] // bins
    idxs = np.arange(n * bins)

    def compute(f):
        return np.nanmean(
            np.nanstd(np.array(np.split(f.take(idxs, axis=-1), n, axis=-1)), axis=-1),
            axis=0,
        )

    return compute
