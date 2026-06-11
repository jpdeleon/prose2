# prose

<p align="center" style="margin-bottom:-50px">
    <img src="docs/_static/prose3.png" width="450">
</p>

<p align="center">
  Modular image processing pipelines for Astronomy
  <br>
  <p align="center">
    <a href="https://github.com/lgrcia/prose"><img src="https://img.shields.io/badge/github-lgrcia/prose-03A487.svg?style=flat" alt="github"/></a>
    <a href="https://github.com/lgrcia/prose/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat" alt="license"/></a>
    <a href="https://arxiv.org/abs/2111.02814"><img src="https://img.shields.io/badge/paper-B166A9.svg?style=flat" alt="paper"/></a>
    <a href="https://prose.readthedocs.io/en/latest"><img src="https://img.shields.io/badge/documentation-black.svg?style=flat" alt="documentation"/></a>
  </p>
</p>

 *prose* is a Python package to build modular image processing pipelines for Astronomy.

*powered by [astropy](https://www.astropy.org/) and [photutils](https://photutils.readthedocs.io)*!

## Example

Here is a quick example pipeline to characterize the point-spread-function (PSF) of an example image


```python
import matplotlib.pyplot as plt
from prose import Sequence, blocks
from prose.simulations import example_image

# getting the example image
image = example_image()

sequence = Sequence(
    [
        blocks.PointSourceDetection(),  # stars detection
        blocks.Cutouts(shape=21),  # cutouts extraction
        blocks.MedianEPSF(),  # PSF building
        blocks.Moffat2D(),  # PSF modeling
    ]
)

sequence.run(image)

# plotting
image.show()  # detected stars

# effective PSF parameters
image.epsf.params
```

While being run on a single image, a Sequence is designed to be run on list of images (paths) and provides the architecture to build powerful pipelines. For more details check [Quickstart](https://prose.readthedocs.io/en/latest/ipynb/quickstart.html) and [What is a pipeline?](https://prose.readthedocs.io/en/latest/ipynb/core.html)

## Example Datasets
* broadband griz: /data/MuSCAT4/250416, 250512
* narrowband griz: 

## End-to-end photometry script (LCO MuSCAT3/4)

`prose/scripts/run_photometry.py` runs the full multi-band reduction
demonstrated in `notebooks/prose_muscat34_template.ipynb` as a command-line
tool. Given a directory of calibrated (BANZAI-reduced) science frames for a
single target, it groups frames per band, builds per-band reference images,
identifies the target, sizes apertures from the Gaia nearest neighbour, runs
parallel aperture photometry, performs automatic differential photometry,
converts GJD-UTC to BJD-TDB, and writes per-band CSV/PNG/GIF products plus
multi-band `lightcurves`, `systematics`, `stacks` plots, an `.npz` archive,
and a timestamped log.

```shell
python -m prose.scripts.run_photometry \
    --target_name TOI-6715 \
    --data_dir /data/MuSCAT4/250416 \
    --results_dir ./TOI-6715_250416 \
    --bands gp rp ip zs --ref_band gp
```

By default (no `--ref_band`) each band self-references its own first frame,
which is the correct choice for MuSCAT3/4 where every band is a separate
camera. Pass `--ref_band gp` to instead align all bands to one band's frame.

Key options: `--ref_band`, `--refid` (reference-frame index per band),
`--gif_stride`,
`--no_gif`, `--test_run` (first 10 frames per band), and `--use_barycorrpy`
(otherwise BJD-TDB uses astropy light-travel-time). BJD conversion requires
`astroplan`.

### Custom aperture grid

By default aperture radii are sized from the Gaia nearest-neighbour
separation. To set an explicit grid (and skip the Gaia query entirely), use
`--aper_radii MIN,MAX,DR` together with `--annulus RIN,ROUT`. The grid is
**inclusive of MAX** (`10,20,2` → `[10, 12, 14, 16, 18, 20]`). `--aper_unit`
selects the unit for both flags:

```shell
# radii in pixels
python -m prose.scripts.run_photometry ... \
    --aper_radii 10,40,3 --annulus 44,52 --aper_unit pix

# radii in units of the per-image FWHM
python -m prose.scripts.run_photometry ... \
    --aper_radii 1,5,0.5 --annulus 6,8 --aper_unit fwhm
```

`--annulus` is required whenever `--aper_radii` is given, and `--annulus` /
`--aper_unit` only apply together with `--aper_radii`.

## Installation

### latest

*prose* is written for python 3 and can be installed from [pypi](https://pypi.org/project/prose/) with:

```shell
pip install prose
```

For the latest version

```shell
pip install 'prose @ git+https://github.com/lgrcia/prose'
```

## Contributions
See our [contributions guidelines](docs/CONTRIBUTING.md)

## Attribution

If you find `prose` useful for your research, cite [Garcia et. al 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.4817G). The BibTeX entry for the paper is:
```
@ARTICLE{prose,
       author = {{Garcia}, Lionel J. and {Timmermans}, Mathilde and {Pozuelos}, Francisco J. and {Ducrot}, Elsa and {Gillon}, Micha{\"e}l and {Delrez}, Laetitia and {Wells}, Robert D. and {Jehin}, Emmanu{\"e}l},
        title = "{PROSE: a PYTHON framework for modular astronomical images processing}",
      journal = {\mnras},
     keywords = {instrumentation: detectors, methods: data analysis, planetary systems, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Earth and Planetary Astrophysics},
         year = 2022,
        month = feb,
       volume = {509},
       number = {4},
        pages = {4817-4828},
          doi = {10.1093/mnras/stab3113},
archivePrefix = {arXiv},
       eprint = {2111.02814},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.4817G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

and read about how to cite the dependencies of your sequences [here](https://prose.readthedocs.io/en/latest/ipynb/acknowledgement.html).
