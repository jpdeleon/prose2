import os
import warnings

from astropy.wcs import FITSFixedWarning

warnings.simplefilter("ignore", FITSFixedWarning)

from prose import config

CONFIG = config.ConfigManager()
if CONFIG.read_only:
    # Spawned workers import prose afresh. Pin their file to the parent's loaded
    # bytes and use an absolute path even if their working directory changes.
    os.environ["PROSE_CONFIG_FILE"] = str(CONFIG.config_file)
    os.environ["PROSE_CONFIG_SHA256"] = CONFIG.config_sha256
CONFIG.check_builtins_changes()

from importlib.metadata import version

from prose import visualization as viz
from prose.core import Block, FITSImage, Image, Sequence, Sources
from prose.fluxes import Fluxes
from prose.io.fitsmanager import FitsManager
from prose.simulations import example_image
from prose.telescope import Telescope

__version__ = version("prose")

# TODO: update Telescope "names" fields
# TODO: document custom Image using _get_data_header
