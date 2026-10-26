import copy
import hashlib
import os
import re
import shutil
from os import path
from pathlib import Path

import numpy as np
import requests
import yaml
from yaml import Loader

from prose.builtins import built_in_telescopes

info = print
package_name = "prose"


class _ConfigurationLoader(yaml.SafeLoader):
    """Read ordinary YAML and legacy tuples, rejecting ambiguous duplicate keys."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in keys:
                    raise ValueError(f"duplicate configuration key: {key!r}")
                keys.add(key)
            except TypeError as error:
                raise ValueError(
                    "configuration mapping keys must be hashable"
                ) from error
        return super().construct_mapping(node, deep=deep)


_ConfigurationLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node, deep=True)),
)


def _band_letter(band) -> str:
    """First letter of a filter name, lower-cased (``"z_s"`` -> ``"z"``)."""
    text = str(band or "").strip().strip("'\"").strip()
    return text[0].lower() if text[:1].isalpha() else ""


class ConfigManager:
    """
    A class for managing configuration settings for the package.

    Parameters
    ----------
    config_file : str or Path, optional
        A schema-1 YAML file containing settings and effective telescope aliases.
        Explicit files are always read-only. If omitted, PROSE_CONFIG_FILE is
        consulted; if neither is supplied, the existing ~/.prose behavior applies.
    expected_sha256 : str, optional
        Expected file checksum. PROSE_CONFIG_SHA256 is consulted only when the
        configuration source is selected from the environment.

    Attributes
    ----------
    config : dict
        A dictionary containing the current configuration settings.
    folder_path : Path
        A Path object representing the folder where the configuration files are stored.
    config_file : Path
        A Path object representing the configuration file.
    telescopes_dict : dict
        A dictionary containing the available telescopes.
    logs : list
        A list of log files.
    read_only : bool
        Whether the configuration was loaded from an explicit file.
    config_sha256 : str or None
        SHA-256 of the loaded file bytes, or None for legacy mutable configuration.

    Methods
    -------
    check_config_file(load=False)
        Checks if the configuration file exists and loads it if it does.
    save()
        Saves the current configuration settings to the configuration file.
    get(key)
        Returns the value of the specified configuration setting.
    """

    def __init__(self, config_file=None, *, expected_sha256=None):
        # Select the source before touching HOME, creating defaults or downloading
        # models. Explicit files are always read-only; no second mode flag is needed.
        if config_file is None:
            config_file = os.environ.get("PROSE_CONFIG_FILE")
            if expected_sha256 is None:
                expected_sha256 = os.environ.get("PROSE_CONFIG_SHA256")
        self.read_only = config_file is not None
        self.config_sha256 = None
        if self.read_only:
            if not str(config_file):
                raise ValueError("an explicit configuration file must not be empty")
            self.config_file = Path(config_file).expanduser().resolve()
            self.folder_path = self.config_file.parent
            self._load_readonly(expected_sha256)
            self.check_config_file(load=True)
            self.telescopes_dict = self.build_telescopes_dict()
            self.logs = []
            return
        if expected_sha256 is not None:
            raise ValueError("PROSE_CONFIG_SHA256 requires PROSE_CONFIG_FILE")
        self.config = None

        self.folder_path = Path.home() / f".{package_name}"
        self.folder_path.mkdir(exist_ok=True)

        self.config_file = self.folder_path / "config"

        self.rename_id_files_to_telescopes()
        self.create_builtins_telescopes_files()
        self.check_config_file(load=True)
        self.telescopes_dict = self.build_telescopes_dict()
        self.check_ballet()
        self.logs = []

    def _load_readonly(self, expected_sha256):
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
        ):
            raise ValueError(
                "expected configuration SHA-256 must contain 64 hex digits"
            )
        data = self.config_file.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise ValueError(f"configuration checksum mismatch: {self.config_file}")
        try:
            document = yaml.load(data, Loader=_ConfigurationLoader)
        except (yaml.YAMLError, ValueError) as error:
            raise ValueError(
                f"invalid configuration {self.config_file}: {error}"
            ) from error
        if (
            not isinstance(document, dict)
            or type(document.get("schema")) is not int
            or document["schema"] != 1
            or not isinstance(document.get("settings"), dict)
            or not isinstance(document.get("telescopes"), dict)
        ):
            raise ValueError(
                "configuration requires schema: 1 and settings/telescopes mappings"
            )
        telescopes = {}
        for alias, definition in document["telescopes"].items():
            if (
                not isinstance(alias, str)
                or not alias
                or not isinstance(definition, dict)
                or not isinstance(definition.get("name"), str)
                or not definition["name"].strip()
            ):
                raise ValueError(
                    "configuration requires named telescope mappings and text aliases"
                )
            key = alias.lower()
            if key in telescopes:
                raise ValueError(f"duplicate configuration telescope alias: {alias!r}")
            telescopes[key] = definition
        # Keep a private baseline: reads never re-open a path that might have
        # changed since startup, and public working dictionaries can be rebuilt.
        self._settings = copy.deepcopy(document["settings"])
        self._telescopes = copy.deepcopy(telescopes)
        self.config_sha256 = digest

    def _require_writable(self):
        if self.read_only:
            raise RuntimeError("prose configuration is read-only")

    def check_config_file(self, load=False):
        if self.read_only:
            if load:
                self.config = copy.deepcopy(self._settings)
            return
        if self.config_file.exists():
            with self.config_file.open(mode="r") as file:
                if load:
                    self.config = yaml.load(file.read(), Loader=Loader)
        else:
            info(f"A config file as been created in {self.folder_path}")
            self.config = {"color": "blue"}
            with self.config_file.open(mode="w") as file:
                yaml.dump(self.config, file, default_flow_style=False)

    def save(self):
        self._require_writable()
        self.check_config_file()
        with self.config_file.open(mode="w") as file:
            yaml.dump(self.config, file, default_flow_style=False)

    def get(self, key):
        return self.config.get(key)

    def set(self, key, value):
        self._require_writable()
        self.config[key] = value
        self.save()

    def rename_id_files_to_telescopes(self):
        """For backward compat with 0.9.6 downward"""
        self._require_writable()
        id_files = list(self.folder_path.glob("*.id"))
        if len(id_files) > 0:
            info("Renaming some old .id telescope files")
            for fpath in id_files:
                shutil.move(fpath, str(fpath).replace(".id", ".telescope"))

    def check_builtins_changes(self):
        if self.read_only:
            return
        for name, telescope in built_in_telescopes.items():
            telescope_file_name = path.join(self.folder_path, f"{name}.telescope")
            if path.exists(telescope_file_name):
                with open(telescope_file_name, mode="r") as f:
                    existing_telescope = yaml.load(f, Loader=yaml.FullLoader)
                    if existing_telescope != telescope:
                        info(
                            f"{Path(telescope_file_name).name} differs from builtins (use prose.CONFIG.update_builtins() to update)"
                        )

    def update_builtins(self):
        self._require_writable()
        self.create_builtins_telescopes_files(force=True)

    def build_telescopes_dict(self):
        if self.read_only:
            return copy.deepcopy(self._telescopes)
        telescope_files = self.folder_path.glob("*.telescope")

        telescope_dict = {}

        for telescope_file in telescope_files:
            with telescope_file.open(mode="r") as f:
                telescope = yaml.load(f, Loader=yaml.FullLoader)
            telescope_dict[telescope["name"].lower()] = telescope
            if "names" in telescope:
                for name in telescope["names"]:
                    telescope_dict[name.lower()] = telescope

        # telescope_dict.update(built_in_telescopes)

        return telescope_dict

    def create_builtins_telescopes_files(self, force=False):
        self._require_writable()
        for name, telescope in built_in_telescopes.items():
            telescope_file_name = path.join(self.folder_path, f"{name}.telescope")
            if (not path.exists(telescope_file_name) and not force) or force:
                self.save_telescope_file(telescope)

    def save_telescope_file(self, file):
        self._require_writable()
        if isinstance(file, str):
            name = Path(file).stem.lower()
            shutil.copyfile(file, self.folder_path / f"{name}.telescope")
            info("Telescope '{}' saved".format(name))
        elif isinstance(file, dict):
            name = file["name"].lower()
            telescope_file_path = self.folder_path / f"{name}.telescope"
            yaml.dump(file, telescope_file_path.open(mode="w"))
            info("Telescope '{}' saved".format(name))
        else:
            raise AssertionError("input type not understood")
        self.telescopes_dict = self.build_telescopes_dict()

    def match_telescope_name(self, name):
        available_telescopes_names = list(self.telescopes_dict.keys())
        has_telescope = np.flatnonzero(
            [t.lower() == name.lower() for t in available_telescopes_names]
        )
        if len(has_telescope) > 0:
            i = np.argmax(
                [
                    len(name)
                    for name in np.array(available_telescopes_names)[has_telescope]
                ]
            )
            return self.telescopes_dict[available_telescopes_names[has_telescope[i]]]
        else:
            return None

    def match_telescope_variant(self, name, band):
        """The telescope answering to *name* whose own name ends in *band*.

        Per-band telescope files (``MuSCAT2_g``, ``MuSCAT2_r``, ...) share the
        same aliases, so :meth:`match_telescope_name` returns whichever was
        registered last. Bands are compared by their first letter, which is
        what distinguishes MuSCAT-family channels (``z_s``, ``zs`` and ``z``
        are one channel). Returns ``None`` unless exactly one variant matches.
        """
        letter = _band_letter(band)
        if not letter or not name:
            return None
        target = name.lower()
        matches = {}
        for telescope in self.telescopes_dict.values():
            own = str(telescope.get("name", ""))
            aliases = {own.lower(), *(str(n).lower() for n in telescope.get("names") or ())}
            if target in aliases and "_" in own:
                if _band_letter(own.rsplit("_", 1)[1]) == letter:
                    matches[own] = telescope
        return next(iter(matches.values())) if len(matches) == 1 else None

    def check_ballet(self):
        self._require_writable()
        model_path = self.folder_path / "centroid.h5"

        if not model_path.exists():
            print("downloading ballet model (~30Mb)")
            model = requests.get(
                "https://github.com/lgrcia/ballet/raw/master/models/centroid.h5"
            ).content
            model_path.open(mode="wb").write(model)
