# One authoritative, read-only configuration

Set `PROSE_CONFIG_FILE` **before the first import of prose** to load one reviewed
configuration instead of the mutable `~/.prose` directory:

```sh
PROSE_CONFIG_FILE=/srv/config/prose-v1.yaml python -m prose.scripts.run_photometry --help
```

All installations may use the same file. There are no independently maintained
per-environment directories. Selecting a file always means read-only mode; no
second flag is needed. With neither configuration environment variable set,
prose retains its existing `~/.prose` behavior.

## File format

```yaml
schema: 1
settings:
  color: blue
telescopes:
  camera: &camera
    name: Camera
    names: [Camera, Camera-Alias]
    gain: 1.8
    keyword_gain: GAIN
    read_noise: 6.0
    keyword_read_noise: RDNOISE
    trimming: !!python/tuple [0, 0]
  camera-alias: *camera
```

`settings` and `telescopes` must be mappings. `telescopes` is the **effective
alias map**: explicitly list each name used for lookup, including a telescope's
primary name and any aliases. The `names` field is preserved but does not create
additional lookup entries. Keys are matched case-insensitively. Duplicate YAML
keys and aliases that differ only in case are rejected rather than silently
choosing a winner. No builtin or user-directory definitions are merged in.

YAML anchors and legacy `!!python/tuple` values are supported. Other Python
object tags are not accepted. Telescope values are passed through to the
existing telescope/FITS code; this mode does not change header-keyword precedence
or scientific calculations. Optional top-level provenance fields may accompany
the three required fields; they do not override settings or telescope definitions.

To migrate an existing effective configuration, export it once from the intended
baseline environment, **without** `PROSE_CONFIG_FILE` or `PROSE_CONFIG_SHA256` set:

```python
from pathlib import Path
import yaml
from prose import CONFIG

# Refuse to overwrite an existing reviewed configuration.
with Path("prose-v1.yaml").open("x") as stream:
    yaml.dump({"schema": 1, "settings": CONFIG.config,
               "telescopes": CONFIG.telescopes_dict}, stream)
```

This export deliberately captures the baseline's resolved alias precedence.
Review the resulting settings, commit the file to version control, and retain
older versions needed to reproduce previous runs. Do not regenerate it at job
startup or edit a file while jobs using it are running.

## Startup, identity and worker processes

```python
import prose

print(prose.CONFIG.config_file)    # resolved absolute file path
print(prose.CONFIG.config_sha256)  # checksum of the bytes actually loaded
print(prose.CONFIG.read_only)      # True
```

An optional `PROSE_CONFIG_SHA256` checks the file against a known 64-digit hex
SHA-256 checksum before parsing. Supplying a checksum without a configuration
file is an error. A missing file, malformed document, unknown schema or checksum
mismatch fails without falling back to `~/.prose` or builtin definitions.

After successful import, prose normalizes `PROSE_CONFIG_FILE` to its absolute
path and sets `PROSE_CONFIG_SHA256` to the loaded checksum in the current
process's environment. Fresh Python workers inherit these values: even if their
working directory changes, they load the same file and reject changed bytes.
Forked workers inherit the parent's loaded configuration. Applications must
preserve these variables when constructing a custom child environment. Changing
the environment of a forkserver that was already started is not possible: select
the file and checksum before starting a forkserver, or start a new server with
the pinned environment. Changing
environment variables after importing prose does not replace an already loaded
`CONFIG`; use a new process to activate another configuration version.

`ConfigManager(config_file, expected_sha256=...)` also supports explicit loading
for standalone consumers. A directly constructed manager does not replace the
package-global `prose.CONFIG` or modify the worker environment, and an explicit
constructor path does not inherit a different file's environment checksum.

## Read-only contract

Startup does not read `~/.prose`, discover `.telescope` symlinks, create default
files or directories, migrate `.id` files, compare/update builtin files or
download the legacy centroid model. `save`, `set`, `save_telescope_file`,
`update_builtins`, `create_builtins_telescopes_files`,
`rename_id_files_to_telescopes` and `check_ballet` raise a read-only error before
mutation or filesystem/network side effects. A caller using `Telescope(save=True)`
must instead prepare an updated configuration outside the scientific run.

The read-only contract applies to configuration APIs and their filesystem side
effects, not recursively immutable working dictionaries or all scientific
pipeline I/O. Public dictionaries and `logs` remain usable by existing code.
`check_config_file(load=True)` restores settings and `build_telescopes_dict()`
returns telescope definitions using independent copies of the privately retained
startup snapshot, never a later reread of the file. Downloads by explicitly
requested optional processing blocks are outside
this configuration-loading contract.
