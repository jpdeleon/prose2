"""JAX/Flax implementation of the Ballet centroiding model."""

import numpy as np


def _jax_dependencies():
    try:
        from flax import linen as nn
        import jax.numpy as jnp
    except ImportError as error:
        raise ModuleNotFoundError(
            "Ballet requires JAX and Flax. Install prose with the `jax` extra."
        ) from error
    return nn, jnp


def _build_cnn():
    nn, jnp = _jax_dependencies()

    class CNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = x - jnp.min(x, axis=(1, 2, 3), keepdims=True)
            scale = jnp.max(x, axis=(1, 2, 3), keepdims=True)
            x = x / jnp.where(scale == 0, 1, scale)
            x = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(x))
            x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="SAME")
            x = nn.relu(nn.Conv(128, (3, 3), padding="SAME")(x))
            x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="SAME")
            x = nn.relu(nn.Conv(256, (3, 3), padding="SAME")(x))
            x = x.reshape((x.shape[0], -1))
            x = nn.sigmoid(nn.Dense(2048)(x))
            x = nn.sigmoid(nn.Dense(512)(x))
            return nn.Dense(2)(x)

    return CNN()


def load_weights_file(file):
    """Load Eloy's exported Ballet weights from an ``.npz`` file."""
    with np.load(file) as weights:
        layers = np.unique(
            [key.removesuffix("_bias").removesuffix("_kernel") for key in weights]
        )
        return {
            layer: {
                "kernel": weights[f"{layer}_kernel"],
                "bias": weights[f"{layer}_bias"],
            }
            for layer in layers
        }


def download_weights():
    """Download the current pretrained 15x15 Ballet weights."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ModuleNotFoundError(
            "Downloading Ballet weights requires `huggingface_hub`. "
            "Install prose with the `jax` extra or pass model_file."
        ) from error
    return hf_hub_download(repo_id="lgrcia/ballet", filename="centroid_15x15.npz")


class Ballet:
    """Pretrained Ballet CNN interface compatible with Eloy's implementation."""

    def __init__(self, model_file=None):
        if model_file is None:
            model_file = download_weights()
        self.cnn = _build_cnn()
        self.params = load_weights_file(model_file)

    def centroid(self, cutouts):
        """Return centroid positions in ``(x, y)`` cutout coordinates."""
        return np.asarray(
            self.cnn.apply({"params": self.params}, np.asarray(cutouts)[..., None])
        )[:, ::-1]
