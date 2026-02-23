"""Model factory."""

from channel import get_channel
from models.image.deepjscc import DeepJSCC
from models.image.adjscc import ADJSCC
from models.image.ntscc import NTSCC
from models.image.witt import WITT
from models.text.deepsc import DeepSC


_IMAGE_MODELS = {
    "deepjscc": DeepJSCC,
    "adjscc": ADJSCC,
    "ntscc": NTSCC,
    "witt": WITT,
}

_TEXT_MODELS = {
    "deepsc": DeepSC,
}

_ALL_MODELS = {**_IMAGE_MODELS, **_TEXT_MODELS}


def get_model(name: str, channel_type: str = "awgn", **kwargs):
    """Create a model by name.

    Args:
        name: Model name (deepjscc, adjscc, ntscc, witt, deepsc).
        channel_type: Channel type for the model.
        **kwargs: Additional model-specific arguments.

    Returns:
        Instantiated model.
    """
    name = name.lower()
    if name not in _ALL_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(_ALL_MODELS.keys())}")

    channel_kwargs = {}
    if "k_factor_db" in kwargs and channel_type == "rician":
        channel_kwargs["k_factor_db"] = kwargs.pop("k_factor_db")
    channel = get_channel(channel_type, **channel_kwargs)

    return _ALL_MODELS[name](channel=channel, **kwargs)


def is_image_model(name: str) -> bool:
    return name.lower() in _IMAGE_MODELS


def is_text_model(name: str) -> bool:
    return name.lower() in _TEXT_MODELS
