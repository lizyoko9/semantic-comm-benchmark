"""Channel model factory."""

from channel.awgn import AWGNChannel
from channel.rayleigh import RayleighChannel
from channel.rician import RicianChannel

_CHANNELS = {
    "awgn": AWGNChannel,
    "rayleigh": RayleighChannel,
    "rician": RicianChannel,
}


def get_channel(name: str, **kwargs):
    """Create a channel model by name."""
    name = name.lower()
    if name not in _CHANNELS:
        raise ValueError(f"Unknown channel: {name}. Available: {list(_CHANNELS.keys())}")
    return _CHANNELS[name](**kwargs)
