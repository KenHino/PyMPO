import importlib.metadata

_DISTRIBUTION_METADATA = importlib.metadata.metadata("pympo")
__version__ = _DISTRIBUTION_METADATA["Version"]
__version_info__ = tuple(map(int, __version__.split(".")))
