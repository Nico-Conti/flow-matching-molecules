from .fm_graph import FMGraph
from .defog import DeFoG

_REGISTRY = {cls.name: cls for cls in (FMGraph, DeFoG)}


def get_method(name):
    """Resolve a method name (e.g. "fm_graph") to a Method instance."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"unknown method {name!r}; available: {sorted(_REGISTRY)}")
