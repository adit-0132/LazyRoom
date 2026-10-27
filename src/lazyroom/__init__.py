"""LazyRoom — an AI-native, agent-first image toolbox.

``from lazyroom import LazyRoom`` gives the application API / Python SDK.
"""
__version__ = "0.1.0"
__all__ = ["LazyRoom", "LazyRoomError", "__version__"]


def __getattr__(name):  # lazy: keep `import lazyroom.x` cheap
    if name == "LazyRoom":
        from .api import LazyRoom

        return LazyRoom
    if name == "LazyRoomError":
        from .core.errors import LazyRoomError

        return LazyRoomError
    raise AttributeError(name)
