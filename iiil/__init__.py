__all__ = ["IIILTask"]


def __getattr__(name):
    # Lazy so that lightweight submodules (gello, detectors, datas) can be imported
    # without pulling in OmniGibson / Isaac Sim.
    if name == "IIILTask":
        from .main import IIILTask

        return IIILTask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
