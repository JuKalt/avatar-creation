"""Method execution registry and launch helpers."""

__all__ = ["MethodSpec", "discover_methods", "load_method_spec", "run_method"]


def __getattr__(name):
    """Load public helpers without importing all datasets in method runtimes."""
    if name in {"MethodSpec", "discover_methods", "load_method_spec"}:
        from . import registry

        return getattr(registry, name)
    if name == "run_method":
        from .runner import run_method

        return run_method
    raise AttributeError(name)
