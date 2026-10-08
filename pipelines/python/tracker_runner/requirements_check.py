"""
Runtime dependency and capability checks.
"""

import importlib


def check_boxmot():
    try:
        module = importlib.import_module("boxmot")

        version = getattr(
            module,
            "__version__",
            "unknown",
        )

        return True, version

    except Exception as exc:
        return False, str(exc)


def check_module(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False