"""
Isaac Sim portable-root and registry path helpers for shared clusters.

On multi-user clusters the Isaac Sim conda install is typically read-only.
These helpers redirect writable paths (cache, logs, registry) to per-user
temp directories so OmniGibson doesn't crash on startup.

Configured via environment variables:
  IIIL_ISAAC_PORTABLE_ROOT  — writable root for Kit cache/config/logs
  OMNIGIBSON_APPDATA_PATH   — OmniGibson appdata (registry extensions, etc.)
"""

import os
import sys


def _ensure_omni_arg(prefix: str, value: str) -> None:
    arg = f"{prefix}={value}"
    if not any(existing == arg or existing.startswith(f"{prefix}=") for existing in sys.argv):
        sys.argv.append(arg)


def _build_omni_runtime_args() -> list[str]:
    args = []

    portable_root = os.environ.get("IIIL_ISAAC_PORTABLE_ROOT")
    if portable_root:
        portable_root = os.path.abspath(portable_root)
        os.makedirs(portable_root, exist_ok=True)
        omni_cache_root = os.path.join(portable_root, "omni_cache")
        omni_docs_root = os.path.join(portable_root, "Documents")
        omni_logs_root = os.path.join(portable_root, "omni_logs")
        omni_global_logs_root = os.path.join(portable_root, "omni_global_logs")
        omni_global_config_root = os.path.join(portable_root, "omni_global_config")
        for subdir in [
            "cache",
            "config",
            "data",
            os.path.join("documents", "Kit", "shared"),
            os.path.join("documents", "Kit", "shared", "screenshots"),
            "logs",
            os.path.join(omni_cache_root, "Kit"),
            os.path.join(omni_docs_root, "Kit", "shared"),
            os.path.join(omni_docs_root, "Kit", "shared", "screenshots"),
            omni_logs_root,
            omni_global_logs_root,
            omni_global_config_root,
        ]:
            os.makedirs(os.path.join(portable_root, subdir), exist_ok=True)
        args.extend(
            [
                f"--portable-root={portable_root}",
                f"--/app/tokens/omni_cache={omni_cache_root}",
                f"--/app/tokens/omni_documents={omni_docs_root}",
                f"--/app/tokens/omni_logs={omni_logs_root}",
                f"--/app/tokens/omni_global_logs={omni_global_logs_root}",
                f"--/app/tokens/omni_global_config={omni_global_config_root}",
            ]
        )

    appdata_root = os.environ.get("OMNIGIBSON_APPDATA_PATH")
    if appdata_root:
        registry_cache = os.environ.get(
            "IIIL_OMNI_REGISTRY_CACHE_PATH",
            os.path.join(appdata_root, "global", "data", "extsCache"),
        )
        registry_root = os.environ.get(
            "IIIL_OMNI_REGISTRY_CACHE_ROOT",
            os.path.join(appdata_root, "global", "data", "exts"),
        )
        registry_version = os.environ.get("IIIL_OMNI_REGISTRY_CACHE_VERSION", "v2")
        registry_full = os.environ.get(
            "IIIL_OMNI_REGISTRY_CACHE_FULL",
            os.path.join(registry_root, registry_version),
        )
        os.makedirs(registry_cache, exist_ok=True)
        os.makedirs(registry_root, exist_ok=True)
        os.makedirs(registry_full, exist_ok=True)

        args.extend(
            [
                f"--/exts/omni.kit.registry.nucleus/cachePath={registry_cache}",
                "--/exts/omni.kit.registry.nucleus/cacheVersionSubdir=",
                f"--/app/extensions/registryCacheRoot={registry_root}",
                f"--/app/extensions/registryCacheFull={registry_full}",
            ]
        )

    return args


def prepare_omni_registry_paths() -> None:
    """Inject portable/registry paths into sys.argv and monkey-patch SimulationApp."""
    runtime_args = _build_omni_runtime_args()
    for arg in runtime_args:
        prefix, value = arg.split("=", 1)
        _ensure_omni_arg(prefix, value)

    try:
        from isaacsim.simulation_app import SimulationApp
    except ImportError:
        return

    if getattr(SimulationApp, "_iiil_registry_patch", False):
        return

    original_init = SimulationApp.__init__

    def patched_init(self, launch_config=None, experience=""):
        cfg = dict(launch_config or {})
        extra_args = cfg.get("extra_args", [])
        if not isinstance(extra_args, list):
            extra_args = []
        else:
            extra_args = list(extra_args)

        for arg in runtime_args:
            prefix = arg.split("=", 1)[0]
            if not any(existing == arg or existing.startswith(f"{prefix}=") for existing in extra_args):
                extra_args.append(arg)

        cfg["extra_args"] = extra_args
        return original_init(self, cfg, experience)

    SimulationApp.__init__ = patched_init
    SimulationApp._iiil_registry_patch = True
