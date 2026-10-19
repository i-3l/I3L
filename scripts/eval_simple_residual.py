#!/usr/bin/env python3
"""Serve SimpleResidualPolicy via ResidualPolicyWrapper for evaluation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../il_lib"))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from il_lib.utils.config_utils import register_omegaconf_resolvers
from il_lib.utils.training_utils import load_state_dict, load_torch
from omegaconf import OmegaConf, ListConfig
from omnigibson.learning.utils.network_utils import WebsocketPolicyServer


def main():
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../il_lib/il_lib/configs"))

    import iiil as _iiil
    extra_search_paths = [f"file://{_iiil.__path__[0]}/configs"]
    try:
        import omnigibson as og
        extra_search_paths.append(f"file://{og.__path__[0]}/learning/configs")
    except ImportError:
        pass

    # Filter out non-Hydra args before composing config
    hydra_args = [o for o in sys.argv[1:] if not o.startswith("--")]
    overrides = hydra_args + [
        f"hydra.searchpath=[{','.join(extra_search_paths)}]",
        "hydra.output_subdir=null",
        "hydra.run.dir=.",
        "hydra/job_logging=none",
        "hydra/hydra_logging=none",
    ]

    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="base_config", overrides=overrides)
        register_omegaconf_resolvers()
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)

        # Forward robot= and task= to base_policy_overrides for inner compose()
        robot_override = "robot=a1_iiil"
        task_override = "task=pnp/base"
        for o in hydra_args:
            if o.startswith("robot="):
                robot_override = o
            elif o.startswith("task="):
                task_override = o

        raw = cfg.module.get("base_policy_overrides", None)
        existing = list(OmegaConf.to_container(raw, resolve=True)) if isinstance(raw, ListConfig) else (list(raw) if raw else [])
        if not any(o.startswith("robot=") for o in existing):
            existing.append(robot_override)
        if not any(o.startswith("task=") for o in existing):
            existing.append(task_override)
        cfg.module.base_policy_overrides = existing

        # Instantiate residual policy (loads base policy internally)
        policy = instantiate(cfg.module, _recursive_=False)

        # Load residual checkpoint (strict=False: no base policy weights in ckpt)
        ckpt = load_torch(cfg.ckpt_path, map_location="cpu")
        policy.load_state_dict(ckpt["state_dict"], strict=False)
        policy = policy.to("cuda")
        policy.eval()

        # Instantiate ResidualPolicyWrapper
        policy_wrapper = instantiate(cfg.policy_wrapper)
        policy_wrapper.policy = policy

        # Parse --port from CLI args
        port = 8000
        for o in sys.argv[1:]:
            if o.startswith("--port="):
                port = int(o.split("=")[1])

        # Parse --stats-path and --traces-path from CLI args
        stats_path = None
        traces_path = None
        for o in sys.argv[1:]:
            if o.startswith("--stats-path="):
                stats_path = o.split("=", 1)[1]
            elif o.startswith("--traces-path="):
                traces_path = o.split("=", 1)[1]

        server = WebsocketPolicyServer(
            policy=policy_wrapper,
            host="0.0.0.0",
            port=port,
        )
        print(f"Serving on 0.0.0.0:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            # Write inference stats on shutdown
            if stats_path and hasattr(policy_wrapper, "get_eval_stats"):
                import json
                stats = policy_wrapper.get_eval_stats()
                os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
                print(f"Wrote inference stats to {stats_path}")
            # Write per-episode action traces
            if traces_path and hasattr(policy_wrapper, "save_traces"):
                os.makedirs(os.path.dirname(traces_path) or ".", exist_ok=True)
                policy_wrapper.save_traces(traces_path)


if __name__ == "__main__":
    main()
