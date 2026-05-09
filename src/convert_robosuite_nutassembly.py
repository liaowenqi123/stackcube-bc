from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def _flatten_obs_group(obs_group: h5py.Group, obs_keys: list[str]) -> np.ndarray:
    parts = []
    for key in obs_keys:
        if key not in obs_group:
            raise KeyError(f"obs key '{key}' not found in demo obs group")
        arr = np.asarray(obs_group[key], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        parts.append(arr.reshape(arr.shape[0], -1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def _flatten_obs_dict(obs_dict: dict, obs_keys: list[str]) -> np.ndarray:
    parts = []
    for key in obs_keys:
        if key not in obs_dict:
            raise KeyError(f"obs key '{key}' not found in runtime observation dict")
        parts.append(np.asarray(obs_dict[key], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32)


def _collect_obs_from_states(
    h5_file: h5py.File,
    demo_name: str,
    obs_keys: list[str],
    action_key: str,
    override_controller: str | None = None,
    skip_xml_replay: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    data_grp = h5_file["data"]
    demo = data_grp[demo_name]
    states = np.asarray(demo["states"], dtype=np.float64)
    acts = np.asarray(demo[action_key], dtype=np.float32)
    n = min(states.shape[0], acts.shape[0])
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    env_info = json.loads(data_grp.attrs["env_info"])
    if override_controller is not None:
        robot_name = env_info["robots"][0] if isinstance(env_info.get("robots"), list) else env_info.get("robots")
        env_info["controller_configs"] = load_composite_controller_config(
            controller=override_controller,
            robot=robot_name,
        )
    env = suite.make(
        **env_info,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
    )

    if not skip_xml_replay:
        model_xml = demo.attrs["model_file"]
        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
    else:
        env.reset()

    obs_rows = []
    act_rows = []
    for i in range(n):
        obs = env._get_observations()
        obs_rows.append(_flatten_obs_dict(obs, obs_keys))
        act_rows.append(acts[i].reshape(-1).astype(np.float32))
        env.step(acts[i])

    env.close()
    return np.asarray(obs_rows, dtype=np.float32), np.asarray(act_rows, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert robosuite hdf5 demos into NPZ(obs, acts)")
    p.add_argument("--input-h5", required=True, help="robosuite demo hdf5 path")
    p.add_argument("--output-npz", required=True, help="output npz path")
    p.add_argument(
        "--obs-keys",
        nargs="+",
        required=True,
        help="ordered observation keys to concatenate as state vector",
    )
    p.add_argument(
        "--action-key",
        default="actions",
        help="dataset key inside each demo group for actions (default: actions)",
    )
    p.add_argument(
        "--override-controller",
        default=None,
        help="optional composite controller name (e.g. BASIC) to override controller in env_info",
    )
    p.add_argument(
        "--skip-xml-replay",
        action="store_true",
        help="skip loading per-demo model xml / simulator state replay; directly roll out actions in a fresh env",
    )
    args = p.parse_args()

    in_path = Path(args.input_h5)
    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_obs = []
    all_acts = []

    with h5py.File(in_path, "r") as f:
        if "data" not in f:
            raise KeyError("expected top-level group 'data' in robosuite demo file")

        data_grp = f["data"]
        demo_names = sorted(list(data_grp.keys()))
        if not demo_names:
            raise RuntimeError("no demos found under /data")

        for demo_name in demo_names:
            demo = data_grp[demo_name]
            if args.action_key not in demo:
                raise KeyError(f"/data/{demo_name} missing action dataset '{args.action_key}'")
            if "obs" in demo:
                obs_mat = _flatten_obs_group(demo["obs"], args.obs_keys)
                acts = np.asarray(demo[args.action_key], dtype=np.float32)
                n = min(obs_mat.shape[0], acts.shape[0])
                if n <= 0:
                    continue
                all_obs.append(obs_mat[:n])
                all_acts.append(acts[:n])
            elif "states" in demo:
                obs_mat, acts = _collect_obs_from_states(
                    f,
                    demo_name,
                    args.obs_keys,
                    args.action_key,
                    override_controller=args.override_controller,
                    skip_xml_replay=args.skip_xml_replay,
                )
                if obs_mat.shape[0] <= 0:
                    continue
                all_obs.append(obs_mat)
                all_acts.append(acts)
            else:
                raise KeyError(f"/data/{demo_name} missing both 'obs' and 'states' datasets")

    if not all_obs:
        raise RuntimeError("no valid transitions collected from demos")

    obs = np.concatenate(all_obs, axis=0).astype(np.float32)
    acts = np.concatenate(all_acts, axis=0).astype(np.float32)

    np.savez_compressed(out_path, obs=obs, acts=acts)

    print(f"saved: {out_path}")
    print(f"obs shape: {obs.shape}, acts shape: {acts.shape}")


if __name__ == "__main__":
    main()
