from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch


def _flatten_obs_tensor(obs) -> np.ndarray:
    """将 gymnasium 的 obs 展平为 1D numpy array。"""
    if isinstance(obs, torch.Tensor):
        return obs.detach().cpu().numpy().reshape(-1).astype(np.float32)
    if isinstance(obs, np.ndarray):
        return obs.reshape(-1).astype(np.float32)
    if isinstance(obs, dict):
        parts = []
        for k in sorted(obs.keys()):
            parts.append(_flatten_obs_tensor(obs[k]))
        return np.concatenate(parts).astype(np.float32)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[:, None]
    return arr.reshape(arr.shape[0], -1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-h5", required=True)
    p.add_argument("--output-npz", required=True)
    p.add_argument(
        "--replay",
        action="store_true",
        default=True,
        help="Replay demos via gym env using set_state_dict to collect obs_mode='state' observations (recommended).",
    )
    p.add_argument(
        "--no-replay",
        action="store_false",
        dest="replay",
        help="Use raw env_states from h5 (legacy behavior).",
    )
    args = p.parse_args()

    if args.replay:
        _convert_with_replay(args)
    else:
        _convert_legacy(args)


def _convert_with_replay(args) -> None:
    """
    通过 gym + set_state_dict 重放 demo，收集 obs_mode='state' 的标准观测。
    关键：用 env_states[0] 设置初始状态，确保与 h5 完全对齐。
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa
    from mani_skill.trajectory import utils as trajectory_utils

    print("Mode: gym replay with set_state_dict (obs_mode='state')  -- recommended")

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ep_starts: List[int] = []
    ep_lengths: List[int] = []
    cursor = 0
    n_skip = 0

    with h5py.File(args.input_h5, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda s: int(s.split("_")[1]),
        )

        # 创建一次 env（seed 无所谓，我们用 set_state_dict）
        env = gym.make(
            "StackCube-v1",
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            render_mode=None,
            max_episode_steps=600,
        )

        for tk in traj_keys:
            g = f[tk]
            if "actions" not in g or "env_states" not in g:
                n_skip += 1
                continue

            actions = g["actions"][()].astype(np.float32)
            T = actions.shape[0]

            # 获取 env_states 列表（T+1 帧：初始 + T 步后）
            env_states = trajectory_utils.dict_to_list_of_dicts(g["env_states"])
            if len(env_states) < T + 1:
                n_skip += 1
                continue

            obs_list: List[np.ndarray] = []

            # 用第一帧 env_state 初始化（而非 seed）
            obs_t, _ = env.reset(seed=0)  # 先 reset 激活环境
            env.unwrapped.set_state_dict(env_states[0])
            obs_t = env.unwrapped.get_obs()
            obs_list.append(_flatten_obs_tensor(obs_t))

            ok = True
            for t in range(T):
                obs_t, r, done, trunc, info = env.step(actions[t])
                if t < T - 1:
                    obs_list.append(_flatten_obs_tensor(obs_t))

            # obs_list 现在有 T 个元素（初始状态 + step[0..T-2] 后的状态）
            if len(obs_list) != T:
                n_skip += 1
                continue

            obs_arr = np.stack(obs_list, axis=0)  # (T, obs_dim)
            xs.append(obs_arr)
            ys.append(actions)
            ep_starts.append(cursor)
            ep_lengths.append(T)
            cursor += T

        env.close()

    if not xs:
        raise RuntimeError("No valid trajectories found.")

    X = np.concatenate(xs, axis=0).astype(np.float32)
    Y = np.concatenate(ys, axis=0).astype(np.float32)

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=X,
        acts=Y,
        ep_starts=np.asarray(ep_starts, dtype=np.int64),
        ep_lengths=np.asarray(ep_lengths, dtype=np.int64),
    )
    print(f"saved: {out}")
    print(
        f"samples={X.shape[0]}, obs_dim={X.shape[1]}, "
        f"act_dim={Y.shape[1]}, episodes={len(ep_starts)}, skipped={n_skip}"
    )


def _convert_legacy(args) -> None:
    """原始 env_states 转换逻辑（保留作 fallback）。"""
    from typing import Dict

    def _flatten_obs_group(group: h5py.Group) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}

        def rec(prefix: str, node):
            for k in node.keys():
                item = node[k]
                key = f"{prefix}/{k}" if prefix else k
                if isinstance(item, h5py.Dataset):
                    out[key] = item[()]
                else:
                    rec(key, item)

        rec("", group)
        return out

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ep_starts: List[int] = []
    ep_lengths: List[int] = []

    with h5py.File(args.input_h5, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda s: int(s.split("_")[1]),
        )

        cursor = 0
        for tk in traj_keys:
            g = f[tk]
            if "actions" not in g:
                continue
            actions = g["actions"][()].astype(np.float32)
            t = actions.shape[0]

            obs_flat = {}
            if "obs" in g:
                obs_flat = _flatten_obs_group(g["obs"])
            if not obs_flat and "env_states" in g:
                obs_flat = _flatten_obs_group(g["env_states"])
            if not obs_flat:
                continue

            parts = []
            for key in sorted(obs_flat.keys()):
                val = obs_flat[key]
                if val.shape[0] == t + 1:
                    val = val[:-1]
                elif val.shape[0] != t:
                    continue
                parts.append(_to_2d(val.astype(np.float32)))

            if not parts:
                continue

            obs_vec = np.concatenate(parts, axis=1)
            xs.append(obs_vec)
            ys.append(actions)
            ep_starts.append(cursor)
            ep_lengths.append(t)
            cursor += t

    if not xs:
        raise RuntimeError("No valid trajectories found in input h5.")

    X = np.concatenate(xs, axis=0).astype(np.float32)
    Y = np.concatenate(ys, axis=0).astype(np.float32)

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=X,
        acts=Y,
        ep_starts=np.asarray(ep_starts, dtype=np.int64),
        ep_lengths=np.asarray(ep_lengths, dtype=np.int64),
    )
    print(f"saved: {out}")
    print(
        f"samples={X.shape[0]}, obs_dim={X.shape[1]}, "
        f"act_dim={Y.shape[1]}, episodes={len(ep_starts)}"
    )


if __name__ == "__main__":
    main()
