import gymnasium as gym
import numpy as np
import torch
import mani_skill.envs  # noqa: F401


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"torch cuda version: {torch.version.cuda}")

    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, terminated, truncated, _ = env.step(action)
    print(f"env step ok, terminated={terminated}, truncated={truncated}")
    env.close()
    print("runtime check passed")


if __name__ == "__main__":
    main()
