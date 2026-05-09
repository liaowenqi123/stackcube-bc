import gymnasium as gym
import mani_skill.envs

env = gym.make('StackCube-v1', obs_mode='state', control_mode='pd_joint_delta_pos', max_episode_steps=1)

# Get the observation specification
obs_spec = env.observation_space
print("Full obs spec:", obs_spec)
print()

# Reset and check the 48-dim observation
obs, info = env.reset()
print("Obs shape:", obs.shape)  # (1, 48)

# The first dim is batch=1, get the actual 48-dim vector
o = obs[0]
print("48-dim obs:")
for i in range(0, 48, 8):
    print(f"  [{i:2d}:{i+8:2d}] {[f'{x:.4f}' for x in o[i:i+8].tolist()]}")

# Check Joint names
robot = env.unwrapped.agent.robot
print("\nActive joints:")
for j in robot.get_active_joints():
    name = j.get_name()
    limit = j.get_limits()
    print(f"  {name}: limit={limit}")

# Check links
print("\nLinks:")
for link in robot.get_links():
    print(f"  {link.get_name()}")

env.close()
