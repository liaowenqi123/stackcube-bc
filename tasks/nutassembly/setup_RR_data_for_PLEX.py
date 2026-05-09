import argparse
import glob
import os
import pathlib
from types import SimpleNamespace

from robomimic.scripts.dataset_states_to_obs import dataset_states_to_obs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert local NutAssemblyRound raw hdf5 demos into robomimic ph-format hdf5."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=r"tasks\nutassembly\data\robomimic\NutAssemblyRound\Panda\raw",
        help="Directory containing local raw *.hdf5 files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=r"tasks\nutassembly\data\robomimic\NutAssemblyRound\Panda\ph",
        help="Directory to place converted hdf5 files.",
    )
    parser.add_argument("--camera-height", type=int, default=84)
    parser.add_argument("--camera-width", type=int, default=84)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(glob.glob(str(input_dir / "*.hdf5")))
    if not raw_files:
        raise FileNotFoundError(f"No .hdf5 files found under: {input_dir}")

    print(f"Found {len(raw_files)} local raw files in {input_dir}")

    for idx, task_file in enumerate(raw_files):
        print(f"\n=== Converting {task_file} ===")
        out_name = f"NutAssemblyRound__{idx}.hdf5"

        convert_args = SimpleNamespace(
            dataset=task_file,
            output_name=out_name,
            done_mode=0,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            include_depth=False,
            shaped=None,
            n=None,
            copy_rewards=None,
            copy_dones=None,
        )

        dataset_states_to_obs(convert_args)

        generated_path = input_dir / out_name
        if not generated_path.exists():
            raise FileNotFoundError(f"Expected converted file not found: {generated_path}")

        os.replace(generated_path, output_dir / out_name)
        print(f"Saved: {output_dir / out_name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
