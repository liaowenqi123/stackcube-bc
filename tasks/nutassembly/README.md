# NutAssembly (robosuite) + Diffusion Policy V1

## 1) Install (inside `dp_robo`)

```powershell
pip install robosuite mujoco
pip install -r requirements.txt
```

## 2) Convert robosuite demos to training NPZ

```powershell
python src\convert_robosuite_nutassembly.py `
  --input-h5 .\data\raw\robosuite\nutassemblysingle_demo.hdf5 `
  --output-npz .\data\processed\nutassemblysingle_state.npz `
  --obs-keys robot0_eef_pos robot0_eef_quat robot0_gripper_qpos robot0_joint_pos_cos robot0_joint_pos_sin robot0_joint_vel object-state
```

## 3) Train V1

```powershell
powershell -ExecutionPolicy Bypass -File .\tasks\nutassembly\run_train_v1.ps1
```

## 4) Evaluate in robosuite

```powershell
powershell -ExecutionPolicy Bypass -File .\tasks\nutassembly\run_eval_v1.ps1
```

## Notes

- This setup is for single-arm `NutAssemblySingle` + `Panda`.
- If your hdf5 uses different observation keys, pass your own `--obs-keys`.
- Keep train / eval observation flattening exactly the same key order.
