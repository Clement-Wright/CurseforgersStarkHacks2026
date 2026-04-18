# Video Task Compiler

The repository currently supports exactly one monocular tabletop profile:

- Robot: `UR5e + Robotiq 2F-85`
- Task family: `pick_place`
- Capture mode: `moving monocular RGB video`
- Operator flow: `prompted_semi_automatic`
- Metric scale recovery: `fiducial`

Phase beta adds monocular video ingestion, timestamp export, COLMAP-based camera
calibration, fiducial normalization, and dense per-frame camera pose export.
The repository still does not perform scene understanding, MuJoCo scene
generation, learning, or ROS 2 deployment.

## What is included

- A Python package in `src/video_task_compiler/`
- A `vtc` CLI with `spec init`, `spec validate`, and `spec schema`
- A `vtc video ingest-monocular` beta pipeline command
- Typed YAML models with cross-file compatibility validation
- A checked-in example bundle in `spec/`
- Tests covering the supported profile, validation rules, and beta pipeline logic

## Quick start

```bash
python -m pip install -e .[dev]
vtc spec validate --spec-dir spec
vtc spec schema --out-dir build/schemas
vtc spec init --template ur5e_monocular_pick_place --output-dir ./example-spec
vtc video ingest-monocular --spec-dir spec --video ./demo.mp4 --out-dir ./artifacts/run01
```

## Spec files

The contract is still defined by three YAML files:

- `spec/robot.yaml`
- `spec/task.yaml`
- `spec/capture.yaml`

These files must be versioned together and validated as one bundle. The capture
contract now includes a required `beta` block for frame extraction, COLMAP, and
normalization settings.

## Beta outputs

The monocular beta pipeline writes:

- `frames/`
- `timestamps.csv`
- `camera_intrinsics.json`
- `camera_poses.json`
- `reprojection_preview.jpg`
- `reconstruction_summary.json`

## Out of scope

- Video decoding and frame extraction
- Human pose tracking
- Object detection or segmentation
- MuJoCo MJCF compilation
- Imitation learning or RL training
- ROS 2 or hardware deployment
