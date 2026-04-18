# Video Task Compiler

Phase alpha bootstraps the contract layer for a video-to-robot task compiler.
The repository currently supports exactly one profile:

- Robot: `UR5e + Robotiq 2F-85`
- Task family: `pick_place`
- Capture mode: `moving monocular RGB video`
- Operator flow: `prompted_semi_automatic`
- Metric scale recovery: `fiducial`

Alpha does not perform video ingestion, reconstruction, MuJoCo scene generation,
learning, or ROS 2 deployment yet. It defines and validates the configuration
artifacts those later phases will depend on.

## What is included

- A Python package in `src/video_task_compiler/`
- A `vtc` CLI with `spec init`, `spec validate`, and `spec schema`
- Typed YAML models with cross-file compatibility validation
- A checked-in example bundle in `spec/`
- Tests covering the supported alpha profile and main failure modes

## Quick start

```bash
python -m pip install -e .[dev]
vtc spec validate --spec-dir spec
vtc spec schema --out-dir build/schemas
vtc spec init --template ur5e_monocular_pick_place --output-dir ./example-spec
```

## Spec files

The alpha contract is defined by three YAML files:

- `spec/robot.yaml`
- `spec/task.yaml`
- `spec/capture.yaml`

These files must be versioned together and validated as one bundle. File-local
validation alone is not sufficient because the task, robot, and capture
contracts share assumptions about embodiment, workspace, and capture geometry.

## Out of scope for alpha

- Video decoding and frame extraction
- Human pose tracking
- Object detection or segmentation
- 3D reconstruction or mesh generation
- MuJoCo MJCF compilation
- Imitation learning or RL training
- ROS 2 or hardware deployment

