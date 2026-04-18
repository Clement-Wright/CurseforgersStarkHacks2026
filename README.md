# Video Task Compiler

The repository currently supports exactly one roadmap-aligned monocular tabletop
profile:

- Robot: `UR5e + Robotiq 2F-85`
- Task family: `pick_place`
- Capture mode: `moving monocular RGB video`
- Operator flow: `prompted_semi_automatic`
- Metric scale recovery: `fiducial`

Alpha now freezes the deterministic project contract. Beta decodes raw monocular
video with PyAV, uses the static pre-roll as the source of truth for COLMAP
reconstruction, localizes later frames against that frozen model, normalizes the
result into a fiducial-aligned metric world frame, and preserves canonical
COLMAP artifacts by default.

## What is included

- A Python package in `src/video_task_compiler/`
- A `vtc` CLI with `spec init`, `spec validate`, and `spec schema`
- A `vtc video ingest-monocular` beta pipeline command
- Typed spec models with cross-file compatibility validation
- A checked-in example bundle in `spec/`, `env/`, and `checklists/`
- Tests covering the supported contract, validation rules, and beta pipeline logic

## Quick start

```bash
python -m pip install -e .[dev]
vtc spec validate --spec-dir spec
vtc spec schema --out-dir build/schemas
vtc spec init --template ur5e_monocular_pick_place --output-dir ./example-bundle
vtc video ingest-monocular --spec-dir spec --video ./demo.mp4 --out-dir ./artifacts/run01
```

## Alpha contract

The supported bundle is no longer just three YAML files. Beta and later phases
assume the full deterministic alpha contract exists:

- `spec/project.yaml`
- `spec/robot.yaml`
- `spec/task.yaml`
- `spec/capture.yaml`
- `spec/ontology.yaml`
- `spec/coordinate_frames.md`
- `checklists/readiness.md`
- `env/sfm.environment.yml`
- `env/human.environment.yml`
- `env/objects.environment.yml`

These files must be versioned together and validated as one bundle. The checked
in profile requires:

- `project.yaml` to own pre-roll, time, acceptance gates, and artifact owners
- `task.yaml` to reference ontology IDs instead of relying on free-form naming
- `coordinate_frames.md` to carry machine-checkable YAML front matter
- the env files to pin `python=3.10`

## Beta outputs

The monocular beta pipeline writes:

- `frames/index.csv`
- `calibration/intrinsics_opencv.yaml`
- `colmap/database.db`
- `colmap/sparse/`
- `colmap/sparse_txt/`
- `camera/intrinsics.json`
- `camera/camera_poses.json`
- `scene/sparse_points.ply`
- `reprojection_preview.jpg`
- `reconstruction_summary.json`

Compatibility copies are also emitted at the run root:

- `timestamps.csv`
- `camera_intrinsics.json`
- `camera_poses.json`

By design, beta reconstructs the sparse model from pre-roll keyframes only. The
rest of the clip is attached by direct localization where possible and is
otherwise left in the exported timeline as `registered=false` /
`pose_status=unlocalized`.

## Out of scope

- Human pose tracking
- Object detection or segmentation
- MuJoCo MJCF compilation
- Imitation learning or RL training
- ROS 2 or hardware deployment
