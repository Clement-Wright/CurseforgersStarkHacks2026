# Video Task Compiler

The repository currently supports exactly one roadmap-aligned monocular tabletop
profile:

- Robot: `UR5e + Robotiq 2F-85`
- Task family: `pick_place`
- Capture mode: `moving monocular RGB video`
- Operator flow: `prompted_semi_automatic`
- Metric scale recovery: `fiducial`

Alpha freezes the deterministic project contract. Beta decodes raw monocular
video with PyAV, uses the static pre-roll as the source of truth for COLMAP
reconstruction, localizes later frames against that frozen model, normalizes the
result into a fiducial-aligned metric world frame, and preserves canonical
COLMAP artifacts by default. Gamma runs 4DHumans over beta's extracted frames,
selects one primary demonstrator track, preserves the native tracklets, and
exports a flat arm-observables layer for downstream robotics code. Delta adds a
local Grounded-SAM-2 object layer that produces stable ontology-scoped masks,
tracks, interaction candidates, and review overlays.

Epsilon, Zeta, Eta, and Theta now exist as **experimental MVP back-half
phases**. They preserve honest proxy-scene provenance, emit compile-validated
MuJoCo asset packages, export a narrow arm+gripper replay trace, and compile a
canonical robot-in-scene MuJoCo task package, but they do **not** yet claim
research-grade dense static reconstruction, learning, or deployment.

## What Is Included

- A Python package in `src/video_task_compiler/`
- A `vtc` CLI with `spec init`, `spec validate`, and `spec schema`
- A `vtc video ingest-monocular` beta pipeline command
- A `vtc human extract-monocular` gamma pipeline command
- A `vtc objects extract-monocular` delta pipeline command
- Experimental `vtc epsilon compile-monocular`, `vtc zeta assetize-monocular`, and `vtc eta retarget-monocular` commands
- Experimental `vtc sim compile-task` command
- Typed spec models with cross-file compatibility validation
- A checked-in example bundle in `spec/`, `env/`, and `checklists/`
- Tests covering the supported contract, validation rules, and the current beta-through-eta MVP logic

## Quick Start

```bash
python -m pip install -e .[dev]
vtc spec validate --spec-dir spec
vtc spec schema --out-dir build/schemas
vtc spec init --template ur5e_monocular_pick_place --output-dir ./example-bundle
vtc video ingest-monocular --spec-dir spec --video ./demo.mp4 --out-dir ./artifacts/run01
vtc human extract-monocular --spec-dir spec --beta-dir ./artifacts/run01 --out-dir ./artifacts/run01 --fourdhumans-root /path/to/4DHumans --smpl-model /path/to/SMPL_NEUTRAL.pkl
vtc objects extract-monocular --spec-dir spec --beta-dir ./artifacts/run01 --out-dir ./artifacts/run01 --grounded-sam2-root /path/to/Grounded-SAM-2
vtc epsilon compile-monocular --spec-dir spec --beta-dir ./artifacts/run01 --delta-dir ./artifacts/run01 --out-dir ./artifacts/run01
vtc zeta assetize-monocular --spec-dir spec --epsilon-dir ./artifacts/run01 --out-dir ./artifacts/run01
vtc eta retarget-monocular --spec-dir spec --gamma-dir ./artifacts/run01 --delta-dir ./artifacts/run01 --epsilon-dir ./artifacts/run01 --zeta-dir ./artifacts/run01 --out-dir ./artifacts/run01
vtc sim compile-task --spec-dir spec --gamma-dir ./artifacts/run01 --epsilon-dir ./artifacts/run01 --zeta-dir ./artifacts/run01 --eta-dir ./artifacts/run01 --out-dir ./artifacts/run01
```

## Alpha Contract

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
- `env/scene.environment.yml`
- `env/sim.environment.yml`

These files must be versioned together and validated as one bundle. The checked
in profile requires:

- `project.yaml` to own pre-roll, time, acceptance gates, artifact owners, and phase modes
- `task.yaml` to reference ontology IDs instead of relying on free-form naming
- `coordinate_frames.md` to carry machine-checkable YAML front matter
- the env files to pin `python=3.10`

## Beta Outputs

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

## Gamma Outputs

The monocular gamma pipeline writes:

- `human/native/4dhumans_tracks.pkl`
- `human/smpl_tracks.pkl`
- `human/arm_observables.parquet`
- `human/reprojection_overlays/`
- `human/summary.json`

## Delta Outputs

The monocular delta pipeline writes:

- `objects/prompts.yaml`
- `objects/object_tracks.json`
- `objects/masks_rle.jsonl`
- `objects/interactions.parquet`
- `objects/overlays/`
- `objects/summary.json`

When `--keep-workdir` is used, the raw Grounded-SAM-2 staging area is also
preserved under `objects/native/`.

## Experimental Epsilon Outputs

The current epsilon MVP writes proxy-scene artifacts while preserving truthful
provenance:

- `scene/world_metric_from_world.json`
- `scene/support_plane.json`
- `scene/static_dense/fused.ply`
- `scene/static_dense/meshed-poisson.ply`
- `scene/static_dense/meshed-delaunay.ply`
- `scene/static_mesh.obj`
- `scene/static_mesh.meta.json`
- `scene/object_init_poses_metric.json`
- `camera/camera_poses_metric.json`
- `scene/epsilon_summary.json`

These paths are stable for later phases, but the checked-in profile marks them
as `proxy_scene` / `inherited_from_beta` until dense mode exists.

## Experimental Zeta Outputs

The current zeta MVP writes:

- `assets/manifest.json`
- `assets/*.meta.json`
- `assets/mujoco_assets.xml`
- `sim/assets_only.xml`
- `sim/assets_only.mjb`
- `sim/assets_smoke.json`
- `assets/zeta_summary.json`

Zeta currently assetizes conservative proxy geometry, but it requires a
successful MuJoCo compile and smoke test to claim simulation readiness.

## Experimental Eta Outputs

The current eta MVP writes:

- `retarget/robot_target.yaml`
- `retarget/robot_base_in_metric_world.json`
- `retarget/contact_schedule.json`
- `retarget/robot_demo.npz`
- `retarget/eta_summary.json`

Eta is intentionally narrow in the current profile: UR5e + 2F-85, task-window
pick/place replay, and the `pinocchio_seed` IK backend contract.

## Experimental Theta Outputs

Theta is now the canonical simulator compiler for the MVP. It writes:

- `sim/scene.xml`
- `sim/scene.mjb`
- `sim/task.json`
- `sim/validation.json`
- `sim/playback/robot_trace.npz`
- `sim/playback/human_arm_ghost.npz`

Theta consumes the Alpha contract plus Epsilon/Zeta/Eta artifacts and is the
first phase that owns the authoritative robot-in-scene MuJoCo task package.

## Out Of Scope

- Research-grade dense static reconstruction
- Geometry-backed visual assetization
- Imitation learning or RL training
- ROS 2 or hardware deployment
- Iota or Kappa execution
- DINO-X or Track-Anything rescue flows in the default delta path
