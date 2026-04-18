---
schema_version: 0.1.0
frames:
  project_world:
    symbol: W
    meaning: Project world frame anchored to the fiducial marker center.
    policy: fiducial_center
  robot_base:
    symbol: Br
    meaning: UR5e base frame used by the compiled task package.
    robot_frame: base_link
  camera_optical:
    symbol: Ci
    meaning: Camera optical frame using the OpenCV convention.
    convention: opencv_optical
  fiducial_tag:
    symbol: Ft
    meaning: Fiducial tag frame with origin at the tag center, +X to the right edge, +Y to the top edge, and +Z out of the tag plane.
---

# Coordinate Frames

`W` is the authoritative project world frame for beta and later phases. It is
derived from the observed fiducial marker pose after COLMAP reconstruction and
normalization.

`Br` maps directly to the robot base frame named in `spec/robot.yaml`.

`Ci` is the optical camera frame used for image decoding, OpenCV intrinsics,
and exported camera poses.

`Ft` is the fiducial-local frame used to define metric scale and final world
alignment.
