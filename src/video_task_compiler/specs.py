from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

API_VERSION = "video-task-compiler/v1alpha1"
TEMPLATE_NAME = "ur5e_monocular_pick_place"


def _path_string(path: Path) -> str:
    return path.as_posix()


def _field_path(parts: tuple[Any, ...]) -> str:
    if not parts:
        return "<root>"
    rendered: list[str] = []
    for part in parts:
        if isinstance(part, int):
            rendered.append(str(part))
        else:
            rendered.append(str(part))
    return ".".join(rendered)


@dataclass(frozen=True)
class ValidationIssue:
    file_path: str
    field_path: str
    reason: str


class SpecValidationError(Exception):
    def __init__(self, issues: list[ValidationIssue]):
        super().__init__("spec validation failed")
        self.issues = issues


class Vec3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    z: float


class Range1D(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float
    max: float


class AABB(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_m: Vec3
    max_m: Vec3

    def contains(self, other: "AABB") -> bool:
        return (
            self.min_m.x <= other.min_m.x <= other.max_m.x <= self.max_m.x
            and self.min_m.y <= other.min_m.y <= other.max_m.y <= self.max_m.y
            and self.min_m.z <= other.min_m.z <= other.max_m.z <= self.max_m.z
        )


class GripperSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["parallel_jaw"]
    model: Literal["robotiq_2f85"]
    open_width_m: float = Field(ge=0.0)
    closed_width_m: float = Field(ge=0.0)


class ActionSpaceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm_dof: int = Field(ge=1)
    gripper_dof: int = Field(ge=1)
    control_mode: Literal["cartesian_delta_pose_plus_gripper"]
    arm_joint_order: list[str] = Field(min_length=6, max_length=6)
    gripper_width_range_m: Range1D


class RobotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: Literal[API_VERSION]
    robot_id: Literal["ur5e_robotiq_2f85"]
    model_ref: str = Field(min_length=1)
    base_frame: str = Field(min_length=1)
    ee_frame: str = Field(min_length=1)
    gripper: GripperSpec
    home_joint_positions_rad: tuple[float, float, float, float, float, float]
    workspace_bounds_m: AABB
    action_space: ActionSpaceSpec
    control_rate_hz: int = Field(gt=0)


class PromptedRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_prompt: str = Field(min_length=1)


class PickObjectSpec(PromptedRegion):
    search_region_m: AABB


class PlaceRegionSpec(PromptedRegion):
    target_region_m: AABB


class SuccessSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Literal["object_in_target_region"]
    position_tolerance_m: float = Field(gt=0.0)
    hold_time_s: float = Field(gt=0.0)


class OperatorAssistSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["prompted_semi_automatic"]
    allow_region_adjustment: bool
    allow_prompt_edit: bool


class AssistantSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["gemini"]


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: Literal[API_VERSION]
    task_id: str = Field(min_length=1)
    robot_id: str = Field(min_length=1)
    family: Literal["pick_place"]
    language_prompt: str = Field(min_length=1)
    pick_object: PickObjectSpec
    place_region: PlaceRegionSpec
    success: SuccessSpec
    operator_assist: OperatorAssistSpec
    assistant: AssistantSpec | None = None


class CameraDeviceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_name: Literal["handheld_phone", "mirrorless_camera", "action_camera"]
    intrinsics_mode: Literal["colmap_estimated", "precalibrated"]


class ResolutionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class MotionProtocolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capture_mode: Literal["handheld_orbit"]
    fixed_zoom: bool
    min_orbit_degrees: int = Field(ge=90, le=360)
    start_with_full_table_view: bool
    avoid_scene_cuts: bool


class FiducialSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["apriltag36h11", "aruco_4x4_50"]
    size_m: float = Field(gt=0.0)
    visible_at_start: bool
    visible_at_end: bool


class CaptureConstraintsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_mostly_static: bool
    single_demonstrator: bool
    object_visibility_ratio_min: float = Field(ge=0.0, le=1.0)
    lock_exposure_if_possible: bool


class FrameExtractionConfigSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_format: Literal["png"]
    keyframe_sample_fps: float = Field(gt=0.0)


class ColmapConfigSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_model: Literal["OPENCV"]
    matcher: Literal["sequential"]
    min_registered_keyframes: int = Field(ge=1)
    min_registered_ratio: float = Field(ge=0.0, le=1.0)
    max_mean_reprojection_error_px: float = Field(gt=0.0)
    max_interpolation_gap_s: float = Field(gt=0.0)


class NormalizationConfigSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_frame: Literal["fiducial_center"]


class BetaCaptureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_extraction: FrameExtractionConfigSpec
    colmap: ColmapConfigSpec
    normalization: NormalizationConfigSpec


class CaptureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: Literal[API_VERSION]
    capture_id: str = Field(min_length=1)
    modality: Literal["rgb_monocular"]
    camera_device: CameraDeviceSpec
    resolution: ResolutionSpec
    fps: int = Field(gt=0)
    motion_protocol: MotionProtocolSpec
    scale_source: Literal["fiducial"]
    fiducial: FiducialSpec
    constraints: CaptureConstraintsSpec
    beta: BetaCaptureSpec


class SpecBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec_dir: Path
    robot_path: Path
    task_path: Path
    capture_path: Path
    robot: RobotSpec
    task: TaskSpec
    capture: CaptureSpec


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecValidationError(
            [ValidationIssue(_path_string(path), "<root>", "missing required file")]
        ) from exc
    if not isinstance(data, dict):
        raise SpecValidationError(
            [ValidationIssue(_path_string(path), "<root>", "expected a YAML mapping")]
        )
    return data


def _parse_model(path: Path, model_type: type[BaseModel]) -> BaseModel:
    try:
        return model_type.model_validate(_load_yaml_file(path))
    except ValidationError as exc:
        issues = [
            ValidationIssue(_path_string(path), _field_path(err["loc"]), err["msg"])
            for err in exc.errors()
        ]
        raise SpecValidationError(issues) from exc


def load_bundle(spec_dir: Path) -> SpecBundle:
    spec_dir = spec_dir.resolve()
    robot_path = spec_dir / "robot.yaml"
    task_path = spec_dir / "task.yaml"
    capture_path = spec_dir / "capture.yaml"

    robot = _parse_model(robot_path, RobotSpec)
    task = _parse_model(task_path, TaskSpec)
    capture = _parse_model(capture_path, CaptureSpec)

    return SpecBundle(
        spec_dir=spec_dir,
        robot_path=robot_path,
        task_path=task_path,
        capture_path=capture_path,
        robot=robot,
        task=task,
        capture=capture,
    )


def _compat_issue(path: Path, field_path: str, reason: str) -> ValidationIssue:
    return ValidationIssue(_path_string(path), field_path, reason)


def validate_bundle(bundle: SpecBundle) -> SpecBundle:
    issues: list[ValidationIssue] = []

    if bundle.task.robot_id != bundle.robot.robot_id:
        issues.append(
            _compat_issue(
                bundle.task_path,
                "robot_id",
                f"task.robot_id '{bundle.task.robot_id}' must match robot.robot_id '{bundle.robot.robot_id}'",
            )
        )

    if bundle.task.family != "pick_place":
        issues.append(
            _compat_issue(bundle.task_path, "family", "alpha only supports the pick_place task family")
        )

    if bundle.capture.modality != "rgb_monocular":
        issues.append(
            _compat_issue(
                bundle.capture_path,
                "modality",
                "alpha only supports rgb_monocular capture",
            )
        )

    if bundle.capture.scale_source != "fiducial":
        issues.append(
            _compat_issue(
                bundle.capture_path,
                "scale_source",
                "monocular capture must use fiducial scale recovery",
            )
        )

    if bundle.robot.action_space.arm_dof != 6:
        issues.append(
            _compat_issue(
                bundle.robot_path,
                "action_space.arm_dof",
                "alpha requires a 6-DoF arm action space",
            )
        )

    if bundle.robot.action_space.gripper_dof != 1:
        issues.append(
            _compat_issue(
                bundle.robot_path,
                "action_space.gripper_dof",
                "alpha requires a single scalar gripper action",
            )
        )

    if bundle.robot.gripper.type != "parallel_jaw":
        issues.append(
            _compat_issue(
                bundle.robot_path,
                "gripper.type",
                "alpha requires a parallel jaw gripper",
            )
        )

    if not bundle.robot.workspace_bounds_m.contains(bundle.task.pick_object.search_region_m):
        issues.append(
            _compat_issue(
                bundle.task_path,
                "pick_object.search_region_m",
                "pick object search region must lie within robot.workspace_bounds_m",
            )
        )

    if not bundle.robot.workspace_bounds_m.contains(bundle.task.place_region.target_region_m):
        issues.append(
            _compat_issue(
                bundle.task_path,
                "place_region.target_region_m",
                "place region must lie within robot.workspace_bounds_m",
            )
        )

    if not bundle.capture.motion_protocol.fixed_zoom:
        issues.append(
            _compat_issue(
                bundle.capture_path,
                "motion_protocol.fixed_zoom",
                "alpha monocular capture requires a fixed zoom setting",
            )
        )

    if not bundle.capture.constraints.scene_mostly_static:
        issues.append(
            _compat_issue(
                bundle.capture_path,
                "constraints.scene_mostly_static",
                "alpha assumes a mostly static tabletop scene",
            )
        )

    if bundle.capture.beta.frame_extraction.image_format != "png":
        issues.append(
            _compat_issue(
                bundle.capture_path,
                "beta.frame_extraction.image_format",
                "beta frame extraction must emit png images",
            )
        )

    if bundle.capture.beta.colmap.camera_model != "OPENCV":
        issues.append(
            _compat_issue(
                bundle.capture_path,
                "beta.colmap.camera_model",
                "beta currently supports only the OPENCV COLMAP camera model",
            )
        )

    if bundle.capture.beta.colmap.matcher != "sequential":
        issues.append(
            _compat_issue(
                bundle.capture_path,
                "beta.colmap.matcher",
                "beta currently supports only the sequential COLMAP matcher",
            )
        )

    if bundle.capture.beta.normalization.world_frame != "fiducial_center":
        issues.append(
            _compat_issue(
                bundle.capture_path,
                "beta.normalization.world_frame",
                "beta normalization must use the fiducial_center world frame",
            )
        )

    if issues:
        raise SpecValidationError(issues)

    return bundle


def emit_json_schemas() -> dict[str, dict[str, Any]]:
    return {
        "robot.schema.json": RobotSpec.model_json_schema(),
        "task.schema.json": TaskSpec.model_json_schema(),
        "capture.schema.json": CaptureSpec.model_json_schema(),
    }
