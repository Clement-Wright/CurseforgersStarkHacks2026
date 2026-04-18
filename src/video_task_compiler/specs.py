from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

API_VERSION = "video-task-compiler/v1alpha1"
PROJECT_SCHEMA_VERSION = "0.1.0"
TEMPLATE_NAME = "ur5e_monocular_pick_place"


def _path_string(path: Path) -> str:
    return path.as_posix()


def _field_path(parts: tuple[Any, ...]) -> str:
    if not parts:
        return "<root>"
    return ".".join(str(part) for part in parts)


@dataclass(frozen=True)
class ValidationIssue:
    file_path: str
    field_path: str
    reason: str


class SpecValidationError(Exception):
    def __init__(self, issues: list[ValidationIssue]):
        super().__init__("spec validation failed")
        self.issues = issues


@dataclass(frozen=True)
class BundlePaths:
    root_dir: Path
    spec_dir: Path
    project_path: Path
    robot_path: Path
    task_path: Path
    capture_path: Path
    ontology_path: Path
    coordinate_frames_path: Path
    readiness_path: Path
    env_paths: dict[str, Path]


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

    ontology_id: str = Field(min_length=1)
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


class ProjectCaptureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_static_preroll: Literal[True]
    preroll_seconds: float = Field(gt=0.0)
    postroll_seconds: float = Field(ge=0.0, default=0.0)
    operator_count: int = Field(ge=1)


class ProjectTimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: Literal["first_decoded_frame"]
    base_unit: Literal["ns"]


class ProjectCoordFrameSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_world: str = Field(min_length=1)
    robot_base: str = Field(min_length=1)
    camera_optical: str = Field(min_length=1)


class ProjectOntologyRefsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_object_id: str = Field(min_length=1)
    receptacle_object_id: str = Field(min_length=1)
    support_surface_id: str = Field(min_length=1)


class ProjectAcceptanceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beta_min_registered_fraction: float = Field(gt=0.0, le=1.0)
    beta_min_registered_keyframes: int = Field(ge=1)
    beta_max_mean_reprojection_error_px: float = Field(gt=0.0)
    gamma_max_id_switches: int = Field(ge=0)
    delta_min_mask_iou_sample: float = Field(gt=0.0, le=1.0)


class ProjectSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[PROJECT_SCHEMA_VERSION]
    project_name: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    sample_raw_video: str = Field(min_length=1)
    task_name: str = Field(min_length=1)
    demo_owner: str = Field(min_length=1)
    artifact_owners: dict[str, str] = Field(min_length=1)
    robot_target_id: str = Field(min_length=1)
    capture: ProjectCaptureSpec
    coord_frames: ProjectCoordFrameSpec
    time: ProjectTimeSpec
    world_frame_policy: Literal["fiducial_center"]
    metric_scale_policy: Literal["fiducial_marker"]
    ontology: ProjectOntologyRefsSpec
    sponsor_resources: list[str] = Field(min_length=1)
    acceptance: ProjectAcceptanceSpec


class OntologyEntitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: Literal["object", "surface", "region"]
    label: str = Field(min_length=1)


class OntologySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[PROJECT_SCHEMA_VERSION]
    ontology_name: str = Field(min_length=1)
    entities: list[OntologyEntitySpec] = Field(min_length=1)


class CoordinateFrameDescriptorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    robot_frame: str | None = None
    convention: str | None = None
    policy: str | None = None


class CoordinateFramesFrontMatterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[PROJECT_SCHEMA_VERSION]
    frames: dict[str, CoordinateFrameDescriptorSpec] = Field(min_length=1)


class EnvironmentFileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    channels: list[str] = []
    dependencies: list[Any] = Field(min_length=1)


class SpecBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bundle_root: Path
    spec_dir: Path
    project_path: Path
    robot_path: Path
    task_path: Path
    capture_path: Path
    ontology_path: Path
    coordinate_frames_path: Path
    readiness_path: Path
    env_paths: dict[str, Path]
    project: ProjectSpec
    robot: RobotSpec
    task: TaskSpec
    capture: CaptureSpec
    ontology: OntologySpec
    coordinate_frames: CoordinateFramesFrontMatterSpec
    readiness_text: str
    env_specs: dict[str, EnvironmentFileSpec]


def resolve_bundle_paths(bundle_or_spec_dir: Path) -> BundlePaths:
    path = bundle_or_spec_dir.resolve()
    if (path / "robot.yaml").exists() and (path / "task.yaml").exists() and (path / "capture.yaml").exists():
        spec_dir = path
        root_dir = path.parent if path.name == "spec" else path
    elif (path / "spec" / "robot.yaml").exists() and (path / "spec" / "task.yaml").exists() and (path / "spec" / "capture.yaml").exists():
        root_dir = path
        spec_dir = path / "spec"
    else:
        raise SpecValidationError(
            [
                ValidationIssue(
                    _path_string(path),
                    "<root>",
                    "expected a bundle root containing spec/ or a spec directory containing robot.yaml, task.yaml, and capture.yaml",
                )
            ]
        )

    env_dir = root_dir / "env"
    return BundlePaths(
        root_dir=root_dir,
        spec_dir=spec_dir,
        project_path=spec_dir / "project.yaml",
        robot_path=spec_dir / "robot.yaml",
        task_path=spec_dir / "task.yaml",
        capture_path=spec_dir / "capture.yaml",
        ontology_path=spec_dir / "ontology.yaml",
        coordinate_frames_path=spec_dir / "coordinate_frames.md",
        readiness_path=root_dir / "checklists" / "readiness.md",
        env_paths={
            "sfm": env_dir / "sfm.environment.yml",
            "human": env_dir / "human.environment.yml",
            "objects": env_dir / "objects.environment.yml",
        },
    )


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


def _load_text_file(path: Path, required_keywords: list[str] | None = None) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SpecValidationError(
            [ValidationIssue(_path_string(path), "<root>", "missing required file")]
        ) from exc
    if not text.strip():
        raise SpecValidationError(
            [ValidationIssue(_path_string(path), "<root>", "expected a non-empty text file")]
        )
    if required_keywords:
        missing = [keyword for keyword in required_keywords if keyword not in text.lower()]
        if missing:
            raise SpecValidationError(
                [
                    ValidationIssue(
                        _path_string(path),
                        "<root>",
                        "missing required checklist content: " + ", ".join(missing),
                    )
                ]
            )
    return text


def _parse_markdown_front_matter(path: Path) -> CoordinateFramesFrontMatterSpec:
    text = _load_text_file(path)
    if not text.startswith("---\n"):
        raise SpecValidationError(
            [ValidationIssue(_path_string(path), "<root>", "expected YAML front matter at the start of the markdown file")]
        )
    closing_marker = text.find("\n---", 4)
    if closing_marker < 0:
        raise SpecValidationError(
            [ValidationIssue(_path_string(path), "<root>", "could not find the closing YAML front matter marker")]
        )
    front_matter_text = text[4:closing_marker]
    try:
        data = yaml.safe_load(front_matter_text)
    except yaml.YAMLError as exc:
        raise SpecValidationError(
            [ValidationIssue(_path_string(path), "<root>", f"invalid YAML front matter: {exc}")]
        ) from exc
    if not isinstance(data, dict):
        raise SpecValidationError(
            [ValidationIssue(_path_string(path), "<root>", "expected YAML front matter to be a mapping")]
        )
    try:
        return CoordinateFramesFrontMatterSpec.model_validate(data)
    except ValidationError as exc:
        issues = [
            ValidationIssue(_path_string(path), _field_path(err["loc"]), err["msg"])
            for err in exc.errors()
        ]
        raise SpecValidationError(issues) from exc


def _parse_environment_file(path: Path) -> EnvironmentFileSpec:
    try:
        return EnvironmentFileSpec.model_validate(_load_yaml_file(path))
    except ValidationError as exc:
        issues = [
            ValidationIssue(_path_string(path), _field_path(err["loc"]), err["msg"])
            for err in exc.errors()
        ]
        raise SpecValidationError(issues) from exc


def _compat_issue(path: Path, field_path: str, reason: str) -> ValidationIssue:
    return ValidationIssue(_path_string(path), field_path, reason)


def _env_pins_python_310(env_spec: EnvironmentFileSpec) -> bool:
    for dependency in env_spec.dependencies:
        if isinstance(dependency, str) and dependency.startswith("python="):
            return dependency == "python=3.10"
    return False


def load_bundle(bundle_or_spec_dir: Path) -> SpecBundle:
    paths = resolve_bundle_paths(bundle_or_spec_dir)

    project = _parse_model(paths.project_path, ProjectSpec)
    robot = _parse_model(paths.robot_path, RobotSpec)
    task = _parse_model(paths.task_path, TaskSpec)
    capture = _parse_model(paths.capture_path, CaptureSpec)
    ontology = _parse_model(paths.ontology_path, OntologySpec)
    coordinate_frames = _parse_markdown_front_matter(paths.coordinate_frames_path)
    readiness_text = _load_text_file(
        paths.readiness_path,
        required_keywords=["raw video", "owners", "environment", "preroll"],
    )
    env_specs = {
        env_name: _parse_environment_file(env_path)
        for env_name, env_path in paths.env_paths.items()
    }

    return SpecBundle(
        bundle_root=paths.root_dir,
        spec_dir=paths.spec_dir,
        project_path=paths.project_path,
        robot_path=paths.robot_path,
        task_path=paths.task_path,
        capture_path=paths.capture_path,
        ontology_path=paths.ontology_path,
        coordinate_frames_path=paths.coordinate_frames_path,
        readiness_path=paths.readiness_path,
        env_paths=paths.env_paths,
        project=project,
        robot=robot,
        task=task,
        capture=capture,
        ontology=ontology,
        coordinate_frames=coordinate_frames,
        readiness_text=readiness_text,
        env_specs=env_specs,
    )


def validate_bundle(bundle: SpecBundle) -> SpecBundle:
    issues: list[ValidationIssue] = []
    ontology_ids = [entity.id for entity in bundle.ontology.entities]
    ontology_lookup = {entity.id: entity for entity in bundle.ontology.entities}

    if len(ontology_ids) != len(set(ontology_ids)):
        issues.append(
            _compat_issue(bundle.ontology_path, "entities", "ontology entity ids must be unique")
        )

    if bundle.project.robot_target_id != bundle.robot.robot_id:
        issues.append(
            _compat_issue(
                bundle.project_path,
                "robot_target_id",
                f"project.robot_target_id '{bundle.project.robot_target_id}' must match robot.robot_id '{bundle.robot.robot_id}'",
            )
        )

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
            _compat_issue(bundle.capture_path, "modality", "alpha only supports rgb_monocular capture")
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
            _compat_issue(bundle.robot_path, "action_space.arm_dof", "alpha requires a 6-DoF arm action space")
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
            _compat_issue(bundle.robot_path, "gripper.type", "alpha requires a parallel jaw gripper")
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

    if bundle.project.ontology.target_object_id != bundle.task.pick_object.ontology_id:
        issues.append(
            _compat_issue(
                bundle.project_path,
                "ontology.target_object_id",
                "project ontology target object id must match task.pick_object.ontology_id",
            )
        )

    if bundle.project.ontology.receptacle_object_id != bundle.task.place_region.ontology_id:
        issues.append(
            _compat_issue(
                bundle.project_path,
                "ontology.receptacle_object_id",
                "project ontology receptacle id must match task.place_region.ontology_id",
            )
        )

    for field_path, entity_id, allowed_kinds in (
        ("ontology.target_object_id", bundle.project.ontology.target_object_id, {"object"}),
        ("ontology.receptacle_object_id", bundle.project.ontology.receptacle_object_id, {"object", "region"}),
        ("ontology.support_surface_id", bundle.project.ontology.support_surface_id, {"surface", "object", "region"}),
        ("pick_object.ontology_id", bundle.task.pick_object.ontology_id, {"object"}),
        ("place_region.ontology_id", bundle.task.place_region.ontology_id, {"object", "region"}),
    ):
        entity = ontology_lookup.get(entity_id)
        if entity is None:
            issues.append(
                _compat_issue(
                    bundle.ontology_path if field_path.startswith("ontology.") else bundle.task_path,
                    field_path,
                    f"ontology id '{entity_id}' was not defined in ontology.yaml",
                )
            )
            continue
        if entity.kind not in allowed_kinds:
            issues.append(
                _compat_issue(
                    bundle.ontology_path if field_path.startswith("ontology.") else bundle.task_path,
                    field_path,
                    f"ontology id '{entity_id}' must be one of kinds {sorted(allowed_kinds)}",
                )
            )

    required_frame_keys = {"project_world", "robot_base", "camera_optical", "fiducial_tag"}
    missing_frame_keys = required_frame_keys.difference(bundle.coordinate_frames.frames.keys())
    if missing_frame_keys:
        issues.append(
            _compat_issue(
                bundle.coordinate_frames_path,
                "frames",
                "coordinate_frames.md is missing required frame descriptors: " + ", ".join(sorted(missing_frame_keys)),
            )
        )
    else:
        project_world = bundle.coordinate_frames.frames["project_world"]
        robot_base = bundle.coordinate_frames.frames["robot_base"]
        camera_optical = bundle.coordinate_frames.frames["camera_optical"]

        if bundle.project.coord_frames.project_world != project_world.symbol:
            issues.append(
                _compat_issue(
                    bundle.project_path,
                    "coord_frames.project_world",
                    "project world frame symbol must match coordinate_frames.md front matter",
                )
            )
        if bundle.project.coord_frames.robot_base != robot_base.symbol:
            issues.append(
                _compat_issue(
                    bundle.project_path,
                    "coord_frames.robot_base",
                    "robot base frame symbol must match coordinate_frames.md front matter",
                )
            )
        if bundle.project.coord_frames.camera_optical != camera_optical.symbol:
            issues.append(
                _compat_issue(
                    bundle.project_path,
                    "coord_frames.camera_optical",
                    "camera optical frame symbol must match coordinate_frames.md front matter",
                )
            )
        if project_world.policy != bundle.project.world_frame_policy:
            issues.append(
                _compat_issue(
                    bundle.coordinate_frames_path,
                    "frames.project_world.policy",
                    "project world frame policy must match project.yaml world_frame_policy",
                )
            )
        if robot_base.robot_frame != bundle.robot.base_frame:
            issues.append(
                _compat_issue(
                    bundle.coordinate_frames_path,
                    "frames.robot_base.robot_frame",
                    "robot base frame mapping must match robot.base_frame",
                )
            )
        if camera_optical.convention != "opencv_optical":
            issues.append(
                _compat_issue(
                    bundle.coordinate_frames_path,
                    "frames.camera_optical.convention",
                    "camera optical frame convention must be opencv_optical",
                )
            )

    if bundle.project.metric_scale_policy != "fiducial_marker":
        issues.append(
            _compat_issue(
                bundle.project_path,
                "metric_scale_policy",
                "roadmap-aligned beta requires fiducial_marker metric scale policy",
            )
        )

    if bundle.project.world_frame_policy != bundle.capture.beta.normalization.world_frame:
        issues.append(
            _compat_issue(
                bundle.project_path,
                "world_frame_policy",
                "project world frame policy must match capture.beta.normalization.world_frame",
            )
        )

    for env_name, env_spec in bundle.env_specs.items():
        if not _env_pins_python_310(env_spec):
            issues.append(
                _compat_issue(
                    bundle.env_paths[env_name],
                    "dependencies",
                    "environment files must pin python=3.10",
                )
            )

    if issues:
        raise SpecValidationError(issues)

    return bundle


def emit_json_schemas() -> dict[str, dict[str, Any]]:
    return {
        "project.schema.json": ProjectSpec.model_json_schema(),
        "robot.schema.json": RobotSpec.model_json_schema(),
        "task.schema.json": TaskSpec.model_json_schema(),
        "capture.schema.json": CaptureSpec.model_json_schema(),
        "ontology.schema.json": OntologySpec.model_json_schema(),
        "coordinate_frames.frontmatter.schema.json": CoordinateFramesFrontMatterSpec.model_json_schema(),
        "environment.schema.json": EnvironmentFileSpec.model_json_schema(),
    }
