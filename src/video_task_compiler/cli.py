from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer
from importlib.resources import files

from .specs import (
    TEMPLATE_NAME,
    SpecValidationError,
    emit_json_schemas,
    load_bundle,
    validate_bundle,
)
from .epsilon_scene import EpsilonSceneError, compile_metric_scene
from .human_extract import HumanExtractError, extract_monocular_human_motion
from .iota_data import IotaDataError, build_state_dataset
from .iota_train import IotaTrainError, finetune_rl_scaffold, train_state_imitation
from .eta_retarget import EtaRetargetError, retarget_monocular_demonstration
from .kappa_deploy import KappaDeployError, export_ros2_workspace
from .object_extract import ObjectExtractError, extract_monocular_objects
from .theta_sim import ThetaSimError, compile_task_package
from .video_ingest import VideoIngestError, ingest_monocular_video
from .zeta_assets import ZetaAssetError, assetize_metric_scene

app = typer.Typer(help="Video Task Compiler utilities.")
spec_app = typer.Typer(help="Manage alpha spec bundles.")
video_app = typer.Typer(help="Run beta video ingest pipelines.")
human_app = typer.Typer(help="Run gamma human motion pipelines.")
objects_app = typer.Typer(help="Run delta object extraction pipelines.")
epsilon_app = typer.Typer(help="Run epsilon metric scene compilation pipelines.")
zeta_app = typer.Typer(help="Run zeta MuJoCo assetization pipelines.")
eta_app = typer.Typer(help="Run eta robot retargeting pipelines.")
sim_app = typer.Typer(help="Run theta simulator compilation pipelines.")
data_app = typer.Typer(help="Run iota dataset export pipelines.")
train_app = typer.Typer(help="Run iota training pipelines.")
deploy_app = typer.Typer(help="Run kappa deployment export pipelines.")
app.add_typer(spec_app, name="spec")
app.add_typer(video_app, name="video")
app.add_typer(human_app, name="human")
app.add_typer(objects_app, name="objects")
app.add_typer(epsilon_app, name="epsilon")
app.add_typer(zeta_app, name="zeta")
app.add_typer(eta_app, name="eta")
app.add_typer(sim_app, name="sim")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(deploy_app, name="deploy")


def _template_dir(template: str) -> Path:
    if template != TEMPLATE_NAME:
        raise typer.BadParameter(
            f"unsupported template '{template}', expected '{TEMPLATE_NAME}'"
        )
    return Path(str(files("video_task_compiler").joinpath("templates", template)))


def _print_issues(exc: SpecValidationError) -> None:
    typer.echo("spec validation failed:")
    for issue in exc.issues:
        typer.echo(f"- {issue.file_path} :: {issue.field_path} :: {issue.reason}")


@spec_app.command("init")
def spec_init(
    template: str = typer.Option(
        TEMPLATE_NAME,
        "--template",
        help="Spec bundle template to materialize.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Bundle root that will receive spec/, env/, and checklists/.",
    ),
) -> None:
    """Create a new alpha spec bundle from the built-in template."""
    template_dir = _template_dir(template)
    output_dir.mkdir(parents=True, exist_ok=True)

    template_files = [path for path in template_dir.rglob("*") if path.is_file()]
    for source in template_files:
        relative_path = source.relative_to(template_dir)
        destination = output_dir / relative_path
        if destination.exists():
            raise typer.BadParameter(f"refusing to overwrite existing file: {destination}")

    for source in template_files:
        relative_path = source.relative_to(template_dir)
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    typer.echo(f"initialized template '{template}' in {output_dir}")


@spec_app.command("validate")
def spec_validate(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Bundle root or spec/ directory for the contract bundle.",
    )
) -> None:
    """Validate a spec bundle and exit non-zero on failure."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc

    typer.echo(f"spec validation passed for {spec_dir}")


@spec_app.command("schema")
def spec_schema(
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory to write JSON Schemas into.",
    )
) -> None:
    """Emit JSON Schema files for editor support and future integrations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = emit_json_schemas()
    for name, schema in schemas.items():
        target = out_dir / name
        target.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    typer.echo(f"wrote {len(schemas)} schema files to {out_dir}")


@video_app.command("ingest-monocular")
def ingest_monocular(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Bundle root or spec/ directory for the validated contract bundle.",
    ),
    video: Path = typer.Option(
        ...,
        "--video",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Input monocular video file.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive beta ingest artifacts.",
    ),
    colmap_bin: Optional[str] = typer.Option(
        None,
        "--colmap-bin",
        help="Optional explicit path to the COLMAP executable.",
    ),
    keep_workdir: bool = typer.Option(
        False,
        "--keep-workdir",
        help="Keep raw COLMAP work artifacts under the output directory.",
    ),
) -> None:
    """Decode a monocular demo video, reconstruct pre-roll geometry, and export normalized poses."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
        if bundle.capture.modality != "rgb_monocular":
            raise VideoIngestError("beta ingest-monocular only supports rgb_monocular capture bundles")
        ingest_monocular_video(
            bundle=bundle,
            video_path=video,
            out_dir=out_dir,
            colmap_bin=colmap_bin,
            keep_workdir=keep_workdir,
        )
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc
    except VideoIngestError as exc:
        typer.echo(f"video ingest failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"video ingest completed successfully in {out_dir}")


@human_app.command("extract-monocular")
def extract_monocular(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Bundle root or spec/ directory for the validated contract bundle.",
    ),
    beta_dir: Path = typer.Option(
        ...,
        "--beta-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Beta artifact root that contains frames/ and camera/ outputs.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive gamma human/ artifacts.",
    ),
    fourdhumans_root: Optional[Path] = typer.Option(
        None,
        "--fourdhumans-root",
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Optional explicit path to a local 4DHumans checkout.",
    ),
    smpl_model: Optional[Path] = typer.Option(
        None,
        "--smpl-model",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Optional explicit path to the neutral SMPL model asset.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Execution device hint for 4DHumans: cpu or cuda.",
    ),
    task_start_frame: Optional[int] = typer.Option(
        None,
        "--task-start-frame",
        min=0,
        help="Optional inclusive task-window start frame index.",
    ),
    task_end_frame: Optional[int] = typer.Option(
        None,
        "--task-end-frame",
        min=0,
        help="Optional inclusive task-window end frame index.",
    ),
    keep_workdir: bool = typer.Option(
        False,
        "--keep-workdir",
        help="Keep raw gamma staging artifacts under the output directory.",
    ),
) -> None:
    """Extract a primary demonstrator track and arm observables from beta artifacts."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
        if bundle.capture.modality != "rgb_monocular":
            raise HumanExtractError("gamma extract-monocular only supports rgb_monocular capture bundles")
        extract_monocular_human_motion(
            bundle=bundle,
            beta_dir=beta_dir,
            out_dir=out_dir,
            fourdhumans_root=fourdhumans_root,
            smpl_model_path=smpl_model,
            device=device,
            keep_workdir=keep_workdir,
            task_start_frame=task_start_frame,
            task_end_frame=task_end_frame,
        )
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc
    except HumanExtractError as exc:
        typer.echo(f"human extract failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"human extract completed successfully in {out_dir}")


@objects_app.command("extract-monocular")
def extract_monocular_objects_cli(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Bundle root or spec/ directory for the validated contract bundle.",
    ),
    beta_dir: Path = typer.Option(
        ...,
        "--beta-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Beta artifact root that contains frames/ and camera/ outputs.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive delta objects/ artifacts.",
    ),
    gamma_dir: Optional[Path] = typer.Option(
        None,
        "--gamma-dir",
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Optional gamma artifact root. Defaults to --beta-dir when omitted.",
    ),
    grounded_sam2_root: Optional[Path] = typer.Option(
        None,
        "--grounded-sam2-root",
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Optional explicit path to a local Grounded-SAM-2 checkout.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Execution device hint for Grounded-SAM-2: cpu or cuda.",
    ),
    task_start_frame: Optional[int] = typer.Option(
        None,
        "--task-start-frame",
        min=0,
        help="Optional inclusive task-window start frame index.",
    ),
    task_end_frame: Optional[int] = typer.Option(
        None,
        "--task-end-frame",
        min=0,
        help="Optional inclusive task-window end frame index.",
    ),
    keep_workdir: bool = typer.Option(
        False,
        "--keep-workdir",
        help="Keep raw delta staging artifacts under the output directory.",
    ),
) -> None:
    """Extract ontology-scoped object masks, tracks, and interaction candidates from beta artifacts."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
        if bundle.capture.modality != "rgb_monocular":
            raise ObjectExtractError("delta extract-monocular only supports rgb_monocular capture bundles")
        extract_monocular_objects(
            bundle=bundle,
            beta_dir=beta_dir,
            out_dir=out_dir,
            grounded_sam2_root=grounded_sam2_root,
            gamma_dir=gamma_dir,
            device=device,
            keep_workdir=keep_workdir,
            task_start_frame=task_start_frame,
            task_end_frame=task_end_frame,
        )
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc
    except ObjectExtractError as exc:
        typer.echo(f"object extract failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"object extract completed successfully in {out_dir}")


@epsilon_app.command("compile-monocular")
def compile_monocular_scene_cli(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Bundle root or spec/ directory for the validated contract bundle.",
    ),
    beta_dir: Path = typer.Option(
        ...,
        "--beta-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Beta artifact root that contains frames/ and camera/ outputs.",
    ),
    delta_dir: Path = typer.Option(
        ...,
        "--delta-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Delta artifact root that contains objects/ outputs.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive epsilon scene/ and camera/ artifacts.",
    ),
) -> None:
    """Compile a metric scene layer from beta camera artifacts and delta object masks."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
        if bundle.capture.modality != "rgb_monocular":
            raise EpsilonSceneError("epsilon compile-monocular only supports rgb_monocular capture bundles")
        compile_metric_scene(
            bundle=bundle,
            beta_dir=beta_dir,
            delta_dir=delta_dir,
            out_dir=out_dir,
        )
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc
    except EpsilonSceneError as exc:
        typer.echo(f"epsilon compile failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"epsilon compile completed successfully in {out_dir}")


@zeta_app.command("assetize-monocular")
def assetize_monocular_scene_cli(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Bundle root or spec/ directory for the validated contract bundle.",
    ),
    epsilon_dir: Path = typer.Option(
        ...,
        "--epsilon-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Artifact root that already contains epsilon scene outputs.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive zeta assets/ artifacts.",
    ),
) -> None:
    """Convert epsilon geometry into MuJoCo-safe visual and collision assets."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
        if bundle.capture.modality != "rgb_monocular":
            raise ZetaAssetError("zeta assetize-monocular only supports rgb_monocular capture bundles")
        assetize_metric_scene(
            bundle=bundle,
            epsilon_dir=epsilon_dir,
            out_dir=out_dir,
        )
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc
    except ZetaAssetError as exc:
        typer.echo(f"zeta assetize failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"zeta assetize completed successfully in {out_dir}")


@eta_app.command("retarget-monocular")
def retarget_monocular_demo_cli(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Bundle root or spec/ directory for the validated contract bundle.",
    ),
    beta_dir: Optional[Path] = typer.Option(
        None,
        "--beta-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Optional Beta artifact root that contains camera/ anchoring outputs.",
    ),
    gamma_dir: Path = typer.Option(
        ...,
        "--gamma-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Gamma artifact root that contains human/ outputs.",
    ),
    delta_dir: Path = typer.Option(
        ...,
        "--delta-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Delta artifact root that contains objects/ outputs.",
    ),
    epsilon_dir: Path = typer.Option(
        ...,
        "--epsilon-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Artifact root that already contains epsilon scene outputs.",
    ),
    zeta_dir: Path = typer.Option(
        ...,
        "--zeta-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Artifact root that already contains zeta assets/ outputs.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive eta retarget/ artifacts.",
    ),
    task_start_frame: Optional[int] = typer.Option(
        None,
        "--task-start-frame",
        min=0,
        help="Optional inclusive task-window start frame index.",
    ),
    task_end_frame: Optional[int] = typer.Option(
        None,
        "--task-end-frame",
        min=0,
        help="Optional inclusive task-window end frame index.",
    ),
) -> None:
    """Retarget the human-and-object demo into a robot-feasible handoff package."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
        if bundle.capture.modality != "rgb_monocular":
            raise EtaRetargetError("eta retarget-monocular only supports rgb_monocular capture bundles")
        retarget_monocular_demonstration(
            bundle=bundle,
            beta_dir=beta_dir,
            gamma_dir=gamma_dir,
            delta_dir=delta_dir,
            epsilon_dir=epsilon_dir,
            zeta_dir=zeta_dir,
            out_dir=out_dir,
            task_start_frame=task_start_frame,
            task_end_frame=task_end_frame,
        )
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc
    except EtaRetargetError as exc:
        typer.echo(f"eta retarget failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"eta retarget completed successfully in {out_dir}")


@sim_app.command("compile-task")
def compile_task_cli(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Bundle root or spec/ directory for the validated contract bundle.",
    ),
    gamma_dir: Path = typer.Option(
        ...,
        "--gamma-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Gamma artifact root that contains human/ outputs.",
    ),
    epsilon_dir: Path = typer.Option(
        ...,
        "--epsilon-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Artifact root that already contains epsilon scene outputs.",
    ),
    zeta_dir: Path = typer.Option(
        ...,
        "--zeta-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Artifact root that already contains zeta assets/ outputs.",
    ),
    eta_dir: Path = typer.Option(
        ...,
        "--eta-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Artifact root that already contains eta retarget/ outputs.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive theta sim/ artifacts.",
    ),
) -> None:
    """Compile the canonical MuJoCo task package from Alpha-through-Eta artifacts."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
        if bundle.capture.modality != "rgb_monocular":
            raise ThetaSimError("sim compile-task only supports rgb_monocular capture bundles")
        compile_task_package(
            bundle=bundle,
            gamma_dir=gamma_dir,
            epsilon_dir=epsilon_dir,
            zeta_dir=zeta_dir,
            eta_dir=eta_dir,
            out_dir=out_dir,
        )
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc
    except ThetaSimError as exc:
        typer.echo(f"sim compile-task failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"sim compile-task completed successfully in {out_dir}")


@data_app.command("build-dataset")
def build_dataset_cli(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Bundle root or spec/ directory for the validated contract bundle.",
    ),
    theta_dir: Path = typer.Option(
        ...,
        "--theta-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Artifact root that already contains theta sim/ outputs.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive iota data/ artifacts.",
    ),
) -> None:
    """Export a state-first robomimic-style HDF5 dataset from Theta artifacts."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
        build_state_dataset(bundle=bundle, theta_dir=theta_dir, out_dir=out_dir)
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc
    except IotaDataError as exc:
        typer.echo(f"data build-dataset failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"data build-dataset completed successfully in {out_dir}")


@train_app.command("imitation")
def train_imitation_cli(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Bundle root or spec/ directory for the validated contract bundle.",
    ),
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Artifact root that already contains data/demos.hdf5.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive iota policies/ and eval/ artifacts.",
    ),
) -> None:
    """Train the lightweight state-only BC baseline from the exported Iota dataset."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
        train_state_imitation(bundle=bundle, dataset_dir=dataset_dir, out_dir=out_dir)
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc
    except IotaTrainError as exc:
        typer.echo(f"train imitation failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"train imitation completed successfully in {out_dir}")


@train_app.command("finetune-rl")
def finetune_rl_cli(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Bundle root or spec/ directory for the validated contract bundle.",
    ),
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Artifact root that already contains data/demos.hdf5.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive fine-tuned policy and eval artifacts.",
    ),
    imitation_policy: Optional[Path] = typer.Option(
        None,
        "--imitation-policy",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Optional explicit path to an existing BC policy artifact.",
    ),
) -> None:
    """Run the local RL fine-tuning scaffold on top of the BC baseline."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
        finetune_rl_scaffold(
            bundle=bundle,
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            imitation_policy_path=imitation_policy,
        )
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc
    except IotaTrainError as exc:
        typer.echo(f"train finetune-rl failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"train finetune-rl completed successfully in {out_dir}")


@deploy_app.command("export-ros2")
def export_ros2_cli(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Bundle root or spec/ directory for the validated contract bundle.",
    ),
    theta_dir: Path = typer.Option(
        ...,
        "--theta-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Artifact root that already contains theta sim/ outputs.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive ros2/ export artifacts.",
    ),
    policy_path: Optional[Path] = typer.Option(
        None,
        "--policy-path",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Optional path to a trained policy artifact to package with the ROS 2 export.",
    ),
) -> None:
    """Export a ROS 2 deployment workspace scaffold from the Theta task package."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
        export_ros2_workspace(
            bundle=bundle,
            theta_dir=theta_dir,
            out_dir=out_dir,
            policy_path=policy_path,
        )
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc
    except KappaDeployError as exc:
        typer.echo(f"deploy export-ros2 failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"deploy export-ros2 completed successfully in {out_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
