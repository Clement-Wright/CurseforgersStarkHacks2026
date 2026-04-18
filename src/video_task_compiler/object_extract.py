from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import pickle
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageDraw

from .specs import PROJECT_SCHEMA_VERSION, SpecBundle


class ObjectExtractError(Exception):
    """Raised when delta object extraction cannot complete successfully."""


class DependencyError(ObjectExtractError):
    """Raised when delta runtime dependencies are missing."""


GD1_LOCAL_MODEL_SCRIPT = "grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py"
SUPPORTED_ADAPTER_FILES = (
    "codex_vtc_grounded_sam2_adapter.py",
    "vtc_grounded_sam2_adapter.py",
)
SMPL_24_RIGHT_WRIST = 21
COCO_17_RIGHT_WRIST = 10


@dataclass(frozen=True)
class DeltaFrameRecord:
    frame_idx: int
    frame_name: str
    image_path: Path
    pts_sec: float
    t_ns: int
    segment: str
    pose_status: str


@dataclass(frozen=True)
class PromptSpec:
    ontology_id: str
    class_name: str
    kind: str
    label: str
    prompt_text: str
    task_reference_prompt: str | None


@dataclass(frozen=True)
class SeedFrameProbe:
    frame_idx: int
    frame_name: str
    segment: str
    detected_ontology_ids: tuple[str, ...]
    detection_scores: dict[str, float]

    @property
    def coverage_count(self) -> int:
        return len(self.detected_ontology_ids)

    @property
    def total_score(self) -> float:
        return float(sum(self.detection_scores.values()))


@dataclass(frozen=True)
class SeedFrameSelection:
    frame_idx: int
    frame_name: str
    image_path: Path
    segment: str
    candidate_pool: str
    coverage_count: int
    detected_ontology_ids: tuple[str, ...]
    detection_scores: dict[str, float]


@dataclass(frozen=True)
class WristObservation:
    frame_idx: int
    t_ns: int | None
    wrist_u: float | None
    wrist_v: float | None
    wrist_conf: float | None


def _path_string(path: Path) -> str:
    return path.resolve().as_posix()


def _mask_utils_module():
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise DependencyError(
            "Missing Python dependency for delta extract: pycocotools"
        ) from exc
    return mask_utils


def missing_delta_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name in ("numpy", "PIL", "pyarrow", "yaml", "pycocotools"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def _discover_grounded_sam2_adapter(root: Path) -> Path | None:
    if os.environ.get("GROUNDED_SAM2_ADAPTER"):
        adapter_path = Path(os.environ["GROUNDED_SAM2_ADAPTER"]).resolve()
        if adapter_path.exists() and adapter_path.is_file():
            return adapter_path
    for relative_name in SUPPORTED_ADAPTER_FILES:
        candidate = root / relative_name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _discover_grounded_sam2_entrypoint(root: Path) -> Path | None:
    candidate = root / GD1_LOCAL_MODEL_SCRIPT
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def resolve_grounded_sam2_root(explicit: Path | None) -> Path:
    candidate = explicit
    if candidate is None and os.environ.get("GROUNDED_SAM2_ROOT"):
        candidate = Path(os.environ["GROUNDED_SAM2_ROOT"])
    if candidate is None:
        raise DependencyError(
            "Grounded-SAM-2 checkout not found. Provide --grounded-sam2-root or set GROUNDED_SAM2_ROOT."
        )
    root = candidate.resolve()
    if not root.exists() or not root.is_dir():
        raise DependencyError(f"Grounded-SAM-2 checkout path does not exist: {root}")
    if _discover_grounded_sam2_entrypoint(root) is None and _discover_grounded_sam2_adapter(root) is None:
        raise DependencyError(
            f"Grounded-SAM-2 checkout must contain {GD1_LOCAL_MODEL_SCRIPT} or one of {SUPPORTED_ADAPTER_FILES}: {root}"
        )
    return root


def ensure_delta_dependencies(grounded_sam2_root: Path | None) -> Path:
    missing = missing_delta_dependencies()
    if missing:
        raise DependencyError(
            "Missing Python dependencies for delta extract: " + ", ".join(sorted(missing))
        )
    return resolve_grounded_sam2_root(grounded_sam2_root)


def run_command(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown subprocess failure"
        raise ObjectExtractError(f"command failed: {' '.join(command)} :: {detail}")


def load_frame_index_csv(path: Path) -> list[DeltaFrameRecord]:
    if not path.exists():
        raise ObjectExtractError(f"missing beta frame index: {path}")

    records: list[DeltaFrameRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
            records.append(
                DeltaFrameRecord(
                    frame_idx=int(row["frame_idx"]),
                    frame_name=row["frame_name"],
                    image_path=image_path,
                    pts_sec=float(row["pts_sec"]),
                    t_ns=int(row["t_ns"]),
                    segment=row["segment"],
                    pose_status=row["pose_status"],
                )
            )
    if not records:
        raise ObjectExtractError(f"beta frame index was empty: {path}")
    return records


def load_camera_pose_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ObjectExtractError(f"missing beta camera pose export: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ObjectExtractError("camera/camera_poses.json did not contain a 'frames' list")
    return payload


def build_object_prompt_specs(bundle: SpecBundle) -> list[PromptSpec]:
    ontology_lookup = {entity.id: entity for entity in bundle.ontology.entities}
    task_reference_prompts = {
        bundle.task.pick_object.ontology_id: bundle.task.pick_object.text_prompt,
        bundle.task.place_region.ontology_id: bundle.task.place_region.text_prompt,
    }
    ordered_ids = [
        bundle.project.ontology.target_object_id,
        bundle.project.ontology.receptacle_object_id,
        bundle.project.ontology.support_surface_id,
    ]
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ObjectExtractError("project ontology ids must be unique for delta extraction")

    prompts: list[PromptSpec] = []
    for ontology_id in ordered_ids:
        entity = ontology_lookup.get(ontology_id)
        if entity is None:
            raise ObjectExtractError(f"project ontology id '{ontology_id}' was not defined in ontology.yaml")
        prompts.append(
            PromptSpec(
                ontology_id=ontology_id,
                class_name=ontology_id,
                kind=entity.kind,
                label=entity.label,
                prompt_text=entity.label,
                task_reference_prompt=task_reference_prompts.get(ontology_id),
            )
        )
    return prompts


def sample_segment_frames(
    frame_records: Sequence[DeltaFrameRecord],
    *,
    segment: str,
    limit: int,
) -> list[DeltaFrameRecord]:
    if segment == "active":
        candidates = [record for record in frame_records if record.segment != "preroll"]
    else:
        candidates = [record for record in frame_records if record.segment == segment]
    if len(candidates) <= limit:
        return candidates
    sampled = np.linspace(0, len(candidates) - 1, limit)
    deduped: list[DeltaFrameRecord] = []
    seen_frame_indices: set[int] = set()
    for sample in sampled:
        record = candidates[int(round(float(sample)))]
        if record.frame_idx not in seen_frame_indices:
            deduped.append(record)
            seen_frame_indices.add(record.frame_idx)
    return deduped


def _python_executable() -> str:
    executable = shutil.which("python") or shutil.which("python3")
    if executable is None:
        raise ObjectExtractError("python executable not found on PATH for Grounded-SAM-2 invocation")
    return executable


def _invoke_grounded_sam2_adapter(
    grounded_sam2_root: Path,
    mode: str,
    request: dict[str, Any],
    work_dir: Path,
    device: str,
) -> dict[str, Any] | None:
    adapter_path = _discover_grounded_sam2_adapter(grounded_sam2_root)
    if adapter_path is None:
        return None

    python_executable = _python_executable()
    request_path = work_dir / f"{mode}_request.json"
    response_path = work_dir / f"{mode}_response.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    command = [
        python_executable,
        str(adapter_path),
        "--mode",
        mode,
        "--request",
        str(request_path),
        "--response",
        str(response_path),
        "--device",
        device,
    ]
    run_command(command, cwd=grounded_sam2_root)
    if not response_path.exists():
        raise ObjectExtractError(
            f"Grounded-SAM-2 adapter did not produce a response file for mode '{mode}': {response_path}"
        )
    return json.loads(response_path.read_text(encoding="utf-8"))


def probe_seed_frame_candidates(
    grounded_sam2_root: Path,
    frame_candidates: Sequence[DeltaFrameRecord],
    prompts: Sequence[PromptSpec],
    *,
    device: str,
    work_dir: Path,
) -> list[SeedFrameProbe]:
    probes: list[SeedFrameProbe] = []
    for record in frame_candidates:
        request = {
            "video_id": record.frame_name,
            "frame_idx": record.frame_idx,
            "image_path": record.image_path.as_posix(),
            "segment": record.segment,
            "prompts": [
                {
                    "ontology_id": prompt.ontology_id,
                    "class_name": prompt.class_name,
                    "prompt_text": prompt.prompt_text,
                    "task_reference_prompt": prompt.task_reference_prompt,
                }
                for prompt in prompts
            ],
        }
        response = _invoke_grounded_sam2_adapter(
            grounded_sam2_root=grounded_sam2_root,
            mode=f"probe_{record.frame_idx}",
            request=request,
            work_dir=work_dir,
            device=device,
        )
        detections = response.get("detections", []) if isinstance(response, dict) else []
        detected_ids: list[str] = []
        scores: dict[str, float] = {}
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            ontology_id = str(detection.get("ontology_id", "")).strip()
            if not ontology_id:
                continue
            score = float(detection.get("score", 0.0))
            if ontology_id not in scores or score > scores[ontology_id]:
                scores[ontology_id] = score
            if ontology_id not in detected_ids:
                detected_ids.append(ontology_id)
        probes.append(
            SeedFrameProbe(
                frame_idx=record.frame_idx,
                frame_name=record.frame_name,
                segment=record.segment,
                detected_ontology_ids=tuple(sorted(detected_ids)),
                detection_scores=scores,
            )
        )
    return probes


def _best_probe(
    probes: Sequence[SeedFrameProbe],
) -> SeedFrameProbe | None:
    if not probes:
        return None
    return sorted(
        probes,
        key=lambda probe: (probe.coverage_count, probe.total_score, -probe.frame_idx),
        reverse=True,
    )[0]


def choose_seed_frame(
    frame_records: Sequence[DeltaFrameRecord],
    prompts: Sequence[PromptSpec],
    preroll_probes: Sequence[SeedFrameProbe],
    active_probes: Sequence[SeedFrameProbe],
) -> SeedFrameSelection:
    records_by_idx = {record.frame_idx: record for record in frame_records}
    required_ids = {prompt.ontology_id for prompt in prompts}
    best_preroll = _best_probe(preroll_probes)
    best_active = _best_probe(active_probes)

    chosen_probe = best_preroll
    candidate_pool = "preroll"
    if best_preroll is None and best_active is not None:
        chosen_probe = best_active
        candidate_pool = "active"
    elif best_preroll is not None and best_preroll.coverage_count < len(required_ids):
        if best_active is not None and best_active.coverage_count > best_preroll.coverage_count:
            chosen_probe = best_active
            candidate_pool = "active"
    if chosen_probe is None:
        fallback_record = next((record for record in frame_records if record.segment == "preroll"), None)
        if fallback_record is None:
            fallback_record = next((record for record in frame_records if record.segment != "preroll"), None)
        if fallback_record is None:
            raise ObjectExtractError("could not choose a seed frame because the beta frame index was empty")
        return SeedFrameSelection(
            frame_idx=fallback_record.frame_idx,
            frame_name=fallback_record.frame_name,
            image_path=fallback_record.image_path,
            segment=fallback_record.segment,
            candidate_pool="fallback",
            coverage_count=0,
            detected_ontology_ids=tuple(),
            detection_scores={},
        )

    selected_record = records_by_idx.get(chosen_probe.frame_idx)
    if selected_record is None:
        raise ObjectExtractError(f"seed frame {chosen_probe.frame_idx} was missing from the beta frame index")
    return SeedFrameSelection(
        frame_idx=selected_record.frame_idx,
        frame_name=selected_record.frame_name,
        image_path=selected_record.image_path,
        segment=selected_record.segment,
        candidate_pool=candidate_pool,
        coverage_count=chosen_probe.coverage_count,
        detected_ontology_ids=chosen_probe.detected_ontology_ids,
        detection_scores=chosen_probe.detection_scores,
    )


def run_grounded_sam2_tracking(
    grounded_sam2_root: Path,
    frames_dir: Path,
    prompts: Sequence[PromptSpec],
    seed_frame: SeedFrameSelection,
    *,
    device: str,
    work_dir: Path,
) -> dict[str, Any]:
    response = _invoke_grounded_sam2_adapter(
        grounded_sam2_root=grounded_sam2_root,
        mode="track",
        request={
            "frames_dir": frames_dir.as_posix(),
            "seed_frame_idx": seed_frame.frame_idx,
            "seed_frame_name": seed_frame.frame_name,
            "prompts": [
                {
                    "ontology_id": prompt.ontology_id,
                    "class_name": prompt.class_name,
                    "kind": prompt.kind,
                    "label": prompt.label,
                    "prompt_text": prompt.prompt_text,
                    "task_reference_prompt": prompt.task_reference_prompt,
                }
                for prompt in prompts
            ],
        },
        work_dir=work_dir,
        device=device,
    )
    if response is None:
        entrypoint = _discover_grounded_sam2_entrypoint(grounded_sam2_root)
        if entrypoint is None:
            raise DependencyError(
                f"Grounded-SAM-2 checkout did not expose {GD1_LOCAL_MODEL_SCRIPT}: {grounded_sam2_root}"
            )
        raise DependencyError(
            "Grounded-SAM-2 checkout was found, but no VTC adapter script was available to normalize its outputs. "
            f"Add one of {SUPPORTED_ADAPTER_FILES} to {grounded_sam2_root} or set GROUNDED_SAM2_ADAPTER."
        )
    if not isinstance(response, dict) or "tracks" not in response:
        raise ObjectExtractError("Grounded-SAM-2 adapter response must contain a top-level 'tracks' field")
    return response


def _frame_size(frame_records: Sequence[DeltaFrameRecord]) -> tuple[int, int]:
    image = Image.open(frame_records[0].image_path)
    width, height = image.size
    image.close()
    return height, width


def _coerce_bbox(frame_payload: dict[str, Any]) -> list[float]:
    bbox = np.asarray(frame_payload.get("bbox_xyxy", frame_payload.get("bbox", [])), dtype=float).reshape(-1)
    if bbox.size == 4:
        return [float(value) for value in bbox]
    if "bbox_xywh" in frame_payload:
        x, y, w, h = np.asarray(frame_payload["bbox_xywh"], dtype=float).reshape(-1)[:4]
        return [float(x), float(y), float(x + w), float(y + h)]
    return [0.0, 0.0, 0.0, 0.0]


def decode_coco_rle_mask(rle_payload: dict[str, Any]) -> np.ndarray:
    mask_utils = _mask_utils_module()
    decoded = mask_utils.decode(
        {
            "size": list(rle_payload["size"]),
            "counts": rle_payload["counts"],
        }
    )
    return np.asarray(decoded, dtype=bool)


def encode_coco_rle_mask(mask: np.ndarray) -> dict[str, Any]:
    mask_utils = _mask_utils_module()
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": str(counts),
    }


def _bbox_to_mask(bbox_xyxy: Sequence[float], image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    mask = np.zeros((height, width), dtype=bool)
    x_min = int(max(0, math.floor(float(bbox_xyxy[0]))))
    y_min = int(max(0, math.floor(float(bbox_xyxy[1]))))
    x_max = int(min(width, math.ceil(float(bbox_xyxy[2]))))
    y_max = int(min(height, math.ceil(float(bbox_xyxy[3]))))
    if x_max > x_min and y_max > y_min:
        mask[y_min:y_max, x_min:x_max] = True
    return mask


def _coerce_mask(
    frame_payload: dict[str, Any],
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    for key in ("mask", "segmentation", "mask_rle"):
        if key not in frame_payload:
            continue
        mask_value = frame_payload[key]
        if isinstance(mask_value, dict) and "counts" in mask_value and "size" in mask_value:
            return decode_coco_rle_mask(mask_value)
        mask_array = np.asarray(mask_value)
        if mask_array.ndim == 2:
            return mask_array.astype(bool)
    bbox = _coerce_bbox(frame_payload)
    return _bbox_to_mask(bbox, image_size)


def _infer_ontology_id(raw_track: dict[str, Any], prompts: Sequence[PromptSpec]) -> str | None:
    if raw_track.get("ontology_id"):
        return str(raw_track["ontology_id"])
    candidate_name = str(raw_track.get("class_name", raw_track.get("label", ""))).strip().lower()
    for prompt in prompts:
        if candidate_name in {prompt.class_name.lower(), prompt.label.lower(), prompt.prompt_text.lower()}:
            return prompt.ontology_id
    return None


def _canonical_track_order(raw_tracks: Iterable[dict[str, Any]], prompts: Sequence[PromptSpec]) -> list[dict[str, Any]]:
    prompt_order = {prompt.ontology_id: index for index, prompt in enumerate(prompts)}

    def track_sort_key(track_payload: dict[str, Any]) -> tuple[Any, ...]:
        ontology_id = _infer_ontology_id(track_payload, prompts) or "zzz"
        frames = track_payload.get("frames", [])
        first_frame = min((int(frame.get("frame_idx", frame.get("frame_index", 10**9))) for frame in frames if isinstance(frame, dict)), default=10**9)
        first_bbox = _coerce_bbox(frames[0]) if frames else [0.0, 0.0, 0.0, 0.0]
        centroid = float(first_bbox[0] + first_bbox[2]) / 2.0
        return (prompt_order.get(ontology_id, 10**6), first_frame, centroid)

    return sorted((track for track in raw_tracks if isinstance(track, dict)), key=track_sort_key)


def normalize_grounded_sam2_tracks(
    raw_payload: dict[str, Any],
    frame_records: Sequence[DeltaFrameRecord],
    prompts: Sequence[PromptSpec],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, int], np.ndarray], bool]:
    prompt_lookup = {prompt.ontology_id: prompt for prompt in prompts}
    frame_lookup = {record.frame_idx: record for record in frame_records}
    image_size = _frame_size(frame_records)
    tracks_payload = raw_payload.get("tracks", [])
    ordered_tracks = _canonical_track_order(tracks_payload, prompts)

    masks_entries: list[dict[str, Any]] = []
    masks_by_track_frame: dict[tuple[str, int], np.ndarray] = {}
    normalized_tracks: list[dict[str, Any]] = []
    rle_decode_success = True
    inferred_counts: dict[str, int] = {}

    for raw_track in ordered_tracks:
        ontology_id = _infer_ontology_id(raw_track, prompts)
        if ontology_id is None or ontology_id not in prompt_lookup:
            continue
        prompt = prompt_lookup[ontology_id]
        track_id = str(raw_track.get("track_id", "")).strip()
        if not track_id:
            inferred_counts[ontology_id] = inferred_counts.get(ontology_id, 0) + 1
            track_id = f"{ontology_id}_{inferred_counts[ontology_id]:03d}"

        seen_frame_indices: set[int] = set()
        normalized_frames: list[dict[str, Any]] = []
        for frame_payload in sorted(raw_track.get("frames", []), key=lambda item: int(item.get("frame_idx", item.get("frame_index", -1)))):
            if not isinstance(frame_payload, dict):
                continue
            frame_idx = int(frame_payload.get("frame_idx", frame_payload.get("frame_index", -1)))
            if frame_idx in seen_frame_indices:
                raise ObjectExtractError(f"duplicate frame_idx {frame_idx} detected inside track '{track_id}'")
            seen_frame_indices.add(frame_idx)
            frame_record = frame_lookup.get(frame_idx)
            if frame_record is None:
                continue
            bbox_xyxy = _coerce_bbox(frame_payload)
            score = float(frame_payload.get("score", raw_track.get("score", 0.0)))
            mask = _coerce_mask(frame_payload, image_size=image_size)
            visible = bool(frame_payload.get("visible", bool(mask.any())))
            mask_rle_ref: str | None = None
            centroid_uv: list[float] | None = None
            if visible and mask.any():
                rle_payload = encode_coco_rle_mask(mask)
                decoded_mask = decode_coco_rle_mask(rle_payload)
                rle_decode_success = rle_decode_success and bool(np.array_equal(mask, decoded_mask))
                masks_entries.append(
                    {
                        "track_id": track_id,
                        "frame_idx": frame_idx,
                        "size": rle_payload["size"],
                        "counts": rle_payload["counts"],
                    }
                )
                mask_rle_ref = f"objects/masks_rle.jsonl:{len(masks_entries)}"
                masks_by_track_frame[(track_id, frame_idx)] = mask
                rows, cols = np.nonzero(mask)
                centroid_uv = [float(np.mean(cols)), float(np.mean(rows))]
            elif visible:
                centroid_uv = [
                    float((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0),
                    float((bbox_xyxy[1] + bbox_xyxy[3]) / 2.0),
                ]
            normalized_frames.append(
                {
                    "frame_idx": frame_idx,
                    "t_ns": frame_record.t_ns,
                    "bbox_xyxy": bbox_xyxy,
                    "score": score,
                    "mask_rle_ref": mask_rle_ref,
                    "centroid_uv": centroid_uv,
                    "visible": visible,
                }
            )
        if normalized_frames:
            normalized_tracks.append(
                {
                    "track_id": track_id,
                    "ontology_id": prompt.ontology_id,
                    "class_name": prompt.class_name,
                    "source": str(raw_payload.get("source", "Grounded-SAM-2")),
                    "frames": normalized_frames,
                }
            )

    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": raw_payload.get("video_id"),
        "ontology": [prompt.ontology_id for prompt in prompts],
        "tracks": normalized_tracks,
    }
    return payload, masks_entries, masks_by_track_frame, rle_decode_success


def _wrist_from_smpl_track_payload(track_payload: dict[str, Any]) -> dict[int, WristObservation]:
    observations: dict[int, WristObservation] = {}
    tracks = track_payload.get("tracks", {})
    primary_track: dict[str, Any] | None = None
    if isinstance(tracks, dict):
        for candidate in tracks.values():
            if isinstance(candidate, dict) and candidate.get("is_primary_demonstrator"):
                primary_track = candidate
                break
    if primary_track is None:
        return observations
    for frame in primary_track.get("frames", []):
        if not isinstance(frame, dict):
            continue
        joints2d = np.asarray(frame.get("joints2d_xyc", []), dtype=float)
        if joints2d.ndim != 2 or joints2d.shape[0] == 0:
            continue
        if joints2d.shape[0] >= 24:
            wrist_index = SMPL_24_RIGHT_WRIST
        elif joints2d.shape[0] >= 17:
            wrist_index = COCO_17_RIGHT_WRIST
        else:
            continue
        if joints2d.shape[0] <= wrist_index:
            continue
        wrist_u = float(joints2d[wrist_index, 0])
        wrist_v = float(joints2d[wrist_index, 1])
        wrist_conf = float(frame.get("visibility", {}).get("right_wrist", joints2d[wrist_index, 2] if joints2d.shape[1] > 2 else 1.0))
        frame_idx = int(frame["frame_idx"])
        observations[frame_idx] = WristObservation(
            frame_idx=frame_idx,
            t_ns=int(frame.get("t_ns")) if frame.get("t_ns") is not None else None,
            wrist_u=wrist_u,
            wrist_v=wrist_v,
            wrist_conf=wrist_conf,
        )
    return observations


def load_optional_gamma_wrist_observations(gamma_dir: Path | None) -> dict[int, WristObservation]:
    if gamma_dir is None:
        return {}
    human_root = gamma_dir / "human"
    parquet_path = human_root / "arm_observables.parquet"
    if parquet_path.exists():
        table = pq.read_table(parquet_path)
        rows = table.to_pylist()
        observations: dict[int, WristObservation] = {}
        for row in rows:
            frame_idx = int(row["frame_idx"])
            wrist_u = row.get("wrist_u", row.get("wrist_r_u"))
            wrist_v = row.get("wrist_v", row.get("wrist_r_v"))
            wrist_conf = row.get("wrist_conf")
            if wrist_u is not None and wrist_v is not None:
                observations[frame_idx] = WristObservation(
                    frame_idx=frame_idx,
                    t_ns=int(row["t_ns"]) if row.get("t_ns") is not None else None,
                    wrist_u=float(wrist_u),
                    wrist_v=float(wrist_v),
                    wrist_conf=float(wrist_conf) if wrist_conf is not None else None,
                )
        if observations:
            return observations
    smpl_path = human_root / "smpl_tracks.pkl"
    if smpl_path.exists():
        with smpl_path.open("rb") as handle:
            payload = pickle.load(handle)
        return _wrist_from_smpl_track_payload(payload if isinstance(payload, dict) else {})
    return {}


def distance_point_to_mask(point_u: float, point_v: float, mask: np.ndarray | None) -> float | None:
    if mask is None or not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    squared = np.square(cols.astype(float) - point_u) + np.square(rows.astype(float) - point_v)
    return float(math.sqrt(float(np.min(squared))))


def build_interaction_rows(
    bundle: SpecBundle,
    normalized_payload: dict[str, Any],
    masks_by_track_frame: dict[tuple[str, int], np.ndarray],
    wrist_observations: dict[int, WristObservation],
) -> list[dict[str, Any]]:
    kind_lookup = {entity.id: entity.kind for entity in bundle.ontology.entities}
    rows: list[dict[str, Any]] = []
    for track in normalized_payload.get("tracks", []):
        ontology_kind = kind_lookup.get(track["ontology_id"], "object")
        for frame in track["frames"]:
            centroid = frame.get("centroid_uv") or [None, None]
            wrist = wrist_observations.get(int(frame["frame_idx"]))
            wrist_u = wrist.wrist_u if wrist is not None else None
            wrist_v = wrist.wrist_v if wrist is not None else None
            wrist_conf = wrist.wrist_conf if wrist is not None else None
            distance_px = None
            likely_contact = False
            if (
                frame["visible"]
                and ontology_kind == "object"
                and wrist is not None
                and wrist.wrist_u is not None
                and wrist.wrist_v is not None
                and wrist.wrist_conf is not None
                and wrist.wrist_conf >= bundle.project.delta.interaction.wrist_confidence_floor
            ):
                distance_px = distance_point_to_mask(
                    wrist.wrist_u,
                    wrist.wrist_v,
                    masks_by_track_frame.get((track["track_id"], int(frame["frame_idx"]))),
                )
                likely_contact = bool(
                    distance_px is not None
                    and distance_px <= bundle.project.delta.interaction.contact_distance_px
                )
            rows.append(
                {
                    "video_id": bundle.project.video_id,
                    "frame_idx": int(frame["frame_idx"]),
                    "t_ns": int(frame["t_ns"]),
                    "track_id": track["track_id"],
                    "class_name": track["class_name"],
                    "centroid_u": centroid[0],
                    "centroid_v": centroid[1],
                    "visible": bool(frame["visible"]),
                    "wrist_u": wrist_u,
                    "wrist_v": wrist_v,
                    "wrist_conf": wrist_conf,
                    "pixel_distance_wrist_to_mask": distance_px,
                    "likely_contact_boolean": likely_contact,
                }
            )
    return rows


def sample_active_overlay_indices(
    frame_records: Sequence[DeltaFrameRecord],
    overlay_count: int,
) -> list[int]:
    active_indices = [record.frame_idx for record in frame_records if record.segment != "preroll"]
    if len(active_indices) <= overlay_count:
        return active_indices
    sampled = np.linspace(0, len(active_indices) - 1, overlay_count)
    deduped: list[int] = []
    for sample in sampled:
        frame_idx = active_indices[int(round(float(sample)))]
        if frame_idx not in deduped:
            deduped.append(frame_idx)
    return deduped


def _track_color(track_id: str) -> tuple[int, int, int]:
    seed = abs(hash(track_id))
    return (
        64 + (seed % 160),
        64 + ((seed // 7) % 160),
        64 + ((seed // 13) % 160),
    )


def _mask_boundary_points(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    core = padded[1:-1, 1:-1]
    neighbors = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = core & ~neighbors
    rows, cols = np.nonzero(boundary)
    return [(int(col), int(row)) for row, col in zip(rows, cols)]


def write_object_overlays(
    overlay_dir: Path,
    frame_records: Sequence[DeltaFrameRecord],
    normalized_payload: dict[str, Any],
    masks_by_track_frame: dict[tuple[str, int], np.ndarray],
    wrist_observations: dict[int, WristObservation],
    overlay_count: int,
) -> list[int]:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    records_by_idx = {record.frame_idx: record for record in frame_records}
    tracks_by_frame: dict[int, list[dict[str, Any]]] = {}
    for track in normalized_payload.get("tracks", []):
        for frame in track["frames"]:
            tracks_by_frame.setdefault(int(frame["frame_idx"]), []).append(
                {
                    "track_id": track["track_id"],
                    "class_name": track["class_name"],
                    **frame,
                }
            )

    sampled_indices = sample_active_overlay_indices(frame_records, overlay_count)
    for frame_idx in sampled_indices:
        frame_record = records_by_idx[frame_idx]
        image = Image.open(frame_record.image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for frame_payload in tracks_by_frame.get(frame_idx, []):
            color = _track_color(str(frame_payload["track_id"]))
            bbox = frame_payload["bbox_xyxy"]
            draw.rectangle(tuple(float(value) for value in bbox), outline=color, width=3)
            mask = masks_by_track_frame.get((str(frame_payload["track_id"]), frame_idx))
            if mask is not None and mask.any():
                draw.point(_mask_boundary_points(mask), fill=color)
            draw.text(
                (float(bbox[0]) + 4.0, max(0.0, float(bbox[1]) - 14.0)),
                f"{frame_payload['track_id']}:{frame_payload['class_name']}",
                fill=color,
            )
        wrist = wrist_observations.get(frame_idx)
        if wrist is not None and wrist.wrist_u is not None and wrist.wrist_v is not None:
            draw.ellipse(
                (
                    wrist.wrist_u - 5,
                    wrist.wrist_v - 5,
                    wrist.wrist_u + 5,
                    wrist.wrist_v + 5,
                ),
                outline=(255, 255, 255),
                width=2,
            )
            draw.text((wrist.wrist_u + 6, wrist.wrist_v + 6), "right_wrist", fill=(255, 255, 255))
        image.save(overlay_dir / f"{frame_record.frame_name}.jpg", quality=90)
        image.close()
    return sampled_indices


def _duplicate_id_count_for_review_sample(
    normalized_payload: dict[str, Any],
    review_frame_indices: Sequence[int],
) -> int:
    review_set = set(review_frame_indices)
    per_frame_class_counts: dict[tuple[int, str], int] = {}
    for track in normalized_payload.get("tracks", []):
        for frame in track["frames"]:
            frame_idx = int(frame["frame_idx"])
            if frame_idx not in review_set or not frame["visible"]:
                continue
            key = (frame_idx, track["class_name"])
            per_frame_class_counts[key] = per_frame_class_counts.get(key, 0) + 1
    return sum(max(0, count - 1) for count in per_frame_class_counts.values())


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=False) + "\n")


def write_prompts_yaml(
    path: Path,
    bundle: SpecBundle,
    prompts: Sequence[PromptSpec],
    seed_frame: SeedFrameSelection,
) -> None:
    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "primary_backbone": bundle.project.delta.primary_backbone,
        "annotation_frame_policy": bundle.project.delta.annotation_frame_policy,
        "selected_seed_frame": {
            "frame_idx": seed_frame.frame_idx,
            "frame_name": seed_frame.frame_name,
            "image_path": seed_frame.image_path.as_posix(),
            "segment": seed_frame.segment,
            "candidate_pool": seed_frame.candidate_pool,
            "coverage_count": seed_frame.coverage_count,
            "detected_ontology_ids": list(seed_frame.detected_ontology_ids),
            "detection_scores": seed_frame.detection_scores,
        },
        "prompts": [
            {
                "ontology_id": prompt.ontology_id,
                "class_name": prompt.class_name,
                "kind": prompt.kind,
                "label": prompt.label,
                "prompt_text": prompt.prompt_text,
                "task_reference_prompt": prompt.task_reference_prompt,
            }
            for prompt in prompts
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_interactions_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path)


def build_delta_summary(
    bundle: SpecBundle,
    prompts: Sequence[PromptSpec],
    seed_frame: SeedFrameSelection,
    normalized_payload: dict[str, Any],
    masks_entries: Sequence[dict[str, Any]],
    interaction_rows: Sequence[dict[str, Any]],
    review_frame_indices: Sequence[int],
    *,
    rle_decode_success: bool,
) -> dict[str, Any]:
    track_counts_by_class: dict[str, int] = {prompt.ontology_id: 0 for prompt in prompts}
    for track in normalized_payload.get("tracks", []):
        track_counts_by_class[track["class_name"]] = track_counts_by_class.get(track["class_name"], 0) + 1
    class_coverage = {
        prompt.ontology_id: track_counts_by_class.get(prompt.ontology_id, 0) > 0
        for prompt in prompts
    }
    duplicate_count = _duplicate_id_count_for_review_sample(normalized_payload, review_frame_indices)
    interaction_counts = {
        prompt.ontology_id: sum(
            1
            for row in interaction_rows
            if row["class_name"] == prompt.ontology_id and row["likely_contact_boolean"]
        )
        for prompt in prompts
    }
    summary = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "source": bundle.project.delta.primary_backbone,
        "selected_seed_frame": {
            "frame_idx": seed_frame.frame_idx,
            "frame_name": seed_frame.frame_name,
            "segment": seed_frame.segment,
            "candidate_pool": seed_frame.candidate_pool,
            "coverage_count": seed_frame.coverage_count,
            "detected_ontology_ids": list(seed_frame.detected_ontology_ids),
            "detection_scores": seed_frame.detection_scores,
        },
        "track_count": len(normalized_payload.get("tracks", [])),
        "track_count_by_class": track_counts_by_class,
        "class_coverage": class_coverage,
        "review_sample_frame_count": len(review_frame_indices),
        "duplicate_id_count_in_review_sample": duplicate_count,
        "mean_labeled_mask_iou": None,
        "mask_entry_count": len(masks_entries),
        "rle_decode_success": rle_decode_success,
        "interaction_candidate_count": sum(1 for row in interaction_rows if row["likely_contact_boolean"]),
        "interaction_counts_by_class": interaction_counts,
        "qc_flags": {
            "all_ontology_entities_tracked": all(class_coverage.values()),
            "duplicate_ids_ok": duplicate_count <= bundle.project.acceptance.delta_max_duplicate_ids_in_review_sample,
            "rle_decode_success": rle_decode_success,
        },
    }
    return summary


def enforce_delta_acceptance(bundle: SpecBundle, summary: dict[str, Any]) -> None:
    if bundle.project.acceptance.delta_require_all_ontology_entities_tracked and not summary["qc_flags"]["all_ontology_entities_tracked"]:
        missing = [ontology_id for ontology_id, covered in summary["class_coverage"].items() if not covered]
        raise ObjectExtractError(
            "delta did not cover every ontology entity required by project.acceptance: "
            + ", ".join(sorted(missing))
        )
    if summary["duplicate_id_count_in_review_sample"] > bundle.project.acceptance.delta_max_duplicate_ids_in_review_sample:
        raise ObjectExtractError(
            f"delta duplicate id count {summary['duplicate_id_count_in_review_sample']} exceeded "
            f"delta_max_duplicate_ids_in_review_sample={bundle.project.acceptance.delta_max_duplicate_ids_in_review_sample}"
        )
    if bundle.project.acceptance.delta_require_rle_decode_success and not summary["rle_decode_success"]:
        raise ObjectExtractError("delta mask serialization failed the COCO RLE decode round-trip check")
    mean_labeled_mask_iou = summary.get("mean_labeled_mask_iou")
    if mean_labeled_mask_iou is not None and mean_labeled_mask_iou < bundle.project.acceptance.delta_min_mask_iou_sample:
        raise ObjectExtractError(
            f"delta mean labeled mask IoU {mean_labeled_mask_iou:.3f} fell below "
            f"delta_min_mask_iou_sample={bundle.project.acceptance.delta_min_mask_iou_sample:.3f}"
        )


def extract_monocular_objects(
    bundle: SpecBundle,
    beta_dir: Path,
    out_dir: Path,
    grounded_sam2_root: Path | None = None,
    gamma_dir: Path | None = None,
    *,
    device: str = "cuda",
    keep_workdir: bool = False,
) -> dict[str, Any]:
    resolved_grounded_sam2_root = ensure_delta_dependencies(grounded_sam2_root)

    frames_dir = beta_dir / "frames"
    frame_index_path = frames_dir / "index.csv"
    camera_pose_path = beta_dir / "camera" / "camera_poses.json"
    if not frames_dir.exists():
        raise ObjectExtractError(f"missing beta frames directory: {frames_dir}")
    if not frame_index_path.exists():
        raise ObjectExtractError(f"missing beta frame index: {frame_index_path}")
    if not camera_pose_path.exists():
        raise ObjectExtractError(f"missing beta camera pose export: {camera_pose_path}")

    frame_records = load_frame_index_csv(frame_index_path)
    load_camera_pose_payload(camera_pose_path)
    prompts = build_object_prompt_specs(bundle)

    objects_root = out_dir / "objects"
    overlays_dir = objects_root / "overlays"
    work_dir = objects_root / "_workdir"
    objects_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    preroll_candidates = sample_segment_frames(
        frame_records,
        segment="preroll",
        limit=bundle.project.delta.overlay_sample_count,
    )
    active_candidates = sample_segment_frames(
        frame_records,
        segment="active",
        limit=bundle.project.delta.overlay_sample_count,
    )
    preroll_probes = probe_seed_frame_candidates(
        grounded_sam2_root=resolved_grounded_sam2_root,
        frame_candidates=preroll_candidates,
        prompts=prompts,
        device=device,
        work_dir=work_dir,
    )
    active_probes = []
    if not preroll_probes or max((probe.coverage_count for probe in preroll_probes), default=0) < len(prompts):
        active_probes = probe_seed_frame_candidates(
            grounded_sam2_root=resolved_grounded_sam2_root,
            frame_candidates=active_candidates,
            prompts=prompts,
            device=device,
            work_dir=work_dir,
        )
    seed_frame = choose_seed_frame(frame_records, prompts, preroll_probes, active_probes)

    raw_payload = run_grounded_sam2_tracking(
        grounded_sam2_root=resolved_grounded_sam2_root,
        frames_dir=frames_dir,
        prompts=prompts,
        seed_frame=seed_frame,
        device=device,
        work_dir=work_dir,
    )
    raw_payload.setdefault("video_id", bundle.project.video_id)

    normalized_payload, masks_entries, masks_by_track_frame, rle_decode_success = normalize_grounded_sam2_tracks(
        raw_payload=raw_payload,
        frame_records=frame_records,
        prompts=prompts,
    )
    gamma_root = gamma_dir.resolve() if gamma_dir is not None and gamma_dir.exists() else beta_dir.resolve()
    wrist_observations = load_optional_gamma_wrist_observations(gamma_root if gamma_root.exists() else None)
    interaction_rows = build_interaction_rows(
        bundle=bundle,
        normalized_payload=normalized_payload,
        masks_by_track_frame=masks_by_track_frame,
        wrist_observations=wrist_observations,
    )
    review_frame_indices = write_object_overlays(
        overlay_dir=overlays_dir,
        frame_records=frame_records,
        normalized_payload=normalized_payload,
        masks_by_track_frame=masks_by_track_frame,
        wrist_observations=wrist_observations,
        overlay_count=bundle.project.delta.overlay_sample_count,
    )

    write_prompts_yaml(objects_root / "prompts.yaml", bundle, prompts, seed_frame)
    (objects_root / "object_tracks.json").write_text(
        json.dumps(normalized_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(objects_root / "masks_rle.jsonl", masks_entries)
    write_interactions_parquet(objects_root / "interactions.parquet", interaction_rows)

    summary = build_delta_summary(
        bundle=bundle,
        prompts=prompts,
        seed_frame=seed_frame,
        normalized_payload=normalized_payload,
        masks_entries=masks_entries,
        interaction_rows=interaction_rows,
        review_frame_indices=review_frame_indices,
        rle_decode_success=rle_decode_success,
    )
    (objects_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if keep_workdir and work_dir.exists():
        native_dir = objects_root / "native"
        if native_dir.exists():
            shutil.rmtree(native_dir)
        shutil.copytree(work_dir, native_dir)
    if not keep_workdir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    enforce_delta_acceptance(bundle, summary)
    return summary
