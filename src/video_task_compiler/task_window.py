from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .specs import PROJECT_SCHEMA_VERSION


TASK_WINDOW_FILENAME = "task_window.json"


class TaskWindowError(Exception):
    """Raised when the task window cannot be resolved or validated."""


@dataclass(frozen=True)
class TaskWindowFrameRecord:
    frame_idx: int
    frame_name: str
    segment: str


@dataclass(frozen=True)
class TaskWindow:
    start_frame_idx: int
    end_frame_idx: int
    source: str

    @property
    def frame_count(self) -> int:
        return self.end_frame_idx - self.start_frame_idx + 1

    def contains(self, frame_idx: int) -> bool:
        return self.start_frame_idx <= int(frame_idx) <= self.end_frame_idx

    def to_payload(self, *, video_id: str) -> dict[str, Any]:
        return {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "video_id": video_id,
            "source": self.source,
            "start_frame_idx": self.start_frame_idx,
            "end_frame_idx": self.end_frame_idx,
        }


def task_window_path(artifact_root: Path) -> Path:
    return artifact_root / TASK_WINDOW_FILENAME


def load_task_window_frame_records(path: Path) -> list[TaskWindowFrameRecord]:
    if not path.exists():
        raise TaskWindowError(f"missing beta frame index for task window resolution: {path}")

    records: list[TaskWindowFrameRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                TaskWindowFrameRecord(
                    frame_idx=int(row["frame_idx"]),
                    frame_name=str(row["frame_name"]),
                    segment=str(row["segment"]),
                )
            )
    if not records:
        raise TaskWindowError(f"beta frame index was empty: {path}")
    return records


def _item_frame_idx(item: Any) -> int:
    if isinstance(item, dict):
        if "frame_idx" not in item:
            raise TaskWindowError("task-window filtering expected dict items with a frame_idx field")
        return int(item["frame_idx"])
    if hasattr(item, "frame_idx"):
        return int(getattr(item, "frame_idx"))
    raise TaskWindowError("task-window filtering expected items with a frame_idx attribute")


def filter_items_to_task_window(items: Sequence[Any], task_window: TaskWindow) -> list[Any]:
    return [item for item in items if task_window.contains(_item_frame_idx(item))]


def _frame_index_set(frame_records: Sequence[Any]) -> set[int]:
    return {_item_frame_idx(record) for record in frame_records}


def _validate_task_window_bounds(
    frame_records: Sequence[Any],
    *,
    start_frame_idx: int,
    end_frame_idx: int,
    source: str,
) -> TaskWindow:
    if start_frame_idx > end_frame_idx:
        raise TaskWindowError(
            f"task window start frame {start_frame_idx} must be less than or equal to end frame {end_frame_idx}"
        )

    frame_indices = _frame_index_set(frame_records)
    missing_bounds = [bound for bound in (start_frame_idx, end_frame_idx) if bound not in frame_indices]
    if missing_bounds:
        raise TaskWindowError(
            "task window bounds must exist in frames/index.csv: "
            + ", ".join(str(bound) for bound in missing_bounds)
        )

    return TaskWindow(
        start_frame_idx=int(start_frame_idx),
        end_frame_idx=int(end_frame_idx),
        source=source,
    )


def _default_non_preroll_window(frame_records: Sequence[Any]) -> TaskWindow:
    active_indices = [
        _item_frame_idx(record)
        for record in frame_records
        if getattr(record, "segment", None) != "preroll"
        or (isinstance(record, dict) and str(record.get("segment")) != "preroll")
    ]
    if not active_indices:
        raise TaskWindowError("could not resolve a default task window because there were no non-preroll frames")
    return TaskWindow(
        start_frame_idx=min(active_indices),
        end_frame_idx=max(active_indices),
        source="default_non_preroll",
    )


def _load_persisted_task_window(
    path: Path,
    frame_records: Sequence[Any],
    *,
    video_id: str,
) -> TaskWindow:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TaskWindowError(f"expected a JSON object in persisted task window: {path}")
    persisted_video_id = payload.get("video_id")
    if persisted_video_id is not None and str(persisted_video_id) != video_id:
        raise TaskWindowError(
            f"persisted task window video_id '{persisted_video_id}' did not match expected video_id '{video_id}'"
        )
    if "start_frame_idx" not in payload or "end_frame_idx" not in payload:
        raise TaskWindowError(f"persisted task window was missing start/end bounds: {path}")
    return _validate_task_window_bounds(
        frame_records,
        start_frame_idx=int(payload["start_frame_idx"]),
        end_frame_idx=int(payload["end_frame_idx"]),
        source=str(payload.get("source", "persisted_artifact")),
    )


def _unique_roots(roots: Iterable[Path | None]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if root is None:
            continue
        resolved = root.resolve()
        key = resolved.as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def resolve_task_window(
    frame_records: Sequence[Any],
    *,
    video_id: str,
    artifact_roots: Sequence[Path | None] = (),
    task_start_frame: int | None = None,
    task_end_frame: int | None = None,
) -> TaskWindow:
    if (task_start_frame is None) != (task_end_frame is None):
        raise TaskWindowError("both --task-start-frame and --task-end-frame must be provided together")

    if task_start_frame is not None and task_end_frame is not None:
        return _validate_task_window_bounds(
            frame_records,
            start_frame_idx=int(task_start_frame),
            end_frame_idx=int(task_end_frame),
            source="cli_override",
        )

    for artifact_root in _unique_roots(artifact_roots):
        persisted_path = task_window_path(artifact_root)
        if persisted_path.exists():
            return _load_persisted_task_window(
                persisted_path,
                frame_records,
                video_id=video_id,
            )

    return _default_non_preroll_window(frame_records)


def persist_task_window(
    task_window: TaskWindow,
    artifact_root: Path,
    *,
    video_id: str,
) -> Path:
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_path = task_window_path(artifact_root)
    output_path.write_text(
        json.dumps(task_window.to_payload(video_id=video_id), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
