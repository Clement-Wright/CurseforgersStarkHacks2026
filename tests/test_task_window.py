from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_task_compiler.task_window import (
    TaskWindowError,
    TaskWindowFrameRecord,
    filter_items_to_task_window,
    persist_task_window,
    resolve_task_window,
)


def _frame_records() -> list[TaskWindowFrameRecord]:
    return [
        TaskWindowFrameRecord(frame_idx=0, frame_name="frame_000000.png", segment="preroll"),
        TaskWindowFrameRecord(frame_idx=1, frame_name="frame_000001.png", segment="demo"),
        TaskWindowFrameRecord(frame_idx=2, frame_name="frame_000002.png", segment="demo"),
        TaskWindowFrameRecord(frame_idx=3, frame_name="frame_000003.png", segment="demo"),
    ]


def test_resolve_task_window_defaults_to_non_preroll() -> None:
    task_window = resolve_task_window(_frame_records(), video_id="demo_0001")

    assert task_window.start_frame_idx == 1
    assert task_window.end_frame_idx == 3
    assert task_window.source == "default_non_preroll"


def test_resolve_task_window_uses_cli_override_bounds() -> None:
    task_window = resolve_task_window(
        _frame_records(),
        video_id="demo_0001",
        task_start_frame=2,
        task_end_frame=3,
    )

    assert task_window.start_frame_idx == 2
    assert task_window.end_frame_idx == 3
    assert task_window.source == "cli_override"


def test_resolve_task_window_reuses_persisted_artifact(tmp_path: Path) -> None:
    artifact_root = tmp_path / "run"
    task_window = resolve_task_window(
        _frame_records(),
        video_id="demo_0001",
        task_start_frame=2,
        task_end_frame=3,
    )
    persist_task_window(task_window, artifact_root, video_id="demo_0001")

    resolved = resolve_task_window(
        _frame_records(),
        video_id="demo_0001",
        artifact_roots=(artifact_root,),
    )

    assert resolved.start_frame_idx == 2
    assert resolved.end_frame_idx == 3
    payload = json.loads((artifact_root / "task_window.json").read_text(encoding="utf-8"))
    assert payload["video_id"] == "demo_0001"


def test_resolve_task_window_rejects_invalid_bounds() -> None:
    with pytest.raises(TaskWindowError, match="both --task-start-frame and --task-end-frame must be provided together"):
        resolve_task_window(_frame_records(), video_id="demo_0001", task_start_frame=1)

    with pytest.raises(TaskWindowError, match="must be less than or equal"):
        resolve_task_window(_frame_records(), video_id="demo_0001", task_start_frame=3, task_end_frame=2)

    with pytest.raises(TaskWindowError, match="must exist in frames/index.csv"):
        resolve_task_window(_frame_records(), video_id="demo_0001", task_start_frame=9, task_end_frame=10)


def test_filter_items_to_task_window_supports_records_and_dicts() -> None:
    task_window = resolve_task_window(
        _frame_records(),
        video_id="demo_0001",
        task_start_frame=2,
        task_end_frame=3,
    )

    filtered_records = filter_items_to_task_window(_frame_records(), task_window)
    filtered_rows = filter_items_to_task_window(
        [{"frame_idx": 1}, {"frame_idx": 2}, {"frame_idx": 3}],
        task_window,
    )

    assert [record.frame_idx for record in filtered_records] == [2, 3]
    assert [row["frame_idx"] for row in filtered_rows] == [2, 3]
