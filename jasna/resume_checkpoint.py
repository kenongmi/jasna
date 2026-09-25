"""Resumable-download / resumable-transcode checkpoint support for Jasna.

This module is intentionally self-contained: it only reads and writes a small
JSON sidecar file next to the output video, plus a set of already-rendered
video fragments in the same working directory. No existing pipeline
behaviour is modified unless a checkpoint file for the same job is found on
disk.

A "job" is identified by the resolved input path + output path + a content
fingerprint of the encoder/restoration settings, so a checkpoint from a
different job configuration is never reused by mistake.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)

CHECKPOINT_SUFFIX = ".jasna-resume.json"
CHECKPOINT_VERSION = 1


@dataclasses.dataclass
class ResumeCheckpoint:
    version: int
    job_fingerprint: str
    input_video: str
    output_video: str
    fragments: list
    last_completed_frame: int
    last_completed_time: float
    total_frames: int
    created_at: float
    updated_at: float

    @classmethod
    def new(
        cls,
        *,
        job_fingerprint: str,
        input_video: str,
        output_video: str,
        total_frames: int,
    ) -> "ResumeCheckpoint":
        now = time.time()
        return cls(
            version=CHECKPOINT_VERSION,
            job_fingerprint=job_fingerprint,
            input_video=input_video,
            output_video=output_video,
            fragments=[],
            last_completed_frame=0,
            last_completed_time=0.0,
            total_frames=total_frames,
            created_at=now,
            updated_at=now,
        )

    def fragment_paths(self) -> list[tuple[Path, float]]:
        return [(Path(p), float(d)) for p, d in self.fragments]

    def add_fragment(self, path: Path, duration: float) -> None:
        self.fragments.append([str(path), float(duration)])

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ResumeCheckpoint":
        return cls(**{f.name: data[f.name] for f in dataclasses.fields(cls)})


def compute_job_fingerprint(*, codec: str, encoder_settings: dict, fp16: bool,
                             detection_model_name: str, vr_mode: str,
                             vr_projection: str, retarget_high_fps: bool) -> str:
    """Fingerprint the render settings that affect frame-for-frame output.

    If any of these change between runs, a stale checkpoint must NOT be
    reused, because the previously rendered fragments would not match the
    settings of the new run.
    """
    payload = json.dumps(
        {
            "codec": codec,
            "encoder_settings": encoder_settings,
            "fp16": fp16,
            "detection_model_name": detection_model_name,
            "vr_mode": vr_mode,
            "vr_projection": vr_projection,
            "retarget_high_fps": retarget_high_fps,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def checkpoint_path_for(output_video: Path) -> Path:
    return output_video.with_name(output_video.name + CHECKPOINT_SUFFIX)


def load_checkpoint(output_video: Path, *, job_fingerprint: str) -> ResumeCheckpoint | None:
    """Load a checkpoint for this output path, if one exists and is valid."""
    path = checkpoint_path_for(output_video)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = ResumeCheckpoint.from_dict(data)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        log.warning("[resume] could not read checkpoint %s: %s", path, exc)
        return None

    if checkpoint.version != CHECKPOINT_VERSION:
        log.info("[resume] checkpoint version mismatch, ignoring: %s", path)
        return None
    if checkpoint.job_fingerprint != job_fingerprint:
        log.info("[resume] checkpoint is for a different job configuration, ignoring: %s", path)
        return None
    if not checkpoint.fragments:
        log.info("[resume] checkpoint has no completed fragments yet, ignoring: %s", path)
        return None
    for frag_path, _duration in checkpoint.fragments:
        if not Path(frag_path).exists():
            log.info("[resume] checkpoint fragment missing on disk (%s), ignoring: %s", frag_path, path)
            return None
    if checkpoint.last_completed_frame <= 0:
        log.info("[resume] checkpoint has no completed frames yet, ignoring: %s", path)
        return None
    return checkpoint


def save_checkpoint(checkpoint: ResumeCheckpoint) -> None:
    """Atomically persist the checkpoint next to the output video."""
    path = checkpoint_path_for(Path(checkpoint.output_video))
    checkpoint.updated_at = time.time()
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(checkpoint.to_dict(), fh)
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def clear_checkpoint(output_video: Path) -> None:
    """Remove the checkpoint sidecar file after a successful, complete run."""
    path = checkpoint_path_for(output_video)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("[resume] could not remove checkpoint %s: %s", path, exc)
