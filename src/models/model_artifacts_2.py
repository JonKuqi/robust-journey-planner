from __future__ import annotations

import json

import os

from pathlib import Path

from src.config.settings import ProjectSettings, get_settings

ACTIVE_MODEL: str | None = None

SLOTS = ("model_a", "model_b")


def _model_dir(settings: ProjectSettings) -> Path:
    return Path(settings.local_artifacts_path) / "model"


def get_active_model(settings: ProjectSettings | None = None) -> str:
    """Return the active model slot name, reading from disk if not set in memory."""
    global ACTIVE_MODEL
    if ACTIVE_MODEL is not None:
        return ACTIVE_MODEL
    settings = settings or get_settings()
    path = _model_dir(settings) / "active_model"
    ACTIVE_MODEL = path.read_text().strip() if path.exists() else "model_a"
    return ACTIVE_MODEL


def update_active_model(slot: str, settings: ProjectSettings | None = None) -> None:
    """Update the in-memory pointer and persist it to disk."""
    global ACTIVE_MODEL
    if slot not in SLOTS:
        raise ValueError(f"slot must be one of {SLOTS}, got {slot!r}")
    ACTIVE_MODEL = slot
    settings = settings or get_settings()
    path = _model_dir(settings) / "active_model"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(slot)


def inactive_model(settings: ProjectSettings | None = None) -> str:
    active = get_active_model(settings)
    return "model_b" if active == "model_a" else "model_a"


def read_dates(filename: str, settings: ProjectSettings | None = None) -> dict[str, str]:
    """Read a START=/END= tracking file. Returns {} if missing."""
    settings = settings or get_settings()
    path = _model_dir(settings) / filename
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def write_dates(filename: str, start: str, end: str, settings: ProjectSettings | None = None) -> None:
    """Write a START=/END= tracking file."""
    settings = settings or get_settings()
    path = _model_dir(settings) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"START={start}\nEND={end}\n")


def read_failed_days(filename: str, settings: ProjectSettings | None = None) -> list[str]:
    """Read a failed-days tracking file. Returns [] if missing."""
    settings = settings or get_settings()
    path = _model_dir(settings) / filename
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_failed_days(filename: str, days: list[str],
    settings: ProjectSettings | None = None,
) -> None:
    """Write a failed-days tracking file."""
    settings = settings or get_settings()
    path = _model_dir(settings) / filename

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(days) + ("\n" if days else ""))


def read_divergence_state(self) -> dict:
        import json
        path = _model_dir(settings) / "divergence_state.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception as exc:
            print(f"  Could not read divergence state ({exc}) — treating as empty.")
            return {}


def write_divergence_state(self, active_slot: str, baseline_pinball: float | None, train_end: str) -> None:
        payload = {
            "active_slot":       active_slot,
            "baseline_pinball":  baseline_pinball,
            "train_end":         train_end,
            "written_at":        date.today().isoformat(),
        }
        path = _model_dir(settings) / "divergence_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def init_model_dirs(settings: ProjectSettings | None = None) -> None:
    """Create the artifact directory structure if it doesn't exist yet."""
    settings = settings or get_settings()
    base = _model_dir(settings)
    for slot in SLOTS:
        (base / slot).mkdir(parents=True, exist_ok=True)
    if not (base / "active_model").exists():
        (base / "active_model").write_text("model_a")
