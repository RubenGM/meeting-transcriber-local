from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeEnvironment:
    matplotlib_cache: Path
    huggingface_cache: Path


def configure_runtime_environment(project_dir: Path) -> RuntimeEnvironment:
    matplotlib_cache = project_dir / ".cache" / "matplotlib"
    huggingface_cache = project_dir / "models" / "huggingface"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    huggingface_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("HF_HOME", str(huggingface_cache))
    return RuntimeEnvironment(
        matplotlib_cache=Path(os.environ["MPLCONFIGDIR"]),
        huggingface_cache=Path(os.environ["HF_HOME"]),
    )
