"""설정 로딩/병합 유틸리티.

experiments/*.yaml 한 개를 읽고, datasets/models/searches 레지스트리 yaml과
병합해 실행에 필요한 완성된 설정 dict를 만든다.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# 프로젝트 루트 = 이 파일 기준 두 단계 위 (src/config.py -> ml_unified_lab/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_registries() -> dict[str, dict[str, Any]]:
    """datasets / models / searches 레지스트리를 한 번에 읽어온다."""
    return {
        "datasets": _read_yaml(CONFIG_DIR / "datasets.yaml").get("datasets", {}),
        "models": _read_yaml(CONFIG_DIR / "models.yaml").get("models", {}),
        "searches": _read_yaml(CONFIG_DIR / "searches.yaml").get("searches", {}),
    }


def load_experiment(config_path: str | Path) -> dict[str, Any]:
    """experiment yaml을 읽고 레지스트리 정보를 합쳐 완성 설정을 만든다."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        # 상대경로면 프로젝트 루트 기준으로 해석
        candidate = PROJECT_ROOT / config_path
        config_path = candidate if candidate.exists() else config_path
    if not config_path.exists():
        raise FileNotFoundError(f"실험 설정 파일을 찾을 수 없습니다: {config_path}")

    exp = _read_yaml(config_path)
    registries = load_registries()

    dataset_name = exp.get("dataset")
    model_name = exp.get("model")
    search_method = (exp.get("search") or {}).get("method", "none")

    dataset_meta = registries["datasets"].get(dataset_name, {})
    model_meta = registries["models"].get(model_name, {})
    search_meta = registries["searches"].get(search_method, {})

    merged = copy.deepcopy(exp)
    merged["_dataset_meta"] = dataset_meta
    merged["_model_meta"] = model_meta
    merged["_search_meta"] = search_meta
    merged["_config_path"] = str(config_path)

    # framework 미지정 시 모델 레지스트리에서 추론
    if "framework" not in merged and model_meta.get("framework"):
        merged["framework"] = model_meta["framework"]
    # task 미지정 시 데이터셋 레지스트리에서 추론
    if "task" not in merged and dataset_meta.get("task"):
        merged["task"] = dataset_meta["task"]

    return merged


def resolve_path(rel_or_abs: str | Path) -> Path:
    """레지스트리에 적힌 source 경로를 프로젝트 루트 기준 절대경로로."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
