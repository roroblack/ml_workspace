#!/usr/bin/env python
"""ML Unified Lab 실험 실행기.

사용법:
    python run_experiment.py --config configs/experiments/iris_logistic_grid.yaml
    python run_experiment.py --list            # 사용 가능한 실험 목록
    python run_experiment.py --all             # 모든 실험 순차 실행
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# src 패키지 import 가능하도록 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import run_experiment  # noqa: E402

EXPERIMENTS_DIR = PROJECT_ROOT / "configs" / "experiments"


def list_experiments() -> list[Path]:
    return sorted(EXPERIMENTS_DIR.glob("*.yaml"))


def main():
    parser = argparse.ArgumentParser(description="ML Unified Lab 실험 실행기")
    parser.add_argument("--config", "-c", help="실험 설정 yaml 경로")
    parser.add_argument("--list", "-l", action="store_true", help="실험 목록 출력")
    parser.add_argument("--all", "-a", action="store_true", help="모든 실험 실행")
    args = parser.parse_args()

    if args.list:
        print("사용 가능한 실험:")
        for p in list_experiments():
            print(f"  - {p.relative_to(PROJECT_ROOT)}")
        return

    if args.all:
        failures = []
        for p in list_experiments():
            print("\n" + "=" * 70)
            try:
                run_experiment(p)
            except Exception as e:
                print(f"[error] {p.name} 실패: {e}")
                traceback.print_exc()
                failures.append((p.name, str(e)))
        print("\n" + "=" * 70)
        if failures:
            print(f"실패한 실험 {len(failures)}건:")
            for name, err in failures:
                print(f"  - {name}: {err}")
        else:
            print("모든 실험 완료.")
        return

    if not args.config:
        parser.error("--config 또는 --list / --all 중 하나를 지정하세요.")

    run_experiment(args.config)


if __name__ == "__main__":
    main()
