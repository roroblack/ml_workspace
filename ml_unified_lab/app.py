"""ML Unified Lab - Streamlit GUI 대시보드.

실행:
    streamlit run app.py
    (또는)  python -m streamlit run app.py

탭
  1. 프리셋 실험   : configs/experiments/*.yaml 을 골라 바로 실행
  2. 직접 구성     : 데이터셋/모델/탐색을 드롭다운 + 하이퍼파라미터 위젯으로 조합해 실행
  3. 코드 보기/편집: src/ · configs/ 소스를 코드 블럭으로 보고 수정하며 복습
  4. 지난 결과     : runs/ 폴더의 과거 실행 결과(지표/그래프) 열람
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_registries  # noqa: E402
from src.param_specs import OPTIMIZERS, TRAIN_SPECS, search_spec, train_spec_key  # noqa: E402
from src.training import run_experiment  # noqa: E402

EXPERIMENTS_DIR = PROJECT_ROOT / "configs" / "experiments"
RUNS_DIR = PROJECT_ROOT / "runs"
GUI_TMP_DIR = RUNS_DIR / "_gui_configs"

VISION_DATASETS = {"mnist", "fashion_mnist", "cifar10"}
TRANSACTION_DATASETS = {"groceries", "online_retail"}

st.set_page_config(page_title="ML Unified Lab", page_icon="🧪", layout="wide")


# --------------------------------------------------------------------------- #
# 공통 유틸
# --------------------------------------------------------------------------- #
@st.cache_data
def registries():
    return load_registries()


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def dataset_kind(name: str) -> str:
    if name in VISION_DATASETS:
        return "vision"
    if name in TRANSACTION_DATASETS:
        return "transactions"
    return "tabular"


def default_plots(task: str, framework: str) -> list[str]:
    if task == "association_rules":
        return []
    if task == "regression":
        plots = ["residual_plot", "prediction_scatter"]
    else:
        plots = ["confusion_matrix"]
    if framework == "torch":
        plots += ["loss_curve", "accuracy_curve"]
    return plots


def execute(config: dict) -> tuple[dict | None, str]:
    """config dict 를 임시 yaml 로 저장 후 실행. (결과, 로그) 반환."""
    GUI_TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = GUI_TMP_DIR / f"{config['experiment_name']}.yaml"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            result = run_experiment(tmp_path)
        return result, buf.getvalue()
    except Exception as e:  # noqa: BLE001
        import traceback
        return None, buf.getvalue() + "\n" + traceback.format_exc()


def show_result(result: dict, log: str):
    """실행 결과(지표/플롯/trial/규칙)를 화면에 렌더링."""
    st.success(f"완료: {result['experiment_name']}  ·  방법={result.get('search_method')}")

    metrics = result.get("metrics", {})
    if metrics:
        cols = st.columns(len(metrics))
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k.upper(), f"{v:.4f}" if isinstance(v, float) else str(v))

    if result.get("best_params"):
        st.caption("best parameters")
        st.json(result["best_params"], expanded=False)

    plots = result.get("plots") or []
    if plots:
        st.subheader("그래프")
        pcols = st.columns(min(len(plots), 3))
        for i, p in enumerate(plots):
            if Path(p).exists():
                pcols[i % len(pcols)].image(p, caption=Path(p).stem, use_column_width=True)

    run_dir = Path(result.get("run_dir", ""))

    # 연관규칙 결과
    rules_csv = run_dir / "association_rules.csv"
    if rules_csv.exists():
        st.subheader("연관규칙 (lift 내림차순)")
        st.dataframe(pd.read_csv(rules_csv), use_container_width=True, hide_index=True)

    # 탐색 trial
    trials_csv = run_dir / "search_trials.csv"
    if trials_csv.exists():
        st.subheader("탐색 trial")
        st.dataframe(pd.read_csv(trials_csv), use_container_width=True, hide_index=True)

    st.caption(f"결과 저장 위치: {run_dir}")
    with st.expander("실행 로그"):
        st.code(log or "(로그 없음)")


def render_search_widgets(model: str, method: str, container) -> dict:
    """모델별 탐색 파라미터 위젯을 그려서 {param: 값 또는 리스트} 반환."""
    chosen: dict = {}
    specs = search_spec(model)
    if not specs:
        container.caption("이 모델은 조정할 하이퍼파라미터가 없습니다 (기본값으로 학습).")
        return chosen

    single = method == "none"
    container.caption("단일 값 선택" if single else "탐색에 포함할 값들(여러 개) 선택")
    for param, kind, options, default, helptext in specs:
        key = f"sw_{model}_{param}"
        if kind == "cat":
            opts = options
            if single:
                chosen[param] = container.selectbox(f"{param} — {helptext}", opts,
                                                    index=0, key=key)
            else:
                chosen[param] = container.multiselect(f"{param} — {helptext}", opts,
                                                      default=default, key=key)
        else:
            # int / float: 프리셋 값 멀티셀렉트 + 직접 입력 허용
            opts = default
            if single:
                chosen[param] = container.select_slider(
                    f"{param} — {helptext}", options=sorted(set(default)),
                    value=default[0], key=key)
            else:
                chosen[param] = container.multiselect(
                    f"{param} — {helptext}", options=sorted(set(default)),
                    default=default, key=key)
    return chosen


def render_train_widgets(framework: str, ds_kind: str, container) -> dict:
    """train 공통 옵션 위젯 → {param: 값}."""
    train: dict = {}
    spec_key = train_spec_key(framework, ds_kind)
    for param, kind, default, helptext in TRAIN_SPECS.get(spec_key, []):
        key = f"tw_{spec_key}_{param}"
        if kind == "cat":
            train[param] = container.selectbox(f"{param} — {helptext}", OPTIMIZERS,
                                               index=OPTIMIZERS.index(default), key=key)
        elif kind == "float":
            train[param] = container.number_input(f"{param} — {helptext}", value=float(default),
                                                  format="%.4f", key=key)
        else:
            train[param] = int(container.number_input(f"{param} — {helptext}", value=int(default),
                                                      step=1, key=key))
    # vision 서브샘플 0 -> 전체 학습(키 제거)
    for k in ("max_train_samples", "max_test_samples"):
        if train.get(k) == 0:
            train.pop(k, None)
    return train


# --------------------------------------------------------------------------- #
# 헤더
# --------------------------------------------------------------------------- #
st.title("🧪 ML Unified Lab")
st.caption("설정 기반 통합 ML/DL 실험 대시보드 — 데이터셋·모델·탐색을 골라 바로 실행하고 코드를 복습하세요.")

if not torch_available():
    st.warning("PyTorch 미설치 — 딥러닝(torch) 실험은 비활성화됩니다. `pip install torch torchvision`")

tab_preset, tab_build, tab_code, tab_runs = st.tabs(
    ["▶ 프리셋 실험", "🧩 직접 구성", "📝 코드 보기/편집", "📁 지난 결과"]
)


# --------------------------------------------------------------------------- #
# 탭 1: 프리셋 실험
# --------------------------------------------------------------------------- #
with tab_preset:
    presets = sorted(EXPERIMENTS_DIR.glob("*.yaml"))
    if not presets:
        st.info("configs/experiments/ 에 실험 설정이 없습니다.")
    else:
        names = [p.stem for p in presets]
        choice = st.selectbox("실험 선택", names, key="preset_choice")
        cfg_path = EXPERIMENTS_DIR / f"{choice}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        c1, c2 = st.columns([2, 1])
        with c1:
            st.code(cfg_path.read_text(encoding="utf-8"), language="yaml")
        with c2:
            st.write("**요약**")
            st.write(f"- 데이터셋: `{cfg.get('dataset')}`")
            st.write(f"- 모델: `{cfg.get('model')}`")
            st.write(f"- 프레임워크: `{cfg.get('framework')}`")
            st.write(f"- 탐색: `{(cfg.get('search') or {}).get('method')}`")

        needs_torch = cfg.get("framework") == "torch"
        disabled = needs_torch and not torch_available()
        if st.button("이 실험 실행", type="primary", disabled=disabled, key="run_preset"):
            with st.spinner("실험 실행 중..."):
                result, log = execute(cfg)
            if result:
                show_result(result, log)
            else:
                st.error("실행 실패")
                st.code(log)


# --------------------------------------------------------------------------- #
# 탭 2: 직접 구성
# --------------------------------------------------------------------------- #
with tab_build:
    reg = registries()
    ds_reg, model_reg, search_reg = reg["datasets"], reg["models"], reg["searches"]

    c1, c2, c3 = st.columns(3)
    with c1:
        dataset = st.selectbox("데이터셋", list(ds_reg.keys()), key="b_ds")
        task = ds_reg[dataset].get("task", "classification")
        ds_kind = dataset_kind(dataset)
        st.caption(f"task: `{task}`  ·  kind: `{ds_kind}`")
    with c2:
        def model_ok(meta):
            return task in meta.get("task_types", [])

        model_opts = [m for m, meta in model_reg.items() if model_ok(meta)]
        model = st.selectbox("모델", model_opts, key="b_model")
        framework = model_reg[model].get("framework", "sklearn")
        st.caption(f"framework: `{framework}`")
    with c3:
        if task == "association_rules":
            method = "apriori"
            st.caption("탐색 방법: `apriori` (고정)")
            trials = 0
        else:
            method = st.selectbox("탐색 방법", list(search_reg.keys()), key="b_search")
            trials = st.number_input(
                "trials", 1, 100,
                value=int(search_reg.get(method, {}).get("default_trials", 10)),
                key="b_trials",
            )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("**하이퍼파라미터**")
        if task == "association_rules":
            chosen = render_search_widgets("apriori", "none", left)
        else:
            chosen = render_search_widgets(model, method, left)
    with right:
        st.markdown("**학습 설정 (train)**")
        if task == "association_rules":
            train = {}
        else:
            train = render_train_widgets(framework, ds_kind, right)

    disabled = framework == "torch" and not torch_available()
    if disabled:
        st.warning("이 조합은 PyTorch가 필요합니다.")

    if st.button("실험 실행", type="primary", disabled=disabled, key="run_build"):
        ts = datetime.now().strftime("%H%M%S")
        config = {
            "experiment_name": f"{dataset}_{model}_{method}_{ts}",
            "dataset": dataset,
            "model": model,
            "framework": framework,
            "task": task,
            "search": {"method": method},
            "train": train,
            "evaluation": {"plots": default_plots(task, framework)},
        }
        if task != "association_rules":
            config["search"]["trials"] = int(trials)

        # 선택값을 search.params / train 에 배치
        if task == "association_rules":
            config["search"]["params"] = chosen  # min_support 등 단일값
        elif method == "none":
            # 단일 실행: sklearn은 search.params(첫값 사용), torch는 train에 반영
            if framework == "sklearn" and chosen:
                config["search"]["params"] = {k: [v] for k, v in chosen.items()}
            elif framework == "torch":
                config["train"].update(chosen)
        else:
            # 탐색: 비어있지 않은 항목만 그리드로
            params = {k: v for k, v in chosen.items() if isinstance(v, list) and v}
            if params:
                config["search"]["params"] = params

        with st.spinner("실험 실행 중... (딥러닝/탐색은 수십 초~수 분 걸릴 수 있어요)"):
            result, log = execute(config)
        if result:
            show_result(result, log)
        else:
            st.error("실행 실패")
            st.code(log)


# --------------------------------------------------------------------------- #
# 탭 3: 코드 보기/편집
# --------------------------------------------------------------------------- #
with tab_code:
    st.markdown(
        "소스 코드를 코드 블럭으로 보고, 필요하면 직접 수정하며 복습하세요. "
        "저장 시 같은 폴더에 `.bak` 백업을 먼저 만든 뒤 덮어씁니다."
    )

    # 편집 허용 범위: 프로젝트 코드/설정 (archive·runs 제외)
    code_files = []
    for pat in ["src/**/*.py", "configs/**/*.yaml", "*.py"]:
        code_files += [p for p in PROJECT_ROOT.glob(pat) if p.is_file()]
    code_files = sorted({p.relative_to(PROJECT_ROOT).as_posix() for p in code_files})

    sel = st.selectbox("파일 선택", code_files, key="code_file")
    fpath = PROJECT_ROOT / sel
    lang = "python" if sel.endswith(".py") else "yaml"
    original = fpath.read_text(encoding="utf-8")

    mode = st.radio("모드", ["보기", "편집"], horizontal=True, key="code_mode")

    if mode == "보기":
        st.code(original, language=lang)
    else:
        st.warning("편집한 내용은 실제 파일에 저장됩니다. 저장 전 `.bak` 백업이 생성됩니다.")
        edited = st.text_area("내용", value=original, height=500, key=f"edit_{sel}")
        col_a, col_b = st.columns([1, 4])
        with col_a:
            if st.button("💾 저장", type="primary", key="code_save"):
                if edited == original:
                    st.info("변경 사항이 없습니다.")
                else:
                    bak = fpath.with_suffix(fpath.suffix + ".bak")
                    bak.write_text(original, encoding="utf-8")
                    fpath.write_text(edited, encoding="utf-8")
                    st.success(f"저장 완료. 백업: {bak.relative_to(PROJECT_ROOT).as_posix()}")
        with col_b:
            if sel.endswith(".py"):
                if st.button("✓ 문법 검사 (편집 내용)", key="code_check"):
                    import py_compile
                    import tempfile
                    tmp = Path(tempfile.gettempdir()) / "syntax_check.py"
                    tmp.write_text(edited, encoding="utf-8")
                    try:
                        py_compile.compile(str(tmp), doraise=True)
                        st.success("문법 OK")
                    except py_compile.PyCompileError as e:
                        st.error(f"문법 오류:\n{e}")


# --------------------------------------------------------------------------- #
# 탭 4: 지난 결과
# --------------------------------------------------------------------------- #
with tab_runs:
    run_dirs = sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")],
        reverse=True,
    ) if RUNS_DIR.exists() else []

    if not run_dirs:
        st.info("아직 실행 결과가 없습니다. 실험을 먼저 실행하세요.")
    else:
        choice = st.selectbox("실행 결과 선택", [d.name for d in run_dirs], key="run_choice")
        run_dir = RUNS_DIR / choice
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            result = json.loads(metrics_file.read_text(encoding="utf-8"))
            result["run_dir"] = str(run_dir)
            show_result(result, log="(저장된 결과 — 로그 없음)")
        else:
            st.warning("metrics.json 이 없는 실행 폴더입니다.")
            st.write([p.name for p in run_dir.iterdir()])
