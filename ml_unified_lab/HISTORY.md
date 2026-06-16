# 작업 히스토리 (HISTORY)

> `ml_unified_lab` 작업 내역을 시간순으로 기록한다.
> 각 항목은 `YYYY-MM-DD HH:MM` 형식의 절대 시각을 포함한다. (규칙: [RULES.md](RULES.md) §6)

---

## 2026-06-15

### 09:34 — 프로젝트 전용 가상환경(.venv) 셋업
- 기존 워크스페이스 `.venv`는 Python 3.14.4 + 패키지 0개라 torch 미지원 → 사용 불가 확인.
- `ml_unified_lab/.venv`를 **Python 3.12.7**로 새로 생성하고 `requirements.txt` 전체 설치.
  (numpy 2.4.6 / pandas 3.0.3 / sklearn 1.9.0 / torch 2.12.0+cpu / streamlit 1.58.0 등)
- `requirements.txt`에 `openpyxl` 추가(online_retail .xlsx 로딩).
- `.gitignore`에 `.venv/`, `*.bak` 추가.
- **검증(.venv 파이썬)**: iris(grid, acc 1.0) · groceries(apriori, 규칙 69) · housing DNN(torch, RMSE 3.09)
  실행 성공. Streamlit `AppTest` 4탭 렌더링·예외 없음. GUI 서버를 `.venv`로 재기동(health 200).
- 참고: 전역(anaconda) 환경은 streamlit 1.37이라 `use_container_width`(현 코드)가 양쪽 모두 호환.
  `.venv`(streamlit 1.58)에선 deprecation 경고만 뜨고 정상 동작.

---

## 2026-06-14

### 23:38 — 다단계 요구사항 접수 (코드뷰어/복습기능/파라미터GUI/조합실행/문서화)
- 사용자가 9개 항목 요청(코드 뷰어·복습 기능 추천·YAML 설명서/파라미터 GUI·모든 조합 실행·
  HISTORY·RULES·리포트 폴더·Colab 가이드·시각 정확성).
- 착수 전 사실 수집: 실습 과제 파일(`archive/.../*실습문제*.txt`, `*.md`)을 읽어
  과제에서 실제로 조정했던 하이퍼파라미터를 확인.
  - 딥러닝: `epochs`, `batch_size`, `learning_rate`, `optimizer(Adam/AdamW)`,
    은닉노드 수·은닉층 수, `dropout`, `SEED`, conv 계층 수, 데이터 증강(RandomRotation/ColorJitter).
  - sklearn: Decision Tree `max_depth`/`min_samples_split`/`criterion`,
    KNN `n_neighbors`, SVM `C`/`kernel`, 평가지표/혼동행렬.
  - 연관규칙: `min_support`(0.005~0.05), confidence, lift.
- `mlxtend` 미설치 확인(연관규칙 구현 시 의존성 없이 처리 필요).

### 23:40 — RULES.md, HISTORY.md, reports/ 운영 시작
- [RULES.md](RULES.md) 작성: 할루시네이션 방지 규칙, AI 하네스 작업 기법,
  하드코딩/폴백 승인·리포트 규칙, GUI 작업 규칙, 리포트/히스토리 규칙, 요구사항 목록.
- [HISTORY.md](HISTORY.md) 작성 시작(본 파일).
- 리포트 저장소를 `reports/`로 지정(기존 인벤토리 CSV와 공존).

### 23:45 — 모든 조합 실행 가능화 (item 4)
- `models.yaml`: `svm`에 regression(SVR) 추가, `lstm` task_types를 [classification, regression, time_series]로 확장,
  `apriori`(framework: special, association_rules) 추가.
- `datasets.yaml`: `groceries`(트랜잭션 CSV) 추가.
- 연관규칙 분석 구현(외부 의존성 없음): `src/association/apriori.py`(support/confidence/lift),
  `src/datasets/transactions.py`, `src/training/association_runner.py`.
- `online_retail`(.xlsx zip) 로더 추가: 최초 1회 파싱 후 CSV 캐시.
- torch 단일 실행도 `train.*` 모델 구조 파라미터를 반영하도록 `torch_runner` 수정.
  sklearn 단일 실행도 지정 파라미터 첫 값을 반영하도록 `runner._run_sklearn` 수정.
- **검증**: groceries apriori(거래 9835, 규칙 69, 1.5s), online_retail(18338 바스켓, 규칙 180),
  svm-regression(concrete), lstm-classification(iris acc 0.80), lstm-regression(housing) 모두 실행 확인.

### 23:50 — 하이퍼파라미터 사양 단일화 + GUI 강화 (item 1·3)
- `src/param_specs.py` 신설: 모델별 탐색 파라미터·프레임워크별 train 옵션을 한 곳에 정의
  (GUI·문서 공용 단일 출처). 실습 과제에서 다룬 파라미터를 모두 포함.
- `app.py` 재작성(요청에 따른 기존 GUI 수정, RULES §4):
  - **🧩 직접 구성** 탭: YAML 직접 입력 대신 모델별 하이퍼파라미터 위젯(멀티셀렉트/슬라이더)으로 전환.
    연관규칙 데이터셋 선택 시 apriori 파라미터(min_support 등) 위젯 자동 전환.
  - **📝 코드 보기/편집** 탭 신설: 소스를 코드 블럭으로 보기, 편집·저장(.bak 백업), 파이썬 문법 검사.
  - 연관규칙 결과(규칙 표) 렌더링 추가.
- **검증**: Streamlit `AppTest`로 4개 탭 렌더링·예외 없음 확인. 직접 구성 탭에서
  iris(single_run)·groceries(apriori) 실행을 GUI 경로로 성공 확인.

### 23:55 — 리포트 작성 (item 2·3·8) 및 reports/ 운영 (item 7)
- `reports/yaml_options_guide.md`: 실험 YAML 전체 옵션·모델별 파라미터 설명서.
- `reports/colab_setup_guide.md`: Colab clone/zip 셋업, CLI 실행, cloudflared로 GUI 띄우기.
  (주의: `ml_unified_lab/`가 아직 미커밋임을 명시)
- `reports/review_feature_recommendations.md`: 복습 도움 기능 우선순위 추천.

### 23:57 — 문서 정리
- HISTORY.md(본 파일)·README 갱신. RULES.md §8에 요구사항 9건 기록 완료.

---

## (이전) 2026-06-14 — 프레임워크 최초 구축 및 딥러닝 연동

> 아래는 본 히스토리 파일 도입 이전에 진행된 내용을 사후 요약한 것이다(정확 분 단위는 git/run 폴더 타임스탬프 참고).

- 빈 `src/` 스캐폴딩에 설정 기반 실험 프레임워크 전체 구현:
  `config`, `datasets`(tabular 8 + vision 3), `models`(sklearn 8 + torch 4),
  `search`(grid/random/bayesian), `training`(runner + torch_runner), `reporting`(metrics + plots).
- CLI `run_experiment.py`(`--list`/`-c`/`--all`) 작성.
- sklearn MVP 3종 실행 검증: iris(acc 1.00), breast_cancer(f1 0.965), concrete(RMSE 4.95/R² 0.914).
- torch·torchvision·optuna 설치 후 딥러닝 경로 검증:
  housing DNN(RMSE 3.06), fashion_mnist CNN(acc 0.86), cifar10 CNN(acc 0.54). Optuna를 torch 탐색에 연동.
- Streamlit GUI(`app.py`) 추가: 프리셋 실험 / 직접 구성 / 지난 결과 탭. `localhost:8501` 동작 확인.
- `requirements.txt`, `README.md`, `.gitignore` 정비.
