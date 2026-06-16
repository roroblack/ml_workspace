"""튜닝 가능한 하이퍼파라미터 사양 (단일 출처).

GUI 위젯 생성과 YAML 설명서 문서가 모두 이 사양을 참조한다.
여기 정의된 값은 실제 팩토리(src/models, src/training)가 받아들이는 인자와 일치해야 한다.
(임의로 늘리지 말 것 — 할루시네이션 방지: RULES.md §1)

각 항목 형식:
  (param, kind, options, default_values, help)
    kind    : "int" | "float" | "cat"
    options : cat일 때 선택지 리스트, 그 외 None
    default : 기본으로 켜둘 값들의 리스트(탐색이면 그리드, 단일이면 첫 값 사용)
"""
from __future__ import annotations

# 모델별 탐색(하이퍼파라미터) 사양 ----------------------------------------------
SEARCH_SPECS: dict[str, list[tuple]] = {
    # --- sklearn ---
    "logistic_regression": [
        ("C", "float", None, [0.1, 1.0, 10.0], "규제 강도의 역수(작을수록 강한 규제)"),
        ("solver", "cat", ["lbfgs", "liblinear", "saga"], ["lbfgs"], "최적화 알고리즘"),
    ],
    "linear_regression": [],
    "knn": [
        ("n_neighbors", "int", None, [3, 5, 7, 9], "이웃 수 K"),
        ("weights", "cat", ["uniform", "distance"], ["uniform", "distance"], "이웃 가중 방식"),
        ("p", "cat", [1, 2], [2], "거리 척도(1=맨해튼, 2=유클리드)"),
    ],
    "svm": [
        ("C", "float", None, [0.1, 1.0, 10.0], "마진-오류 trade-off"),
        ("kernel", "cat", ["rbf", "linear", "poly", "sigmoid"], ["rbf", "linear"], "커널 함수"),
        ("gamma", "cat", ["scale", "auto"], ["scale"], "커널 계수"),
    ],
    "naive_bayes": [
        ("var_smoothing", "float", None, [1e-9, 1e-8, 1e-7], "분산 안정화 항(라플라스 유사)"),
    ],
    "decision_tree": [
        ("max_depth", "int", None, [3, 5, 10], "트리 최대 깊이(과적합 제어)"),
        ("min_samples_split", "int", None, [2, 5, 10], "분할 최소 샘플 수"),
    ],
    "random_forest": [
        ("n_estimators", "int", None, [100, 200], "트리 개수"),
        ("max_depth", "int", None, [5, 10], "트리 최대 깊이"),
    ],
    "gradient_boosting": [
        ("n_estimators", "int", None, [100, 200], "부스팅 단계 수"),
        ("learning_rate", "float", None, [0.05, 0.1], "학습률"),
        ("max_depth", "int", None, [3, 5], "약한 학습기 깊이"),
    ],
    # --- torch ---
    "mlp": [
        ("lr", "float", None, [0.001, 0.01], "학습률"),
        ("hidden_units", "int", None, [64, 128, 256], "은닉 노드 수"),
        ("dropout", "float", None, [0.0, 0.2], "드롭아웃 비율"),
        ("optimizer", "cat", ["adam", "adamw", "sgd", "rmsprop"], ["adam"], "옵티마이저"),
        ("batch_size", "int", None, [32, 64], "배치 크기"),
    ],
    "dnn": [
        ("lr", "float", None, [0.001, 0.01], "학습률"),
        ("hidden_units", "int", None, [64, 128, 256], "은닉 노드 수"),
        ("depth", "int", None, [2, 3], "은닉층 수"),
        ("dropout", "float", None, [0.0, 0.2], "드롭아웃 비율"),
        ("optimizer", "cat", ["adam", "adamw", "sgd", "rmsprop"], ["adam"], "옵티마이저"),
        ("batch_size", "int", None, [32, 64], "배치 크기"),
    ],
    "cnn_basic": [
        ("lr", "float", None, [0.0005, 0.001, 0.003], "학습률(AdamW 권장 0.0003~0.001)"),
        ("optimizer", "cat", ["adam", "adamw", "sgd", "rmsprop"], ["adam", "adamw"], "옵티마이저"),
        ("batch_size", "int", None, [64, 128], "배치 크기"),
    ],
    "lstm": [
        ("lr", "float", None, [0.001, 0.01], "학습률"),
        ("hidden_units", "int", None, [32, 64, 128], "LSTM 은닉 차원"),
        ("num_layers", "int", None, [1, 2], "LSTM 층 수"),
        ("optimizer", "cat", ["adam", "adamw", "sgd", "rmsprop"], ["adam"], "옵티마이저"),
        ("batch_size", "int", None, [32, 64], "배치 크기"),
    ],
    # --- special: 연관규칙 ---
    "apriori": [
        ("min_support", "float", None, [0.005, 0.01, 0.02, 0.05], "최소 지지도"),
        ("min_confidence", "float", None, [0.2, 0.3, 0.5], "최소 신뢰도"),
        ("max_len", "int", None, [2, 3], "항목집합 최대 크기"),
    ],
}

# 프레임워크별 학습(train) 공통 옵션 ------------------------------------------
# (param, kind, default, help)  — 단일 값 위젯으로 노출
TRAIN_SPECS: dict[str, list[tuple]] = {
    "sklearn": [
        ("test_size", "float", 0.2, "검증셋 비율"),
        ("random_state", "int", 42, "랜덤 시드(SEED)"),
    ],
    "torch": [
        ("epochs", "int", 5, "학습 에폭 수"),
        ("batch_size", "int", 64, "배치 크기(탐색 미사용 시 적용)"),
        ("lr", "float", 0.001, "학습률(탐색 미사용 시 적용)"),
        ("optimizer", "cat", "adam", "옵티마이저(탐색 미사용 시 적용)"),
        ("random_state", "int", 42, "랜덤 시드(SEED)"),
    ],
    "torch_vision": [
        ("epochs", "int", 3, "학습 에폭 수"),
        ("batch_size", "int", 64, "배치 크기"),
        ("lr", "float", 0.001, "학습률"),
        ("optimizer", "cat", "adamw", "옵티마이저"),
        ("max_train_samples", "int", 8000, "CPU용 학습 서브샘플 수(전체학습=0)"),
        ("max_test_samples", "int", 2000, "평가 서브샘플 수(전체=0)"),
        ("random_state", "int", 42, "랜덤 시드(SEED)"),
    ],
    "special": [
        ("min_support", "float", 0.01, "최소 지지도"),
        ("min_confidence", "float", 0.3, "최소 신뢰도"),
        ("max_len", "int", 2, "항목집합 최대 크기"),
    ],
}

OPTIMIZERS = ["adam", "adamw", "sgd", "rmsprop"]


def search_spec(model: str) -> list[tuple]:
    return SEARCH_SPECS.get(model, [])


def train_spec_key(framework: str, dataset_kind: str) -> str:
    if framework == "torch" and dataset_kind == "vision":
        return "torch_vision"
    return framework
