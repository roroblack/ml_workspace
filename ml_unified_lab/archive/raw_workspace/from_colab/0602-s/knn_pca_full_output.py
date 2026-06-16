import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def show_eval_result(title, y_true, y_pred, class_names):
    """정확도, 혼동행렬, 분류 리포트를 한 번에 출력합니다."""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{name}" for name in class_names],
        columns=[f"Pred_{name}" for name in class_names],
    )

    print(f"\n{title}")
    print(f"정확도: {acc:.4f}")
    print("혼동행렬:")
    display(cm_df)
    print("분류 리포트:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return acc, cm, cm_df


# ============================================================
# 0. 공통 설정
# ============================================================

class_names = ["Benignant", "Malevolent"]
y_true = y_test_tensor.cpu().numpy()


# ============================================================
# 1. Min-Max 정규화 + K=23 평가
# ============================================================

knn_minmax_model = TorchKNNClassifier(k=23)
knn_minmax_model.fit(X_train_minmax_tensor, y_train_tensor)

y_pred_minmax_tensor = knn_minmax_model.predict(X_test_minmax_tensor)
y_pred_minmax = y_pred_minmax_tensor.cpu().numpy()

minmax_accuracy, minmax_cm, minmax_cm_df = show_eval_result(
    title="Min-Max 정규화 + K=23 결과",
    y_true=y_true,
    y_pred=y_pred_minmax,
    class_names=class_names,
)


# ============================================================
# 2. Z-score 표준화 + K=23 평가
# ============================================================

knn_zscore_model = TorchKNNClassifier(k=23)
knn_zscore_model.fit(X_train_zscore_tensor, y_train_tensor)

y_pred_zscore_tensor = knn_zscore_model.predict(X_test_zscore_tensor)
y_pred_zscore = y_pred_zscore_tensor.cpu().numpy()

zscore_accuracy, zscore_cm, zscore_cm_df = show_eval_result(
    title="Z-score 표준화 + K=23 결과",
    y_true=y_true,
    y_pred=y_pred_zscore,
    class_names=class_names,
)


# ============================================================
# 3. 다양한 k 값에 따른 Min-Max 성능 비교
# ============================================================

k_values = [1, 5, 11, 15, 21, 23, 27]
results = []

for k in k_values:
    model = TorchKNNClassifier(k=k)
    model.fit(X_train_minmax_tensor, y_train_tensor)

    pred_tensor = model.predict(X_test_minmax_tensor)
    pred = pred_tensor.cpu().numpy()
    acc = accuracy_score(y_true, pred)
    cm_current = confusion_matrix(y_true, pred)

    results.append(
        {
            "k": k,
            "accuracy": acc,
            "confusion_matrix": cm_current,
            "pred": pred,
        }
    )

    print(f"k={k:2d}, accuracy={acc:.4f}")

results_df = pd.DataFrame(
    [{"k": item["k"], "accuracy": item["accuracy"]} for item in results]
)

print("\nk 값별 정확도:")
display(results_df.sort_values(by="accuracy", ascending=False).reset_index(drop=True))

print("\nk 값별 혼동행렬 및 분류 리포트:")
for item in sorted(results, key=lambda x: (-x["accuracy"], x["k"])):
    cm_df = pd.DataFrame(
        item["confusion_matrix"],
        index=[f"Actual_{name}" for name in class_names],
        columns=[f"Pred_{name}" for name in class_names],
    )

    print(f"\nk={item['k']}, accuracy={item['accuracy']:.4f}")
    display(cm_df)
    print(classification_report(y_true, item["pred"], target_names=class_names))

plt.figure(figsize=(8, 5))
plt.plot(results_df["k"], results_df["accuracy"], marker="o")
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy by k Value (Min-Max)")
plt.xticks(k_values)
plt.grid(True)
plt.show()


# ============================================================
# 4. PCA x k 그리드 서치
# ============================================================

pca_dims = [2, 4, 6, 8, 10]
grid_results = []
best_acc = 0.0
best_params = None

print(f"\n그리드 서치 시작 (PCA 후보: {pca_dims}, k 후보: {k_values})")

for n in pca_dims:
    pca = PCA(n_components=n, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_zscore)
    X_test_pca = pca.transform(X_test_zscore)

    X_train_pca_tensor = torch.tensor(X_train_pca, dtype=torch.float32).to(device)
    X_test_pca_tensor = torch.tensor(X_test_pca, dtype=torch.float32).to(device)
    explained_var = float(np.sum(pca.explained_variance_ratio_))

    for k in k_values:
        model = TorchKNNClassifier(k=k)
        model.fit(X_train_pca_tensor, y_train_tensor)

        pred_tensor = model.predict(X_test_pca_tensor)
        pred_np = pred_tensor.cpu().numpy()
        acc = accuracy_score(y_true, pred_np)
        cm = confusion_matrix(y_true, pred_np)

        grid_results.append(
            {
                "pca_n": n,
                "k": k,
                "accuracy": acc,
                "explained_var": explained_var,
                "cm": cm,
                "pred": pred_np,
            }
        )

        print(
            f"PCA={n:2d}, k={k:2d}, accuracy={acc:.4f}, explained_var={explained_var:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            best_params = {
                "pca_n": n,
                "k": k,
                "accuracy": acc,
                "explained_var": explained_var,
                "cm": cm,
                "pred": pred_np,
            }


# ============================================================
# 5. PCA 그리드 서치 최적 결과 출력
# ============================================================

print(
    f"\n최적 파라미터: PCA n_components={best_params['pca_n']}, "
    f"k={best_params['k']}"
)
print(f"최고 정확도: {best_params['accuracy']:.4f}")
print(f"누적 설명분산비: {best_params['explained_var']:.4f}")

best_cm_df = pd.DataFrame(
    best_params["cm"],
    index=[f"Actual_{name}" for name in class_names],
    columns=[f"Pred_{name}" for name in class_names],
)

print("\n최적 PCA 모델 혼동행렬:")
display(best_cm_df)

print("최적 PCA 모델 분류 리포트:")
print(classification_report(y_true, best_params["pred"], target_names=class_names))


# ============================================================
# 6. 전체 결과 테이블 출력
# ============================================================

summary_df = pd.DataFrame(
    [
        {"experiment": "Min-Max + K=23", "accuracy": minmax_accuracy},
        {"experiment": "Z-score + K=23", "accuracy": zscore_accuracy},
        {
            "experiment": f"PCA({best_params['pca_n']}) + K={best_params['k']}",
            "accuracy": best_params["accuracy"],
        },
    ]
).sort_values(by="accuracy", ascending=False)

print("\n주요 실험 정확도 비교:")
display(summary_df.reset_index(drop=True))

grid_results_df = pd.DataFrame(grid_results)

print("\nPCA + k 전체 조합 결과:")
display(
    grid_results_df[["pca_n", "k", "accuracy", "explained_var"]]
    .sort_values(by="accuracy", ascending=False)
    .reset_index(drop=True)
)

pivot_df = grid_results_df.pivot(index="pca_n", columns="k", values="accuracy")
print("\nPCA 차원 및 k 값별 정확도 테이블:")
display(pivot_df)


# ============================================================
# 7. 시각화
# ============================================================

plt.figure(figsize=(10, 6))
for n in pca_dims:
    subset = grid_results_df[grid_results_df["pca_n"] == n].sort_values(by="k")
    plt.plot(subset["k"], subset["accuracy"], marker="o", label=f"PCA n={n}")

plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy by k and PCA Components")
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()
