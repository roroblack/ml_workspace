"""
선형모델에서의 과적합 검증 실험 (A~F)
- 데이터 진짜 관계가 완전한 선형(y = X·w + ε)인 상황에서
  활성함수/다항특성 없는 순수 OLS가 언제 과적합하는지 정량 측정.

실행: python overfit_lab.py
산출: 콘솔 표 + results/*.png + results/summary.json

이 파일은 import 해도 안전하다(실험은 main()에서만 실행). test_overfit.py 가 재사용.
"""
import json
import os
import sys

# Windows 콘솔(cp949)에서 유니코드 출력이 깨지지 않도록
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 파일 저장 전용
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

RNG = np.random.default_rng(42)


def make_data(n, p, sigma, n_informative=5, rng=None):
    """진짜로 선형인 데이터.
    전체 p개 특성 중 앞쪽 n_informative개만 실제 신호(가중치 != 0),
    나머지는 '쓸모없는 노이즈 특성'(진짜 가중치 0).
    => 특성을 늘리는 게 곧 '불필요한 자유도'를 늘리는 것.
    """
    rng = rng or RNG
    X = rng.standard_normal((n, p))
    w = np.zeros(p)
    k = min(n_informative, p)
    w[:k] = rng.standard_normal(k) * 2.0  # 신호 크기 고정
    eps = rng.standard_normal(n) * sigma
    y = X @ w + eps
    return X, y, w


def fit_eval(Xtr, ytr, Xte, yte, model=None):
    model = model or LinearRegression()
    model.fit(Xtr, ytr)
    tr = float(np.mean((model.predict(Xtr) - ytr) ** 2))
    te = float(np.mean((model.predict(Xte) - yte) ** 2))

    def r2(X, y):
        yhat = model.predict(X)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
        return 1.0 - ss_res / ss_tot
    return dict(train_mse=tr, test_mse=te,
                train_r2=float(r2(Xtr, ytr)), test_r2=float(r2(Xte, yte)),
                gap_mse=te - tr, gap_r2=float(r2(Xtr, ytr)) - float(r2(Xte, yte)))


def repeated(n, p, sigma, reps=20, model_factory=None, n_te=4000):
    """여러 번 반복해서 평균±표준편차."""
    keys = ["train_mse", "test_mse", "train_r2", "test_r2", "gap_mse", "gap_r2"]
    acc = {k: [] for k in keys}
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        Xtr, ytr, w = make_data(n, p, sigma, rng=rng)
        Xte = rng.standard_normal((n_te, p))
        eps = rng.standard_normal(n_te) * sigma
        yte = Xte @ w + eps
        model = model_factory() if model_factory else LinearRegression()
        res = fit_eval(Xtr, ytr, Xte, yte, model)
        for k in keys:
            acc[k].append(res[k])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    summary = {}

    # -----------------------------------------------------------------------
    # 실험 A: p 고정, n 변화  (사용자 직관: 데이터 많으면 과적합 사라진다)
    # -----------------------------------------------------------------------
    banner("실험 A | p=20 고정, sigma=1.0 고정, n 변화  -> 데이터 양의 효과")
    A_p, A_sigma = 20, 1.0
    A_ns = [30, 50, 100, 500, 5000]
    A = []
    print(f"{'n':>6} {'n/p':>6} {'train_mse':>10} {'test_mse':>10} {'gap_mse':>10} {'test_R2':>8}")
    for n in A_ns:
        s = repeated(n, A_p, A_sigma)
        A.append((n, s))
        print(f"{n:>6} {n/A_p:>6.1f} {s['train_mse'][0]:>10.3f} {s['test_mse'][0]:>10.3f} "
              f"{s['gap_mse'][0]:>10.3f} {s['test_r2'][0]:>8.3f}")
    summary["A"] = {"p": A_p, "sigma": A_sigma,
                    "rows": [{"n": n, "n_over_p": n / A_p, **{k: v[0] for k, v in s.items()}} for n, s in A]}

    # -----------------------------------------------------------------------
    # 실험 B: n 고정, p 변화  (반례: 특성 늘리면 과적합 폭발)
    # -----------------------------------------------------------------------
    banner("실험 B | n=200 고정, sigma=1.0 고정, p 변화  -> 특성 수의 효과")
    B_n, B_sigma = 200, 1.0
    B_ps = [5, 50, 150, 190, 198]
    B = []
    print(f"{'p':>6} {'n/p':>6} {'train_mse':>10} {'test_mse':>10} {'gap_mse':>10} {'test_R2':>8}")
    for p in B_ps:
        s = repeated(B_n, p, B_sigma)
        B.append((p, s))
        print(f"{p:>6} {B_n/p:>6.2f} {s['train_mse'][0]:>10.3f} {s['test_mse'][0]:>10.3f} "
              f"{s['gap_mse'][0]:>10.3f} {s['test_r2'][0]:>8.3f}")
    summary["B"] = {"n": B_n, "sigma": B_sigma,
                    "rows": [{"p": p, "n_over_p": B_n / p, **{k: v[0] for k, v in s.items()}} for p, s in B]}

    # -----------------------------------------------------------------------
    # 실험 C: 여러 (n,p) 조합을 n/p 비율로 통합
    # -----------------------------------------------------------------------
    banner("실험 C | 다양한 (n,p) -> 과적합 갭이 n/p 비율 하나로 정렬되는가")
    C_sigma = 1.0
    C_combos = []
    for n in [100, 200, 400]:
        for p in [10, 25, 50, 80, 120, int(n * 0.95)]:
            if 1 <= p < n:
                C_combos.append((n, p))
    C = []
    for n, p in C_combos:
        s = repeated(n, p, C_sigma, reps=12)
        C.append((n, p, n / p, s["gap_r2"][0], s["test_r2"][0]))
    C.sort(key=lambda r: r[2])
    print(f"{'n':>5} {'p':>5} {'n/p':>6} {'gap_R2':>8} {'test_R2':>8}")
    for n, p, ratio, gap, te in C:
        print(f"{n:>5} {p:>5} {ratio:>6.2f} {gap:>8.3f} {te:>8.3f}")
    summary["C"] = {"sigma": C_sigma,
                    "rows": [{"n": n, "p": p, "n_over_p": r, "gap_r2": g, "test_r2": t}
                             for n, p, r, g, t in C]}

    # -----------------------------------------------------------------------
    # 실험 D: 노이즈 강도의 효과 (sigma=0 이면 과적합 0 이어야)
    # -----------------------------------------------------------------------
    banner("실험 D | n=200, p=50 고정, sigma 변화  -> 노이즈 강도의 효과")
    D_n, D_p = 200, 50
    D_sigmas = [0.0, 0.1, 1.0, 5.0]
    D = []
    print(f"{'sigma':>6} {'sigma^2':>8} {'train_mse':>10} {'test_mse':>10} {'gap_mse':>10}")
    for sg in D_sigmas:
        s = repeated(D_n, D_p, sg)
        D.append((sg, s))
        print(f"{sg:>6.2f} {sg**2:>8.3f} {s['train_mse'][0]:>10.4f} {s['test_mse'][0]:>10.4f} "
              f"{s['gap_mse'][0]:>10.4f}")
    summary["D"] = {"n": D_n, "p": D_p,
                    "rows": [{"sigma": sg, "sigma2": sg**2, **{k: v[0] for k, v in s.items()}} for sg, s in D]}

    # -----------------------------------------------------------------------
    # 실험 E: Double Descent (보간 임계점 p=n 에서 테스트오차 폭발)
    # -----------------------------------------------------------------------
    banner("실험 E | Double Descent: n=100 고정, p 를 1..200 스윕 (p>n 은 최소노름 해)")
    E_n, E_sigma, E_D = 100, 1.0, 200
    E_ps = sorted(set([1, 5, 10, 20, 40, 60, 80, 90, 95, 99, 100, 101, 105, 110,
                       120, 140, 160, 180, 200]))
    E_reps = 30
    E = []
    for p in E_ps:
        tr_list, te_list = [], []
        for r in range(E_reps):
            rng = np.random.default_rng(5000 + r)
            Xfull = rng.standard_normal((E_n, E_D))
            w = rng.standard_normal(E_D) / np.sqrt(E_D)
            ytr = Xfull @ w + rng.standard_normal(E_n) * E_sigma
            Xtr = Xfull[:, :p]
            beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)  # 최소노름 해 (p>n 에서도 동작)
            Xte_full = rng.standard_normal((4000, E_D))
            yte = Xte_full @ w + rng.standard_normal(4000) * E_sigma
            Xte = Xte_full[:, :p]
            tr_list.append(np.mean((Xtr @ beta - ytr) ** 2))
            te_list.append(np.mean((Xte @ beta - yte) ** 2))
        E.append((p, float(np.mean(tr_list)), float(np.median(te_list))))
    print(f"{'p':>5} {'p/n':>6} {'train_mse':>10} {'test_mse(med)':>14}")
    for p, tr, te in E:
        print(f"{p:>5} {p/E_n:>6.2f} {tr:>10.4f} {te:>14.4f}")
    summary["E"] = {"n": E_n, "sigma": E_sigma, "D": E_D,
                    "rows": [{"p": p, "p_over_n": p / E_n, "train_mse": tr, "test_mse": te}
                             for p, tr, te in E]}

    # -----------------------------------------------------------------------
    # 실험 F: 정규화(Ridge) 대조 — 데이터 못 늘릴 때의 해법
    # -----------------------------------------------------------------------
    banner("실험 F | 실험 B 조건에서 OLS vs Ridge(alpha=10)  -> 정규화의 과적합 억제")
    F_n, F_sigma = 200, 1.0
    F_ps = [50, 150, 190, 198]
    F = []
    print(f"{'p':>6} {'OLS test_R2':>12} {'Ridge test_R2':>14} {'개선':>8}")
    for p in F_ps:
        s_ols = repeated(F_n, p, F_sigma, reps=12)
        s_rdg = repeated(F_n, p, F_sigma, reps=12, model_factory=lambda: Ridge(alpha=10.0))
        F.append((p, s_ols, s_rdg))
        print(f"{p:>6} {s_ols['test_r2'][0]:>12.3f} {s_rdg['test_r2'][0]:>14.3f} "
              f"{s_rdg['test_r2'][0]-s_ols['test_r2'][0]:>8.3f}")
    summary["F"] = {"n": F_n, "sigma": F_sigma,
                    "rows": [{"p": p, "ols_test_r2": o["test_r2"][0], "ridge_test_r2": r["test_r2"][0]}
                             for p, o, r in F]}

    # =======================================================================
    # 그래프
    # =======================================================================
    plt.figure(figsize=(7, 5))
    ns = [n for n, _ in A]
    plt.plot(ns, [s["train_mse"][0] for _, s in A], "o-", label="train MSE")
    plt.plot(ns, [s["test_mse"][0] for _, s in A], "s-", label="test MSE")
    plt.plot(ns, [s["gap_mse"][0] for _, s in A], "^--", label="gap (test-train)")
    plt.axhline(A_sigma**2, color="gray", ls=":", label=f"noise floor sigma^2={A_sigma**2}")
    plt.xscale("log")
    plt.xlabel("n (samples, log scale)"); plt.ylabel("MSE")
    plt.title(f"Exp A: more data kills overfitting  (p={A_p} fixed)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "A_data_amount.png"), dpi=110); plt.close()

    plt.figure(figsize=(7, 5))
    ps = [p for p, _ in B]
    plt.plot(ps, [s["train_mse"][0] for _, s in B], "o-", label="train MSE")
    plt.plot(ps, [s["test_mse"][0] for _, s in B], "s-", label="test MSE")
    plt.plot(ps, [s["gap_mse"][0] for _, s in B], "^--", label="gap (test-train)")
    plt.axvline(B_n, color="red", ls=":", label=f"p=n={B_n}")
    plt.xlabel("p (features)"); plt.ylabel("MSE")
    plt.title(f"Exp B: features cause overfitting  (n={B_n} fixed)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "B_feature_count.png"), dpi=110); plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter([r[2] for r in C], [r[3] for r in C], c=[r[0] for r in C], cmap="viridis", s=60)
    plt.colorbar(label="n (samples)")
    plt.xlabel("n / p  ratio"); plt.ylabel("overfitting gap (train_R2 - test_R2)")
    plt.title("Exp C: overfitting collapses onto n/p ratio")
    plt.xscale("log"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "C_ratio_collapse.png"), dpi=110); plt.close()

    plt.figure(figsize=(7, 5))
    sgs = [sg for sg, _ in D]
    plt.plot([sg**2 for sg in sgs], [s["gap_mse"][0] for _, s in D], "^-", label="gap (test-train)")
    plt.xlabel("noise variance  sigma^2"); plt.ylabel("overfitting gap (MSE)")
    plt.title(f"Exp D: overfitting gap scales with noise  (n={D_n}, p={D_p})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "D_noise.png"), dpi=110); plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot([p / E_n for p, _, _ in E], [tr for _, tr, _ in E], "o-", label="train MSE")
    plt.plot([p / E_n for p, _, _ in E], [te for _, _, te in E], "s-", label="test MSE (median)")
    plt.axvline(1.0, color="red", ls=":", label="interpolation threshold p=n")
    plt.ylim(0, min(8, max(te for _, _, te in E) * 1.1))
    plt.xlabel("p / n"); plt.ylabel("MSE")
    plt.title("Exp E: Double Descent in a linear model")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "E_double_descent.png"), dpi=110); plt.close()

    plt.figure(figsize=(7, 5))
    fps = [p for p, _, _ in F]
    plt.plot(fps, [o["test_r2"][0] for _, o, _ in F], "o-", label="OLS test R2")
    plt.plot(fps, [r["test_r2"][0] for _, _, r in F], "s-", label="Ridge test R2")
    plt.xlabel("p (features)"); plt.ylabel("test R2")
    plt.title(f"Exp F: Ridge tames overfitting  (n={F_n} fixed)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "F_ridge.png"), dpi=110); plt.close()

    with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    banner("완료: results/ 에 그래프 6개 + summary.json 저장됨")
    print("그래프:", ", ".join(sorted(os.listdir(RESULTS))))
    return summary


if __name__ == "__main__":
    main()
