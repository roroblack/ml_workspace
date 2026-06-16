"""
선형모델 과적합 가설 자동 검증 테스트.
실행: python test_overfit.py   (pytest 없이도 동작)

검증 명제:
  T1  데이터가 충분히 많으면(n>>p) 과적합 갭 ~ 0  (사용자 직관 확인)
  T2  특성 수 p 가 n 에 가까워지면 과적합 갭 폭발  (직관의 반례)
  T3  노이즈 sigma=0 이면 선형데이터에서 과적합 갭 ~ 0  (과적합=노이즈 암기)
  T4  과적합 갭은 n 단독보다 n/p 비율로 더 잘 정렬된다
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from overfit_lab import repeated, make_data, fit_eval
import numpy as np


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}  {detail}")
    return cond


results = []

# T1: n>>p 면 갭 작다
s_small = repeated(n=40, p=20, sigma=1.0)
s_large = repeated(n=5000, p=20, sigma=1.0)
results.append(check(
    "T1 데이터 많으면 과적합 소멸",
    s_large["gap_mse"][0] < 0.1 and s_large["gap_mse"][0] < s_small["gap_mse"][0],
    f"(n=40 gap={s_small['gap_mse'][0]:.3f} -> n=5000 gap={s_large['gap_mse'][0]:.3f})"))

# T2: p->n 면 갭 폭발
s_lowp = repeated(n=200, p=10, sigma=1.0)
s_highp = repeated(n=200, p=198, sigma=1.0)
results.append(check(
    "T2 특성 많으면 과적합 폭발",
    s_highp["gap_mse"][0] > 5 * s_lowp["gap_mse"][0] + 0.5,
    f"(p=10 gap={s_lowp['gap_mse'][0]:.3f} -> p=198 gap={s_highp['gap_mse'][0]:.3f})"))

# T3: sigma=0 이면 갭 ~ 0
s_noiseless = repeated(n=200, p=50, sigma=0.0)
results.append(check(
    "T3 노이즈 0 이면 과적합 0",
    abs(s_noiseless["gap_mse"][0]) < 1e-6,
    f"(sigma=0 gap={s_noiseless['gap_mse'][0]:.2e})"))

# T4: n/p 같으면 서로 다른 (n,p)라도 갭이 비슷
#     (n=100,p=20)와 (n=400,p=80) 둘 다 n/p=5 -> 갭 유사, n만 4배인데도.
sA = repeated(n=100, p=20, sigma=1.0, reps=15)   # n/p=5
sB = repeated(n=400, p=80, sigma=1.0, reps=15)   # n/p=5
sC = repeated(n=400, p=20, sigma=1.0, reps=15)   # n/p=20 (대조)
ratio_match_close = abs(sA["gap_mse"][0] - sB["gap_mse"][0]) < 0.15
ratio_drives = sC["gap_mse"][0] < sA["gap_mse"][0]
results.append(check(
    "T4 n/p 비율이 과적합을 지배",
    ratio_match_close and ratio_drives,
    f"(n/p=5: {sA['gap_mse'][0]:.3f} vs {sB['gap_mse'][0]:.3f} | n/p=20: {sC['gap_mse'][0]:.3f})"))

print("\n" + "-" * 50)
passed = sum(results)
print(f"결과: {passed}/{len(results)} 통과")
if passed != len(results):
    raise SystemExit(1)
print("모든 가설 검증 통과 ✔")
