# Colab에서 바로 셋업 · 테스트하는 방법

> 목적: 내 PC가 아니어도 Google Colab에서 이 프로젝트를 즉시 띄워 실험/복습한다.
> Colab은 PyTorch·sklearn이 기본 설치돼 있고 무료 GPU도 쓸 수 있어 딥러닝 실험에 유리하다.

작성: 2026-06-14 · 규칙: [RULES.md](../RULES.md)

---

## ⚠️ 먼저 알아둘 것 (현재 저장소 상태)

- 깃 원격: `https://github.com/roroblack/ml_workspace`
- **2026-06-14 기준 `ml_unified_lab/` 폴더는 아직 커밋/푸시되지 않았다.** (`git status`에서 untracked)
  따라서 **clone 방식(A)을 쓰려면 먼저 커밋·푸시가 필요**하다. 당장 올리고 싶지 않으면
  **ZIP 업로드 방식(B)** 을 쓰면 된다.
- `archive/raw_workspace/`의 CSV 데이터(heart/housing/concrete/groceries 등)는 실험에 필요하므로
  업로드/클론에 **반드시 포함**되어야 한다. (단, `online+retail.zip`은 23MB로 큼)

---

## 방식 A — GitHub에서 clone (먼저 푸시한 경우)

내 PC에서 한 번 푸시:

```powershell
cd C:\Users\playdata2\Documents\ml_workspace
git add ml_unified_lab
git commit -m "add ml_unified_lab"
git push origin <브랜치>
```

Colab 노트북 셀:

```python
!git clone https://github.com/roroblack/ml_workspace.git
%cd ml_workspace/ml_unified_lab
!pip install -q pyyaml joblib optuna streamlit   # torch/sklearn/pandas는 Colab 기본 제공
```

## 방식 B — ZIP 업로드 (푸시 없이 바로)

내 PC에서 `ml_unified_lab` 폴더를 zip으로 압축 → Colab 좌측 파일창에 업로드. 그 후:

```python
!unzip -q ml_unified_lab.zip -d /content/
%cd /content/ml_unified_lab
!pip install -q pyyaml joblib optuna streamlit
```

> 용량을 줄이려면 `runs/`와 `archive/.../online+retail.zip`을 빼고 압축해도 된다.
> (online_retail 실험만 포기하면 됨. groceries 연관규칙은 작은 CSV라 그대로 동작.)

---

## 1) CLI로 실험 실행 (가장 간단)

```python
!python run_experiment.py --list
!python run_experiment.py -c configs/experiments/iris_logistic_grid.yaml
!python run_experiment.py -c configs/experiments/fashion_mnist_cnn_optuna.yaml
```

결과는 `runs/{시간}_{이름}/`에 저장된다. 노트북에서 지표/그래프 바로 보기:

```python
import json, glob, os
from IPython.display import Image
latest = sorted(glob.glob('runs/2*'))[-1]
print(json.load(open(f'{latest}/metrics.json', encoding='utf-8'))['metrics'])
for p in glob.glob(f'{latest}/plots/*.png'):
    display(Image(p))
```

### GPU로 딥러닝 가속 (선택)

Colab 상단 메뉴: **런타임 → 런타임 유형 변경 → GPU**. 그 후 vision 실험의
`max_train_samples`/`max_test_samples` 줄을 키우거나 제거하면 전체 학습이 빨라진다.
(설정은 `configs/experiments/*.yaml` 또는 GUI에서 조정)

---

## 2) GUI(Streamlit)를 Colab에서 띄우기

Colab은 외부에서 접속할 수 없으므로 터널이 필요하다. **cloudflared** 방식 권장:

```python
!pip install -q streamlit
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
# 백그라운드로 streamlit 실행
get_ipython().system_raw('streamlit run app.py --server.port 8501 --server.headless true &')
import time; time.sleep(5)
# 공개 URL 생성 (출력된 trycloudflare.com 주소로 접속)
!./cloudflared tunnel --url http://localhost:8501
```

출력에 나오는 `https://....trycloudflare.com` 주소를 클릭하면 GUI가 열린다.

> 대안: `!npm install -g localtunnel` 후 `!npx localtunnel --port 8501`.
> 터널 방식은 세션이 끊기면 주소도 사라지니, 가벼운 복습은 **CLI 방식(1)** 이 더 편하다.

---

## 3) Colab 점검 체크리스트

- [ ] `pip install` 후 `import torch, sklearn, optuna, streamlit` 에러 없는지
- [ ] `archive/raw_workspace/...` 데이터가 실제로 올라갔는지 (`!ls archive/raw_workspace`)
- [ ] `!python run_experiment.py --list` 에 6개 실험이 보이는지
- [ ] iris 실험이 몇 초 내 끝나고 `metrics.json`이 생기는지
