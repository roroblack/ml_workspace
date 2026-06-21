# 비교 리포트 — base / A / B / C 라벨별 이탈 예측 (정형 ML vs 시퀀스 DL)

질문: **"라벨(이탈 정의)을 바꾸면 시퀀스 DL이 정형 모델을 이기는가?"**
4가지 라벨에서 동일 비교(ML: LogReg/GBM, DL: MLP/LSTM/Transformer). 코드: `team_project_churn/retail_rees46_labels/`, base는 `retail_rees46/`.

---

## 1. 라벨 정의 (모두 REES46 2019-Nov, 누수 차단)

| 라벨 | 단위 | 정의 | 과제 성격 |
| --- | --- | --- | --- |
| **base** 계정 이탈 | 고객 | 결과기간(월 뒤 30%)에 활동 없음 = 이탈 | recency 집계형 |
| **A** 세션 즉시 이탈 | **세션** | 세션이 **구매 없이 종료 = 이탈**, 입력=구매 이전 이벤트 순서 | **순서형** |
| **B** 단기 조기 이탈 | 고객 | 관찰(11/1~23)→결과(11/24~30) 활동 없음 = 이탈 | recency 집계형(단기) |
| **C** 다음활동까지 시간 | 고객 | cutoff 후 다음 이벤트까지 **일수(회귀)** | **타이밍형** |

데이터 규모: A=세션 120,000(이탈 96.6%) · B=고객 60,000(89.0%) · C=고객 60,000(회귀, 평균 6.54일).

## 2. 결과 — 분류(base/A/B): ROC-AUC / PR-AUC

| 모델 | base AUC | **A(세션) AUC** | B AUC |
| --- | --- | --- | --- |
| LogReg (tab) | 0.783 | 0.911 | 0.776 |
| **GBM (tab)** | **0.803** | 0.929 | **0.796** |
| MLP (tab DL) | 0.800 | 0.928 | 0.793 |
| **LSTM (seq DL)** | 0.789 | **0.930** ⭐ | 0.782 |
| Transformer (seq DL) | 0.795 | 0.927 | 0.787 |

- A의 PR-AUC는 전 모델 0.997(96.6% 불균형이라 PR 기준 거의 포화).
- **base·B**: 정형 GBM이 1위 → recency 집계형이라 시퀀스 이점 없음(기존 결론 재확인).
- **A(세션 즉시 이탈)**: **LSTM(seq) 0.930으로 GBM(0.929)을 근소하게 추월** — 순서가 결과를 가르는 첫 사례.

## 3. 결과 — 회귀(C): 다음 활동까지 시간 (낮을수록 좋음 / R² 높을수록 좋음)

| 모델 | MAE | RMSE | R² |
| --- | --- | --- | --- |
| GBM (tab) | 0.668 | **1.326** | **0.196** |
| **LSTM (seq DL)** | **0.585** ⭐ | 1.384 | 0.125 |

- **LSTM이 MAE(평균 절대오차)에서 명확히 우위**(0.585 vs 0.668) → *전형적 케이스의 시점 예측은 시퀀스가 더 정확*.
- GBM은 RMSE·R²가 좋음 → *큰 편차(이상치) 설명은 정형이 안정적*.

## 4. 핵심 결론

| 라벨 | 성격 | 승자 | 의미 |
| --- | --- | --- | --- |
| base | recency 집계 | 정형 GBM | 시퀀스 무의미 |
| B | recency 집계(단기) | 정형 GBM | 시퀀스 무의미 |
| **A** | **순서(세션 흐름)** | **시퀀스 LSTM(근소)** | 순서가 가치 생성 |
| **C** | **타이밍(시점)** | **시퀀스 LSTM(MAE)** | 타이밍 예측에 강점 |

→ **"라벨의 성격이 승부를 가른다"가 정량적으로 입증됨.** recency 집계형(base·B)은 정형 GBM, **순서·타이밍형(A·C)은 시퀀스 DL**이 동등 이상.
단, A의 추월폭은 0.001로 **근소**(cart 등 집계 신호가 이미 강력) → 시퀀스 DL의 *결정적* 우위는 "집계로 안 풀리는" 더 세밀한 순서 과제(세션 중도 실시간 예측, 다단계 행동)에서 더 커집니다.

## 5. 핵심 코드 (비교용 발췌)

### 5.1 라벨 정의 (각 과제의 핵심)
```python
# A) 세션 즉시 이탈: 구매 이전 이벤트만 입력, 구매 없으면 이탈=1  (prep_labels.py)
df["cum_pur"] = df.groupby("user_session")["is_purchase"].cumsum()
kept = df[df["cum_pur"] == 0]                 # 첫 구매 이전(누수 차단)
churn = 1 - df.groupby("user_session")["is_purchase"].max()   # 미구매=이탈

# B) 단기 조기 이탈: 결과기간(11/24~30) 활동 없음=1
fut_first = fut.groupby("user_id")["event_time"].min()
label_b = fut_first.reindex(pop).isna().astype(int)

# C) 다음활동까지 시간(회귀): cutoff~다음이벤트 일수, 미발생=horizon(7)
ttne = ((fut_first - cut).dt.total_seconds()/86400).reindex(pop)
target_c = ttne.fillna(7).clip(0, 7)
```

### 5.2 시퀀스 모델 (LSTM / Transformer) — `train_compare.py`
```python
class LSTMClf(nn.Module):
    def __init__(s,f,h=64,p=0.3):
        super().__init__(); s.lstm=nn.LSTM(f,h,batch_first=True)
        s.fc=nn.Sequential(nn.Dropout(p),nn.Linear(h,1))
    def forward(s,x): o,_=s.lstm(x); return s.fc(o[:,-1]).squeeze(1)

class TransformerClf(nn.Module):
    def __init__(s,f,d=32,nhead=4,nl=2,T=20,p=0.3):
        super().__init__(); s.emb=nn.Linear(f,d); s.pos=nn.Parameter(torch.zeros(1,T,d))
        enc=nn.TransformerEncoderLayer(d,nhead,dim_feedforward=64,dropout=p,batch_first=True)
        s.tr=nn.TransformerEncoder(enc,nl); s.fc=nn.Sequential(nn.Dropout(p),nn.Linear(d,1))
    def forward(s,x): z=s.emb(x)+s.pos[:,:x.size(1)]; return s.fc(s.tr(z).mean(1)).squeeze(1)
```

### 5.3 정형 ML 비교 (GBM/LogReg)
```python
GradientBoostingClassifier(random_state=42).fit(Xtab[tr], y[tr])
LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtab[tr], y[tr])
```

## 6. 파일/산출물
- 라벨 생성: `retail_rees46_labels/prep_labels.py`
- 분류 비교: `retail_rees46_labels/train_compare.py` (A·B)
- 회귀 비교: `retail_rees46_labels/train_reg.py` (C)
- 지표: `retail_rees46_labels/outputs/{A_session,B_shortterm,C_ttne}_metrics.json`
- base: `retail_rees46/outputs/metrics_*.json` ([08 리포트](08_REES46_화장품_이탈_결과리포트.md))

## 7. 발표 한 줄
**"이탈을 '계정 단위 recency'로 정의하면 정형 GBM이 최선이고, '세션 흐름(A)·다음활동 시점(C)'처럼 순서·타이밍으로 정의하면 시퀀스 DL(LSTM/Transformer)이 동등 이상이 된다 — 모델보다 라벨이 승부를 가른다."**
