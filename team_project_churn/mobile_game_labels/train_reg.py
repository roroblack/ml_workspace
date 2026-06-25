# -*- coding: utf-8 -*-
"""라벨 C(다음 활동까지 시간, 회귀) 비교: GBM(tab) vs LSTM(seq). 지표: MAE·RMSE·R2.
사용법: python train_reg.py BC target_c C_ttne"""
import os, sys, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
IDS = {"session", "user_id", "churn", "label_b", "target_c"}
subdir, target_col, label_name = sys.argv[1], sys.argv[2], sys.argv[3]
DATA = os.path.join(HERE, "data", subdir); OUT = os.path.join(HERE, "outputs")


class LSTMReg(nn.Module):
    def __init__(s, f, h=64, p=0.3):
        super().__init__(); s.lstm=nn.LSTM(f,h,batch_first=True); s.fc=nn.Sequential(nn.Dropout(p),nn.Linear(h,1))
    def forward(s,x): o,_=s.lstm(x); return s.fc(o[:,-1]).squeeze(1)


def reg_metrics(y, p):
    return {"MAE":float(mean_absolute_error(y,p)),"RMSE":float(mean_squared_error(y,p)**0.5),"R2":float(r2_score(y,p))}


def main():
    df=pd.read_csv(os.path.join(DATA,"tabular.csv"))
    feat=[c for c in df.columns if c not in IDS]
    Xtab=df[feat].astype(float).values; y=df[target_col].values.astype(np.float32)
    sp=np.load(os.path.join(DATA,"split.npz")); tr,te=sp["tr"],sp["te"]
    sc=StandardScaler().fit(Xtab[tr]); Xtab=sc.transform(Xtab)
    Xseq=np.load(os.path.join(DATA,"seq.npz"))["X"].astype(np.float32)
    smu=Xseq[tr].reshape(-1,Xseq.shape[2]).mean(0); ssd=Xseq[tr].reshape(-1,Xseq.shape[2]).std(0)+1e-8
    Xseq=(Xseq-smu)/ssd; F=Xseq.shape[2]
    print(f"[{label_name}] train {len(tr)} test {len(te)} | target 평균 {y.mean():.2f}")

    res={}
    gb=GradientBoostingRegressor(random_state=SEED).fit(Xtab[tr],y[tr])
    res["GBM(tab)"]=reg_metrics(y[te],gb.predict(Xtab[te]))

    torch.manual_seed(SEED); m=LSTMReg(F)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); crit=nn.SmoothL1Loss()
    dl=DataLoader(TensorDataset(torch.tensor(Xseq[tr]),torch.tensor(y[tr])),batch_size=512,shuffle=True)
    for _ in range(20):
        m.train()
        for xb,yb in dl:
            opt.zero_grad(set_to_none=True); crit(m(xb),yb).backward(); opt.step()
    m.eval()
    with torch.no_grad(): p=m(torch.tensor(Xseq[te])).numpy()
    res["LSTM(seq)"]=reg_metrics(y[te],p)

    for k,v in res.items(): print(f"  {k:12s} MAE {v['MAE']:.3f} RMSE {v['RMSE']:.3f} R2 {v['R2']:.3f}")
    json.dump(res,open(os.path.join(OUT,f"{label_name}_metrics.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"저장 → outputs/{label_name}_metrics.json")


if __name__=="__main__":
    main()
