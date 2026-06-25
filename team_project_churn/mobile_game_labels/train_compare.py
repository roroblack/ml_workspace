# -*- coding: utf-8 -*-
"""분류 라벨(A 세션이탈 / B 단기이탈) 비교: ML(LogReg/GBM) + DL(MLP/LSTM/Transformer).
사용법: python train_compare.py <subdir> <target_col> <label_name>
  예) python train_compare.py A  churn   A_session
      python train_compare.py BC label_b B_shortterm
ML과 DL이 동일 분할(split.npz)을 사용. 지표: AUC·PR-AUC·Recall·F1·Acc."""
import os, sys, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
dev = torch.device("cpu")
IDS = {"session", "user_id", "churn", "label_b", "target_c"}

subdir, target_col, label_name = sys.argv[1], sys.argv[2], sys.argv[3]
DATA = os.path.join(HERE, "data", subdir); OUT = os.path.join(HERE, "outputs")


class MLP(nn.Module):
    def __init__(s, d, h=(64, 32), p=0.3):
        super().__init__(); L=[]; prev=d
        for u in h: L+=[nn.Linear(prev,u),nn.ReLU(),nn.BatchNorm1d(u),nn.Dropout(p)]; prev=u
        L+=[nn.Linear(prev,1)]; s.net=nn.Sequential(*L)
    def forward(s,x): return s.net(x).squeeze(1)

class LSTMClf(nn.Module):
    def __init__(s,f,h=64,p=0.3):
        super().__init__(); s.lstm=nn.LSTM(f,h,batch_first=True); s.fc=nn.Sequential(nn.Dropout(p),nn.Linear(h,1))
    def forward(s,x): o,_=s.lstm(x); return s.fc(o[:,-1]).squeeze(1)

class TransformerClf(nn.Module):
    def __init__(s,f,d=32,nhead=4,nl=2,T=20,p=0.3):
        super().__init__(); s.emb=nn.Linear(f,d); s.pos=nn.Parameter(torch.zeros(1,T,d))
        enc=nn.TransformerEncoderLayer(d,nhead,dim_feedforward=64,dropout=p,batch_first=True)
        s.tr=nn.TransformerEncoder(enc,nl); s.fc=nn.Sequential(nn.Dropout(p),nn.Linear(d,1))
    def forward(s,x): z=s.emb(x)+s.pos[:,:x.size(1)]; return s.fc(s.tr(z).mean(1)).squeeze(1)


def mets(yte, proba):
    pred=(proba>=0.5).astype(int)
    return {"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),
            "recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0),
            "roc_auc":roc_auc_score(yte,proba),"pr_auc":average_precision_score(yte,proba)}


def train_torch(model, Xtr, ytr, Xte, yte, epochs=15, bs=512, lr=1e-3):
    torch.manual_seed(SEED); model=model.to(dev)
    pw=torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)],dtype=torch.float32)
    crit=nn.BCEWithLogitsLoss(pos_weight=pw); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    dl=DataLoader(TensorDataset(torch.tensor(Xtr,dtype=torch.float32),torch.tensor(ytr,dtype=torch.float32)),
                  batch_size=bs,shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb,yb in dl:
            opt.zero_grad(set_to_none=True); crit(model(xb),yb).backward(); opt.step()
    model.eval()
    with torch.no_grad(): proba=torch.sigmoid(model(torch.tensor(Xte,dtype=torch.float32))).numpy()
    return proba


def main():
    df=pd.read_csv(os.path.join(DATA,"tabular.csv"))
    feat=[c for c in df.columns if c not in IDS]
    Xtab=df[feat].astype(float).values; y=df[target_col].values.astype(int)
    sp=np.load(os.path.join(DATA,"split.npz")); tr,te=sp["tr"],sp["te"]
    sc=StandardScaler().fit(Xtab[tr]); Xtab=sc.transform(Xtab)
    Xseq=np.load(os.path.join(DATA,"seq.npz"))["X"].astype(np.float32)
    smu=Xseq[tr].reshape(-1,Xseq.shape[2]).mean(0); ssd=Xseq[tr].reshape(-1,Xseq.shape[2]).std(0)+1e-8
    Xseq=(Xseq-smu)/ssd; F=Xseq.shape[2]
    print(f"[{label_name}] train {len(tr)} test {len(te)} | 양성률 {y[tr].mean():.3f} | seq {Xseq.shape}")

    res={}
    lr_=LogisticRegression(max_iter=2000,class_weight="balanced",random_state=SEED).fit(Xtab[tr],y[tr])
    res["LogReg(tab)"]=mets(y[te],lr_.predict_proba(Xtab[te])[:,1])
    gb=GradientBoostingClassifier(random_state=SEED).fit(Xtab[tr],y[tr])
    res["GBM(tab)"]=mets(y[te],gb.predict_proba(Xtab[te])[:,1])
    res["MLP(tab)"]=mets(y[te],train_torch(MLP(Xtab.shape[1]),Xtab[tr],y[tr],Xtab[te],y[te]))
    res["LSTM(seq)"]=mets(y[te],train_torch(LSTMClf(F),Xseq[tr],y[tr],Xseq[te],y[te]))
    res["Transformer(seq)"]=mets(y[te],train_torch(TransformerClf(F,T=Xseq.shape[1]),Xseq[tr],y[tr],Xseq[te],y[te]))
    for k,v in res.items(): print(f"  {k:18s} AUC {v['roc_auc']:.3f} PR-AUC {v['pr_auc']:.3f} Recall {v['recall']:.3f} F1 {v['f1']:.3f}")
    json.dump(res,open(os.path.join(OUT,f"{label_name}_metrics.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"저장 → outputs/{label_name}_metrics.json")


if __name__=="__main__":
    main()
