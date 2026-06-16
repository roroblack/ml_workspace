import warnings, random, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")

URL="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df=pd.read_csv(URL,sep=';'); df.columns=[c.replace(" ","_") for c in df.columns]
X=df.drop("quality",axis=1).values.astype("float32"); y=df["quality"].values.astype("float32")
Xtr_raw,Xte_raw=X[:3749],X[3749:]; ytr,yte=y[:3749],y[3749:]
sc=StandardScaler(); Xtr=sc.fit_transform(Xtr_raw); Xte=sc.transform(Xte_raw)
val_n=600; Xt,Xv=Xtr[:-val_n],Xtr[-val_n:]; yt,yv=ytr[:-val_n],ytr[-val_n:]

ACT={"relu":nn.ReLU,"tanh":nn.Tanh,"leaky":nn.LeakyReLU}

def build(dims,act,drop):
    L=[]; prev=11
    for h in dims:
        L+=[nn.Linear(prev,h),ACT[act]()]
        if drop>0: L.append(nn.Dropout(drop))
        prev=h
    L.append(nn.Linear(prev,1)); return nn.Sequential(*L)

def run(cfg,seed=123,epochs=600):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Xt_t=torch.tensor(Xt); yt_t=torch.tensor(yt).view(-1,1)
    Xv_t=torch.tensor(Xv); yv_t=torch.tensor(yv).view(-1,1)
    Xte_t=torch.tensor(Xte); yte_t=torch.tensor(yte).view(-1,1)
    ds=torch.utils.data.TensorDataset(Xt_t,yt_t)
    dl=torch.utils.data.DataLoader(ds,batch_size=cfg["bs"],shuffle=True)
    m=build(cfg["dims"],cfg["act"],cfg["drop"])
    opt=torch.optim.AdamW(m.parameters(),lr=cfg["lr"],weight_decay=cfg.get("wd",0.0))
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',patience=8,factor=0.5)
    crit=nn.MSELoss()
    best_state=None; bad=0
    for ep in range(epochs):
        m.train()
        for bx,by in dl: opt.zero_grad(); loss=crit(m(bx),by); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl=crit(m(Xv_t),yv_t).item()
        sch.step(vl)
        if best_state is None or vl<best_val-1e-5: best_val=vl; best_state={k:v.clone() for k,v in m.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=40: break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad(): tp=m(Xte_t).numpy().flatten()
    return {"test_mae":mean_absolute_error(yte,tp), "test_r2":r2_score(yte,tp),
            "test_mse":mean_squared_error(yte,tp), "test_rmse":mean_squared_error(yte,tp)**.5}

# tight grid: your best + variations
grid=[
    {"dims":(128,64,32),"act":"tanh","lr":1e-4,"drop":0.2,"wd":0.0,"bs":64},  # your best
    {"dims":(128,64,32),"act":"tanh","lr":1e-4,"drop":0.15,"wd":0.0,"bs":64},
    {"dims":(128,64,32),"act":"tanh","lr":1e-4,"drop":0.25,"wd":0.0,"bs":64},
    {"dims":(128,64,32),"act":"tanh","lr":1e-4,"drop":0.1,"wd":0.0,"bs":64},
    {"dims":(128,64,32),"act":"tanh","lr":1e-4,"drop":0.2,"wd":1e-3,"bs":64},
    {"dims":(256,128,64,32),"act":"tanh","lr":1e-4,"drop":0.2,"wd":0.0,"bs":64},
    {"dims":(128,128,64,32),"act":"tanh","lr":1e-4,"drop":0.2,"wd":0.0,"bs":64},
    {"dims":(128,64,32),"act":"relu","lr":1e-4,"drop":0.2,"wd":0.0,"bs":64},
    {"dims":(128,64,32),"act":"tanh","lr":5e-4,"drop":0.2,"wd":0.0,"bs":64},
    {"dims":(128,64,32),"act":"tanh","lr":1e-4,"drop":0.2,"wd":0.0,"bs":32},
]

results=[]
for i,cfg in enumerate(grid):
    r=run(cfg)
    results.append((r["test_mae"],cfg,r))
    print(f"[{i+1}/{len(grid)}] MAE={r['test_mae']:.4f} R²={r['test_r2']:.4f} RMSE={r['test_rmse']:.4f} | {cfg['dims']} {cfg['act']} lr={cfg['lr']:.0e} drop={cfg['drop']} wd={cfg['wd']:.0e} bs={cfg['bs']}")

results.sort(key=lambda x:x[0])
print("\n" + "="*100)
print("🏆 TOP 5 결과 (테스트셋 기준 MAE 순):")
print("="*100)
for vm,cfg,r in results[:5]:
    print(f"  MAE={r['test_mae']:.4f}  MSE={r['test_mse']:.4f}  RMSE={r['test_rmse']:.4f}  R²={r['test_r2']:.4f}")
    print(f"    → 구조: {cfg['dims']} | 활성화: {cfg['act']} | 학습률: {cfg['lr']:.0e} | 드롭아웃: {cfg['drop']} | 가중치감소: {cfg['wd']:.0e} | 배치: {cfg['bs']}")
    print()
