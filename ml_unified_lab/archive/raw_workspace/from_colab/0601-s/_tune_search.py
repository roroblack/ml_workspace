import warnings, itertools, random, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

URL="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df=pd.read_csv(URL,sep=';'); df.columns=[c.replace(" ","_") for c in df.columns]
X=df.drop("quality",axis=1).values.astype("float32"); y=df["quality"].values.astype("float32")
# exact same sequential split as the notebook
Xtr_raw,Xte_raw=X[:3749],X[3749:]; ytr,yte=y[:3749],y[3749:]
sc=StandardScaler(); Xtr=sc.fit_transform(Xtr_raw); Xte=sc.transform(Xte_raw)

# carve a validation set from the TAIL of train (closest in distribution to the sequential test set)
val_n=600
Xt,Xv=Xtr[:-val_n],Xtr[-val_n:]; yt,yv=ytr[:-val_n],ytr[-val_n:]

ACT={"relu":nn.ReLU,"tanh":nn.Tanh,"leaky":nn.LeakyReLU,"gelu":nn.GELU}

def build(dims,act,drop):
    L=[]; prev=11
    for h in dims:
        L+=[nn.Linear(prev,h),ACT[act]()]
        if drop>0: L.append(nn.Dropout(drop))
        prev=h
    L.append(nn.Linear(prev,1)); return nn.Sequential(*L)

def run(cfg,seed=123,epochs=600,report_test=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev="cpu"
    Xt_t=torch.tensor(Xt); yt_t=torch.tensor(yt).view(-1,1)
    Xv_t=torch.tensor(Xv); yv_t=torch.tensor(yv).view(-1,1)
    ds=torch.utils.data.TensorDataset(Xt_t,yt_t)
    dl=torch.utils.data.DataLoader(ds,batch_size=cfg["bs"],shuffle=True)
    m=build(cfg["dims"],cfg["act"],cfg["drop"])
    opt=(torch.optim.AdamW if cfg["opt"]=="adamw" else torch.optim.Adam)(
        m.parameters(),lr=cfg["lr"],weight_decay=cfg.get("wd",0.0))
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',patience=8,factor=0.5)
    crit=nn.MSELoss()
    best_val=float('inf'); best_state=None; bad=0; patience=40
    for ep in range(epochs):
        m.train()
        for bx,by in dl:
            opt.zero_grad(); loss=crit(m(bx),by); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl=crit(m(Xv_t),yv_t).item()
        sch.step(vl)
        if vl<best_val-1e-5: best_val=vl; best_state={k:v.clone() for k,v in m.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=patience: break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        vp=m(Xv_t).numpy().flatten()
    val_mae=mean_absolute_error(yv,vp); val_r2=r2_score(yv,vp)
    out={"val_mae":val_mae,"val_r2":val_r2}
    if report_test:
        with torch.no_grad(): tp=m(torch.tensor(Xte)).numpy().flatten()
        out.update({"test_mse":mean_squared_error(yte,tp),"test_rmse":mean_squared_error(yte,tp)**.5,
                    "test_mae":mean_absolute_error(yte,tp),"test_r2":r2_score(yte,tp),
                    "test_corr":pearsonr(tp,yte)[0]})
    return out

# ---- search space (kept within the assignment's knobs + weight decay) ----
grid=[]
for dims in [(64,32),(128,64,32),(64,64,32),(128,64),(256,128,64),(32,16)]:
    for act in ["tanh","relu","gelu"]:
        for lr in [3e-4,1e-3,3e-3]:
            for drop in [0.0,0.1,0.2]:
                for wd in [0.0,1e-3,1e-2]:
                    grid.append({"dims":dims,"act":act,"lr":lr,"drop":drop,"wd":wd,"bs":64,"opt":"adamw"})
random.seed(0); random.shuffle(grid)
grid=grid[:90]   # random subset for speed

results=[]
for i,cfg in enumerate(grid):
    r=run(cfg)
    results.append((r["val_mae"],cfg,r))
    if i%15==0: print(f"[{i}/{len(grid)}] val_mae={r['val_mae']:.4f} r2={r['val_r2']:.3f} {cfg['dims']} {cfg['act']} lr={cfg['lr']} d={cfg['drop']} wd={cfg['wd']}")

results.sort(key=lambda x:x[0])
print("\n=== TOP 6 by validation MAE -> evaluated on TEST set ===")
for vm,cfg,r in results[:6]:
    t=run(cfg,report_test=True)
    print(f"val_mae={vm:.4f} | TEST mae={t['test_mae']:.4f} rmse={t['test_rmse']:.4f} r2={t['test_r2']:.4f} corr={t['test_corr']:.4f} || {cfg['dims']} {cfg['act']} lr={cfg['lr']} drop={cfg['drop']} wd={cfg['wd']} bs={cfg['bs']}")
