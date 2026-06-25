# -*- coding: utf-8 -*-
# 풀링 수정이 실제 IMDB에서 80%를 넘는지 빠르게 검증합니다. (tensorflow 없이 npz 직접 로딩)
import os, time, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

SEED=111; NUM_WORDS=10000; DATA_DIR=os.path.expanduser("~/.keras/datasets")
torch.manual_seed(SEED); torch.set_num_threads(os.cpu_count() or 4)
LOG=os.path.join(os.path.dirname(os.path.abspath(__file__)),"validate_pooling_log.txt")
lf=open(LOG,"w",encoding="utf-8")
def log(m): print(m,flush=True); lf.write(str(m)+"\n"); lf.flush()

def load_imdb(num_words,start=1,oov=2,idx_from=3,seed=113):
    with np.load(os.path.join(DATA_DIR,"imdb.npz"),allow_pickle=True) as f:
        xtr,ytr=f["x_train"],f["y_train"]; xte,yte=f["x_test"],f["y_test"]
    rng=np.random.RandomState(seed)
    i=np.arange(len(xtr)); rng.shuffle(i); xtr,ytr=xtr[i],ytr[i]
    i=np.arange(len(xte)); rng.shuffle(i); xte,yte=xte[i],yte[i]
    xs=np.concatenate([xtr,xte]); ys=np.concatenate([ytr,yte])
    xs=[[start]+[w+idx_from for w in x] for x in xs]
    xs=[[w if w<num_words else oov for w in x] for x in xs]
    s=len(xtr); return (xs[:s],ys[:s]),(xs[s:],ys[s:])

def pad(seqs,maxlen):
    out=np.zeros((len(seqs),maxlen),dtype=np.int64)
    for i,s in enumerate(seqs):
        if s: t=s[:maxlen]; out[i,:len(t)]=t
    return out

class DS(Dataset):
    def __init__(s,t,l): s.t=torch.tensor(t,dtype=torch.long); s.l=torch.tensor(np.asarray(l),dtype=torch.float32)
    def __len__(s): return len(s.t)
    def __getitem__(s,i): return s.t[i],s.l[i]

class M(nn.Module):
    def __init__(s,nw,e,h,nl,dp,bi,pool):
        super().__init__(); s.pool=pool
        s.emb=nn.Embedding(nw,e,padding_idx=0)
        s.lstm=nn.LSTM(e,h,num_layers=nl,batch_first=True,bidirectional=bi,dropout=dp if nl>1 else 0.0)
        s.dp=nn.Dropout(dp); fd=h*(2 if bi else 1)
        s.fc=nn.Linear(fd*(2 if pool=="meanmax" else 1),1)
    def forward(s,x):
        mask=(x!=0); o,_=s.lstm(s.emb(x))
        if s.pool=="last": feat=o[:,-1,:]
        else:
            m=mask.unsqueeze(-1).float()
            if s.pool=="mean": feat=(o*m).sum(1)/m.sum(1).clamp(min=1.0)
            elif s.pool=="max": feat=torch.nan_to_num(o.masked_fill(~mask.unsqueeze(-1),float("-inf")).max(1).values,neginf=0.0)
            else:
                mean=(o*m).sum(1)/m.sum(1).clamp(min=1.0)
                mx=torch.nan_to_num(o.masked_fill(~mask.unsqueeze(-1),float("-inf")).max(1).values,neginf=0.0)
                feat=torch.cat([mean,mx],1)
        return s.fc(s.dp(feat)).squeeze(1)

def run(pool, maxlen=200, e=128, h=128, nl=1, bi=True, lr=1e-3, epochs=6):
    (xtr,ytr),(xte,yte)=RAW
    torch.manual_seed(SEED)
    Xtr=pad(xtr,maxlen); Xte=pad(xte,maxlen)
    tl=DataLoader(DS(Xtr,ytr),batch_size=128,shuffle=True)
    vl=DataLoader(DS(Xte,yte),batch_size=256)
    m=M(NUM_WORDS,e,h,nl,0.3,bi,pool)
    crit=nn.BCEWithLogitsLoss(); opt=optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    sch=optim.lr_scheduler.StepLR(opt,step_size=3,gamma=0.5)
    best=0.0
    for ep in range(epochs):
        m.train()
        for x,y in tl:
            opt.zero_grad(set_to_none=True); crit(m(x),y).backward(); opt.step()
        sch.step(); m.eval(); c=t=0; P=[]; L=[]
        with torch.no_grad():
            for x,y in vl:
                p=(torch.sigmoid(m(x))>=0.5).float(); c+=(p==y).sum().item(); t+=y.size(0)
                P.append(p); L.append(y)
        acc=c/t; f1=f1_score(torch.cat(L).numpy().astype(int),torch.cat(P).numpy().astype(int))
        best=max(best,acc)
        log(f"  [{pool}] ep{ep+1}/{epochs} test_acc {acc*100:.2f}% f1 {f1*100:.2f}%")
    log(f"  -> [{pool}] best test_acc {best*100:.2f}%")
    return best

log(f"threads={torch.get_num_threads()} 로딩...")
RAW=load_imdb(NUM_WORDS)
log("학습 시작 (bi=True, embed=128, hidden=128, MAX_LEN=200, lr=1e-3, 6ep)")
for pool in ["mean","meanmax","last"]:
    t0=time.time(); run(pool); log(f"     ({pool} {time.time()-t0:.0f}s)")
log("완료")
lf.close()
