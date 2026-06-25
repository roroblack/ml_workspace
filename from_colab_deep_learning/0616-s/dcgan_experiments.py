# -*- coding: utf-8 -*-
# DCGAN(MNIST) 성능 개선 방향 실험 — 섹션 11의 1~7 항목을 실제로 검증합니다.
# CPU 시간을 고려해 MNIST 부분집합(SUBSET)으로 여러 설정을 비교합니다.
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
import torchvision.transforms as T
import torchvision.utils as vutils

SEED = 1234
NOISE_DIM = 100
SUBSET = 10000          # 시간 절약을 위한 학습 부분집합 크기
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dcgan_experiments")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt")
RESULT = os.path.join(OUT, "metrics.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(os.cpu_count() or 4)

lf = open(LOG, "w", encoding="utf-8")
def log(m): print(m, flush=True); lf.write(str(m) + "\n"); lf.flush()


class Generator(nn.Module):
    def __init__(self, noise_dim=100, image_channels=1, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, ch * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ch * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch, image_channels, 3, 1, 1, bias=False),
            nn.Tanh())
    def forward(self, z): return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, image_channels=1, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 4, 1, 4, 1, 0, bias=False))
    def forward(self, x): return self.net(x).view(-1)


def weights_init(m):
    cn = m.__class__.__name__
    if cn.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif cn.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_loader(batch_size):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = MNIST(root=os.path.join(HERE, "mnist"), train=True, download=True, transform=tf)
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(ds), generator=g)[:SUBSET].tolist()
    ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


FIXED_NOISE = torch.randn(25, NOISE_DIM, 1, 1, generator=torch.Generator().manual_seed(7), device=device)


def run(cfg):
    name = cfg["name"]
    torch.manual_seed(SEED)
    loader = get_loader(cfg["batch_size"])
    G = Generator(NOISE_DIM, 1, cfg["ch"]).to(device); G.apply(weights_init)
    D = Discriminator(1, cfg["ch"]).to(device); D.apply(weights_init)
    crit = nn.BCEWithLogitsLoss()
    go = optim.Adam(G.parameters(), lr=cfg["g_lr"], betas=(0.5, 0.999))
    do = optim.Adam(D.parameters(), lr=cfg["d_lr"], betas=(0.5, 0.999))
    d_steps = cfg.get("d_steps", 1)

    n_params_g = sum(p.numel() for p in G.parameters())
    n_params_d = sum(p.numel() for p in D.parameters())
    hist = []
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        G.train(); D.train()
        agg = {"g": 0.0, "d": 0.0, "dr": 0.0, "df": 0.0, "n": 0}
        for real, _ in loader:
            real = real.to(device); bs = real.size(0)
            rl = torch.ones(bs, device=device); fl = torch.zeros(bs, device=device)
            # ---- D 단계 (d_steps회) ----
            for _ in range(d_steps):
                do.zero_grad(set_to_none=True)
                real_logits = D(real)
                z = torch.randn(bs, NOISE_DIM, 1, 1, device=device)
                fake = G(z)
                fake_logits = D(fake.detach())
                d_loss = crit(real_logits, rl) + crit(fake_logits, fl)
                d_loss.backward(); do.step()
            # ---- G 단계 ----
            go.zero_grad(set_to_none=True)
            z = torch.randn(bs, NOISE_DIM, 1, 1, device=device)
            fake = G(z)
            g_loss = crit(D(fake), rl)
            g_loss.backward(); go.step()
            # ---- 통계 (균형 진단용 D 확률) ----
            with torch.no_grad():
                agg["dr"] += torch.sigmoid(real_logits).mean().item() * bs
                agg["df"] += torch.sigmoid(fake_logits).mean().item() * bs
            agg["g"] += g_loss.item() * bs; agg["d"] += d_loss.item() * bs; agg["n"] += bs
        n = agg["n"]
        row = {"epoch": ep, "g_loss": agg["g"]/n, "d_loss": agg["d"]/n,
               "d_real_prob": agg["dr"]/n, "d_fake_prob": agg["df"]/n}
        hist.append(row)
        log(f"  [{name}] ep{ep}/{cfg['epochs']} g {row['g_loss']:.3f} d {row['d_loss']:.3f} "
            f"D(real) {row['d_real_prob']:.3f} D(fake) {row['d_fake_prob']:.3f}")
    elapsed = time.time() - t0
    # 최종 생성 이미지 저장
    G.eval()
    with torch.no_grad():
        imgs = G(FIXED_NOISE).detach().cpu()
    grid_path = os.path.join(OUT, f"{name}.png")
    vutils.save_image(imgs, grid_path, nrow=5, normalize=True)
    log(f"  -> [{name}] {elapsed:.0f}s | G params {n_params_g:,} D params {n_params_d:,} | 이미지 {grid_path}")
    return {"name": name, "cfg": cfg, "elapsed_sec": round(elapsed, 1),
            "g_params": n_params_g, "d_params": n_params_d, "history": hist,
            "grid": os.path.basename(grid_path)}


def main():
    log(f"device={device} threads={torch.get_num_threads()} subset={SUBSET}")
    # 참고용 실제 이미지 그리드 저장
    loader = get_loader(128)
    real0, _ = next(iter(loader))
    vutils.save_image(real0[:25], os.path.join(OUT, "real_reference.png"), nrow=5, normalize=True)

    base = dict(batch_size=128, ch=64, g_lr=2e-4, d_lr=2e-4, epochs=6, d_steps=1)
    configs = [
        {**base, "name": "C1_baseline",      "epochs": 10},                       # 항목1 기준(에포크별 추이)
        {**base, "name": "C2_batch64",       "batch_size": 64},                    # 항목2
        {**base, "name": "C3_channels128",   "ch": 128},                           # 항목3
        {**base, "name": "C4_lr1e-4",        "g_lr": 1e-4, "d_lr": 1e-4},          # 항목4
        {**base, "name": "C5_D_too_strong",  "d_steps": 2, "d_lr": 4e-4},          # 항목5,6 (감별자 과강)
    ]
    results = []
    for cfg in configs:
        log(f"\n===== {cfg['name']} =====")
        try:
            results.append(run(cfg))
            with open(RESULT, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log(f"  !! ERROR {cfg['name']}: {e}")
    log("\n완료. metrics -> " + RESULT)


if __name__ == "__main__":
    main()
    lf.close()
