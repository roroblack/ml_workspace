"""torch 모델 팩토리 (torch 필요).

MLP/DNN(tabular), CNN(image), LSTM(time series) 정의.
torch가 없으면 import 시점이 아니라 build 시점에 명확한 에러를 던진다.
"""
from __future__ import annotations


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except ImportError as e:  # pragma: no cover - 환경 의존
        raise ImportError(
            "torch 모델은 PyTorch가 필요합니다. `pip install torch` 후 다시 실행하세요."
        ) from e


def build_mlp(input_dim: int, output_dim: int, *, hidden_units=128, depth=2,
              dropout=0.0, task="classification"):
    """간단한 MLP/DNN (tabular)."""
    _require_torch()
    import torch.nn as nn

    layers: list = []
    in_dim = input_dim
    hidden = [hidden_units] * depth if isinstance(hidden_units, int) else list(hidden_units)
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        in_dim = h
    out = output_dim if task == "classification" else 1
    layers.append(nn.Linear(in_dim, out))
    return nn.Sequential(*layers)


def build_cnn(input_shape, n_classes: int):
    """기본 CNN (image_classification). input_shape=(C,H,W)."""
    _require_torch()
    import torch.nn as nn

    c, h, w = input_shape

    class BasicCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            flat = 64 * (h // 4) * (w // 4)
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(),
                nn.Dropout(0.25), nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return BasicCNN()


def build_lstm(input_dim: int, output_dim: int, *, hidden_units=64, num_layers=1,
               task="classification"):
    """기본 LSTM (time series)."""
    _require_torch()
    import torch.nn as nn

    class BasicLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_units, num_layers, batch_first=True)
            out = output_dim if task == "classification" else 1
            self.head = nn.Linear(hidden_units, out)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (B, F) -> (B, 1, F)
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return BasicLSTM()


def build_rnn(input_dim: int, output_dim: int, *, hidden_units=64, num_layers=1,
              task="classification"):
    """기본 RNN (time series)."""
    _require_torch()
    import torch.nn as nn

    class BasicRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(input_dim, hidden_units, num_layers, batch_first=True, nonlinearity="tanh")
            out = output_dim if task == "classification" else 1
            self.head = nn.Linear(hidden_units, out)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    return BasicRNN()


def build_gru(input_dim: int, output_dim: int, *, hidden_units=64, num_layers=1,
              task="classification"):
    """기본 GRU (time series)."""
    _require_torch()
    import torch.nn as nn

    class BasicGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_units, num_layers, batch_first=True)
            out = output_dim if task == "classification" else 1
            self.head = nn.Linear(hidden_units, out)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])

    return BasicGRU()


def build_attention_lstm(input_dim: int, output_dim: int, *, hidden_units=64,
                         dropout=0.0, task="classification"):
    """LSTM + Attention 풀링 (모든 시점을 가중합)."""
    _require_torch()
    import torch
    import torch.nn as nn

    class AttnLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_units, batch_first=True)
            self.attn = nn.Linear(hidden_units, 1)
            self.drop = nn.Dropout(dropout)
            out = output_dim if task == "classification" else 1
            self.head = nn.Linear(hidden_units, out)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.lstm(x)                                   # (B,T,H)
            w = torch.softmax(self.attn(out).squeeze(-1), dim=1)    # (B,T)
            ctx = (out * w.unsqueeze(-1)).sum(dim=1)                # (B,H)
            return self.head(self.drop(ctx))

    return AttnLSTM()


def build_transformer(input_dim: int, output_dim: int, *, d_model=32, nhead=4,
                      num_layers=2, dim_feedforward=64, dropout=0.1, max_len=512,
                      task="classification"):
    """Transformer Encoder 기반 시퀀스 분류기 (mean pooling)."""
    _require_torch()
    import torch.nn as nn

    class TransformerClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Linear(input_dim, d_model)
            self.pos = nn.Parameter(__import__("torch").zeros(1, max_len, d_model))
            layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
            self.tr = nn.TransformerEncoder(layer, num_layers)
            out = output_dim if task == "classification" else 1
            self.head = nn.Linear(d_model, out)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            T = x.size(1)
            z = self.emb(x) + self.pos[:, :T, :]
            z = self.tr(z)
            return self.head(z.mean(dim=1))

    return TransformerClf()


def build_torch_model(name: str, bundle, params: dict | None = None):
    """레지스트리 이름 기반 torch 모델 생성."""
    params = params or {}
    task = bundle.task

    if bundle.kind == "vision" or name == "cnn_basic":
        return build_cnn(bundle.input_shape, bundle.n_classes)

    input_dim = bundle.X_train.shape[1]
    output_dim = bundle.n_classes or 1
    if name in ("mlp", "dnn"):
        return build_mlp(
            input_dim, output_dim,
            hidden_units=params.get("hidden_units", 128),
            depth=params.get("depth", 2 if name == "dnn" else 1),
            dropout=params.get("dropout", 0.0),
            task=task,
        )
    if name == "lstm":
        return build_lstm(
            input_dim, output_dim,
            hidden_units=params.get("hidden_units", 64),
            num_layers=params.get("num_layers", 1),
            task=task,
        )
    if name == "rnn":
        return build_rnn(
            input_dim, output_dim,
            hidden_units=params.get("hidden_units", 64),
            num_layers=params.get("num_layers", 1),
            task=task,
        )
    if name == "gru":
        return build_gru(
            input_dim, output_dim,
            hidden_units=params.get("hidden_units", 64),
            num_layers=params.get("num_layers", 1),
            task=task,
        )
    if name == "attention_lstm":
        return build_attention_lstm(
            input_dim, output_dim,
            hidden_units=params.get("hidden_units", 64),
            dropout=params.get("dropout", 0.0),
            task=task,
        )
    if name == "transformer":
        return build_transformer(
            input_dim, output_dim,
            d_model=params.get("d_model", 32),
            nhead=params.get("nhead", 4),
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.1),
            task=task,
        )
    raise ValueError(f"알 수 없는 torch 모델: '{name}'")
