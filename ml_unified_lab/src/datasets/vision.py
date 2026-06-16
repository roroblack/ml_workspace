"""Vision 데이터셋 로더 (torchvision 필요).

torch/torchvision이 설치되어 있지 않으면 명확한 에러를 던진다.
"""
from __future__ import annotations

from ..config import PROJECT_ROOT
from .types import DatasetBundle

_VISION_SPECS = {
    "mnist": dict(cls="MNIST", channels=1, size=28, n_classes=10,
                  mean=(0.1307,), std=(0.3081,)),
    "fashion_mnist": dict(cls="FashionMNIST", channels=1, size=28, n_classes=10,
                          mean=(0.2860,), std=(0.3530,)),
    "cifar10": dict(cls="CIFAR10", channels=3, size=32, n_classes=10,
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
}


def load_vision(name: str, config: dict | None = None) -> DatasetBundle:
    try:
        import torchvision
        from torchvision import transforms
    except ImportError as e:  # pragma: no cover - 환경 의존
        raise ImportError(
            f"'{name}' 데이터셋은 torchvision이 필요합니다. "
            "`pip install torch torchvision` 후 다시 실행하세요."
        ) from e

    spec = _VISION_SPECS[name]
    data_root = str(PROJECT_ROOT / "runs" / "_data_cache")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(spec["mean"], spec["std"]),
    ])

    ds_cls = getattr(torchvision.datasets, spec["cls"])
    train_ds = ds_cls(root=data_root, train=True, download=True, transform=tf)
    test_ds = ds_cls(root=data_root, train=False, download=True, transform=tf)

    return DatasetBundle(
        name=name,
        task="image_classification",
        kind="vision",
        train_dataset=train_ds,
        test_dataset=test_ds,
        input_shape=(spec["channels"], spec["size"], spec["size"]),
        n_classes=spec["n_classes"],
        target_names=[str(i) for i in range(spec["n_classes"])],
    )
