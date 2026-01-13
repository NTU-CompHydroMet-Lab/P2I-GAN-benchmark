from copy import deepcopy
from pathlib import Path
import json

from torch.utils.data import DataLoader

from .sti_dataset import Dataset  # 舊 Dataset 實作


class P2IDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        data_cfg = cfg["data"]
        train_cfg = cfg["train"]
        self.num_workers = train_cfg.get("num_workers", 0)
        self.pin_memory = train_cfg.get("pin_memory", True)
        self.persistent_workers = train_cfg.get("persistent_workers", True)
        self.prefetch_factor = train_cfg.get("prefetch_factor", 2)

        train_cfg = data_cfg["train"]
        self.train_args = self._build_dataset_args(train_cfg)

        shared_params = self._extract_shared_params(self.train_args)

        test_cfg = data_cfg.get("test")
        self.test_args = None
        self.test_shuffle = False
        if test_cfg:
            self.test_args = self._build_dataset_args(test_cfg, defaults=shared_params)
            self.test_shuffle = bool(test_cfg.get("shuffle", False))

        valid_cfg = data_cfg.get("valid")
        self.valid_args = None
        self.valid_shuffle = False
        if valid_cfg:
            # validation 預設沿用訓練的尺寸/時間長度，避免與模型長度不一致
            self.valid_args = self._build_dataset_args(valid_cfg, defaults=shared_params)
            self.valid_shuffle = bool(valid_cfg.get("shuffle", False))

    def train_dataloader(self):
        train_bs = self.cfg["train"]["batch_size"]
        return self._create_loader(self.train_args, shuffle=True, batch_size=train_bs)

    def val_dataloader(self):
        if not self.valid_args:
            return None
        train_bs = self.cfg["train"]["batch_size"]
        return self._create_loader(self.valid_args, shuffle=self.valid_shuffle, batch_size=train_bs)

    def test_dataloader(self):
        if not self.test_args:
            return None
        test_bs = 1
        return self._create_loader(self.test_args, shuffle=self.test_shuffle, batch_size=test_bs)

    def _create_loader(self, dataset_args, shuffle, batch_size):
        dataset = Dataset(dataset_args)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def _build_dataset_args(self, split_cfg, defaults=None):
        defaults = defaults or {}
        args = {}

        for key in ("w", "h", "sample_length"):
            if key in defaults:
                args[key] = defaults[key]
            if key in split_cfg:
                value = split_cfg[key]
                if value is None and key in args:
                    args.pop(key)
                elif value is not None:
                    args[key] = value

        mask_cfg = deepcopy(defaults.get("mask", {}))
        if split_cfg.get("mask"):
            mask_cfg.update(split_cfg["mask"])
        if mask_cfg:
            args["mask"] = mask_cfg

        if "data_root" in split_cfg:
            args["data_root"] = split_cfg["data_root"]
        elif "data_root1" in split_cfg:
            args["data_root"] = split_cfg["data_root1"]
        else:
            raise KeyError("Dataset config requires 'data_root'.")

        return args

    def _extract_shared_params(self, dataset_args):
        shared = {}
        for key in ("w", "h", "sample_length"):
            if key in dataset_args:
                shared[key] = dataset_args[key]
        if "mask" in dataset_args:
            shared["mask"] = deepcopy(dataset_args["mask"])
        return shared


def _describe_tensor(name, tensor):
    tensor = tensor.detach().cpu()
    flat = tensor.reshape(-1)
    stats = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
    }
    sample_values = flat[:5].tolist()
    print(
        f"[{name}] shape={stats['shape']} dtype={stats['dtype']} "
        f"min={stats['min']:.4f} max={stats['max']:.4f} mean={stats['mean']:.4f}"
    )
    print(f"    sample={sample_values}")


def _inspect_loader(loader, label):
    if loader is None:
        print(f"[{label}] loader not configured")
        return
    try:
        batch = next(iter(loader))
    except StopIteration:
        print(f"[{label}] loader is empty")
        return

    video, masked, mask = batch
    print(f"[{label}] batch size={video.shape[0]}")
    _describe_tensor(f"{label}/video", video)
    _describe_tensor(f"{label}/masked", masked)
    _describe_tensor(f"{label}/mask", mask)


if __name__ == "__main__":
    CONFIG_PATH = Path(
        "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/p2igan_bench/config/p2igan_baseline.json"
    )
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_module = P2IDataModule(cfg)

    print("==== Inspecting train loader ====")
    _inspect_loader(data_module.train_dataloader(), "train")

    print("==== Inspecting validation loader ====")
    _inspect_loader(data_module.val_dataloader(), "valid")

    print("==== Inspecting test loader ====")
    _inspect_loader(data_module.test_dataloader(), "test")
