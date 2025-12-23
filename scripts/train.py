from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import mlflow
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from p2igan_bench.data.dataloader import P2IDataModule
from p2igan_bench.models import build_discriminator, build_generator
from p2igan_bench.modules import ReconstructionLoss, gan_loss
from p2igan_bench.metrics import MetricConfig, RainfallMetricSuite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P2I-GAN benchmark model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("p2igan_bench/config/p2igan_baseline.json"),
        help="Path to JSON/YAML config file.",
    )
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--tracking-uri", type=str, default=None, help="Optional MLflow tracking URI")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(f)
        return json.load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_dict(data: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key))
        elif isinstance(value, (list, tuple)):
            items[new_key] = json.dumps(value)
        elif value is not None:
            items[new_key] = value
    return items


class Trainer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        seed_everything(cfg.get("seed", 42))
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        self.data_module = P2IDataModule(cfg)
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        self.test_loader = self.data_module.test_dataloader()

        self.generator = build_generator(cfg).to(self.device)
        self.discriminator = build_discriminator(cfg).to(self.device) if cfg["loss"].get("use_gan", 0) else None

        opt_cfg = cfg["train"]["optimizer"]
        self.opt_g = Adam(
            self.generator.parameters(),
            lr=opt_cfg["lr"],
            betas=(opt_cfg.get("beta1", 0.0), opt_cfg.get("beta2", 0.99)),
        )
        self.opt_d = None
        if self.discriminator is not None:
            self.opt_d = Adam(
                self.discriminator.parameters(),
                lr=opt_cfg["lr"],
                betas=(opt_cfg.get("beta1", 0.0), opt_cfg.get("beta2", 0.99)),
            )

        self.rec_loss = ReconstructionLoss(
            hole_weight=cfg["loss"].get("hole_weight", 1.0),
            valid_weight=cfg["loss"].get("valid_weight", 1.0),
        )
        self.use_gan = bool(cfg["loss"].get("use_gan", 0))
        self.gan_loss_type = cfg["loss"].get("gan_loss", "nsgan")
        self.gan_real_label = cfg["loss"].get("target_real_label", 1.0)
        self.gan_fake_label = cfg["loss"].get("target_fake_label", 0.0)

        self.save_dir = Path(cfg.get("save_dir", "weights"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = cfg["train"]
        self.log_every = int(train_cfg.get("log_step", 100))
        self.global_step = 0
        self.max_steps = train_cfg.get("iterations")
        self.max_epochs = train_cfg.get("max_epochs")
        if self.max_epochs is None:
            loader_len = max(1, len(self.train_loader))
            if self.max_steps:
                self.max_epochs = math.ceil(self.max_steps / loader_len)
            else:
                self.max_epochs = train_cfg.get("niter", 1)
        if self.max_steps is None:
            self.max_steps = self.max_epochs * max(1, len(self.train_loader))

        self.best_val = float("inf")
        metric_cfg = MetricConfig()
        self.val_metrics = RainfallMetricSuite(metric_cfg)
        self.test_metrics = RainfallMetricSuite(metric_cfg)

    def train(self) -> None:
        experiment_name = self.cfg.get("experiment_name")
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        run_name = self.cfg.get("run_name")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(flatten_dict(self.cfg))
            for epoch in range(1, self.max_epochs + 1):
                train_loss = self._train_one_epoch(epoch)
                mlflow.log_metric("train/loss", train_loss, step=self.global_step)

                if self.val_loader is not None:
                    val_results = self.evaluate(self.val_loader, self.val_metrics)
                    for key, value in val_results.items():
                        mlflow.log_metric(f"val/{key}", value, step=self.global_step)
                    if val_results.get("loss", float("inf")) < self.best_val:
                        self.best_val = val_results["loss"]
                        checkpoint = self.save_dir / "best.pt"
                        self._save_checkpoint(checkpoint, epoch)
                        mlflow.log_artifact(str(checkpoint))

                if self.global_step >= self.max_steps:
                    break

            if self.test_loader is not None:
                test_results = self.evaluate(self.test_loader, self.test_metrics)
                for key, value in test_results.items():
                    mlflow.log_metric(f"test/{key}", value, step=self.global_step)

    def _train_one_epoch(self, epoch: int) -> float:
        self.generator.train()
        if self.discriminator is not None:
            self.discriminator.train()

        running_loss = 0.0
        steps = 0

        for batch in self.train_loader:
            frames, masked_frames, masks = self._prepare_batch(batch)

            preds = self.generator(masked_frames, masks)
            loss_g, loss_dict = self.rec_loss(preds, frames, masks)

            if self.use_gan and self.discriminator is not None:
                logits_fake = self.discriminator(preds.detach())
                logits_real = self.discriminator(frames)
                loss_d = gan_loss(
                    logits_real,
                    True,
                    loss_type=self.gan_loss_type,
                    is_disc=True,
                    target_real_label=self.gan_real_label,
                    target_fake_label=self.gan_fake_label,
                ) + gan_loss(
                    logits_fake,
                    False,
                    loss_type=self.gan_loss_type,
                    is_disc=True,
                    target_real_label=self.gan_real_label,
                    target_fake_label=self.gan_fake_label,
                )
                self.opt_d.zero_grad()
                loss_d.backward()
                self.opt_d.step()

                logits_fake_for_g = self.discriminator(preds)
                adv_loss = gan_loss(
                    logits_fake_for_g,
                    True,
                    loss_type=self.gan_loss_type,
                    is_disc=False,
                    target_real_label=self.gan_real_label,
                    target_fake_label=self.gan_fake_label,
                ) * self.cfg["loss"].get("adversarial_weight", 0.5)
                loss_g = loss_g + adv_loss
            else:
                loss_d = None

            self.opt_g.zero_grad()
            loss_g.backward()
            self.opt_g.step()

            running_loss += float(loss_g.detach())
            steps += 1
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                mlflow.log_metric("train/step_loss", float(loss_g.detach()), step=self.global_step)
                for key, value in loss_dict.items():
                    mlflow.log_metric(f"train/{key}", value, step=self.global_step)

            if self.global_step >= self.max_steps:
                break

        return running_loss / max(1, steps)

    def evaluate(self, loader: Optional[DataLoader], metrics: RainfallMetricSuite) -> Dict[str, float]:
        if loader is None:
            return {}

        self.generator.eval()
        metrics.reset()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for batch in loader:
                frames, masked_frames, masks = self._prepare_batch(batch)
                preds = self.generator(masked_frames, masks)
                loss, _ = self.rec_loss(preds, frames, masks)
                total_loss += float(loss.detach())
                batches += 1
                metrics.update(preds, frames)

        results = metrics.compute()
        results["loss"] = total_loss / max(1, batches)
        return results

    def _prepare_batch(self, batch: Iterable[torch.Tensor]):
        frames, masked_frames, masks = batch
        frames = frames.permute(0, 1, 4, 2, 3).to(self.device, non_blocking=True)
        masked_frames = masked_frames.permute(0, 1, 4, 2, 3).to(self.device, non_blocking=True)
        masks = masks.permute(0, 1, 4, 2, 3).to(self.device, non_blocking=True)
        return frames, masked_frames, masks

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "generator": self.generator.state_dict(),
            "optimizer_g": self.opt_g.state_dict(),
        }
        if self.discriminator is not None and self.opt_d is not None:
            state["discriminator"] = self.discriminator.state_dict()
            state["optimizer_d"] = self.opt_d.state_dict()
        torch.save(state, path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    if args.run_name:
        config["run_name"] = args.run_name
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    elif "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
