import argparse
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.data import (  # noqa: E402
    DEFAULT_NORMALIZE_MEAN,
    DEFAULT_NORMALIZE_STD,
    create_dataloaders,
)
from models.metrics_utils import (  # noqa: E402
    compute_confusion_matrix,
    unnormalize_images,
    wilson_ci,
)
from models.plots import plot_confusion, plot_loss_acc, plot_top_errors  # noqa: E402


DEFAULT_TRAIN_CSV = PROJECT_ROOT / "dataset" / "emnist-letters-train.csv"
DEFAULT_VAL_CSV = PROJECT_ROOT / "models" / "emnist-letters-test.csv"
DEFAULT_CHECKPOINTS = PROJECT_ROOT / "checkpoints"
DEFAULT_LOGS = PROJECT_ROOT / "logs"
DEFAULT_REPORTS = PROJECT_ROOT / "reports"


def set_seed(seed: int, cuda: bool) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_environment() -> Dict[str, str]:
    env_info = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "not available",
    }
    if torch.cuda.is_available():
        env_info["cuda_device"] = torch.cuda.get_device_name(0)
    else:
        env_info["cuda_device"] = "cpu-only"
    print("Torch verzija:", env_info["torch_version"])
    print("CUDA verzija:", env_info["cuda_version"])
    print("Uređaj:", env_info["cuda_device"])
    return env_info


class CNN(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    grad_clip: float,
    use_amp: bool,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    scheduler_step_per_batch: bool,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        if scaler and use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler and scheduler_step_per_batch:
            scheduler.step()

        batch_size_actual = labels.size(0)
        total_loss += loss.item() * batch_size_actual
        total_correct += int(torch.sum(torch.argmax(outputs, dim=1) == labels))
        total_samples += batch_size_actual

    avg_loss = total_loss / total_samples if total_samples else 0.0
    avg_acc = total_correct / total_samples if total_samples else 0.0
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    return_details: bool = False,
) -> Tuple[float, float, int, int, Optional[torch.Tensor], Optional[List[int]], Optional[List[int]]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    images_collected: List[torch.Tensor] = []
    preds_collected: List[int] = []
    labels_collected: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += int(torch.sum(preds == labels))
            total_samples += labels.size(0)

            if return_details:
                images_collected.append(images.cpu())
                preds_collected.extend(preds.cpu().tolist())
                labels_collected.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_samples if total_samples else 0.0
    avg_acc = total_correct / total_samples if total_samples else 0.0
    if return_details:
        stacked_images = torch.cat(images_collected) if images_collected else torch.empty(0)
    else:
        stacked_images = None
    return (
        avg_loss,
        avg_acc,
        total_correct,
        total_samples,
        stacked_images,
        labels_collected if return_details else None,
        preds_collected if return_details else None,
    )


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    steps_per_epoch: int,
    base_lr: float,
    max_lr: Optional[float],
    pct_start: float,
) -> Tuple[Optional[optim.lr_scheduler._LRScheduler], bool]:
    if scheduler_type == "onecycle":
        cycle_max_lr = max_lr if max_lr is not None else base_lr * 3.0
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cycle_max_lr,
            epochs=epochs,
            steps_per_epoch=max(1, steps_per_epoch),
            pct_start=pct_start,
            anneal_strategy="cos",
        )
        return scheduler, True
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )
        return scheduler, False
    return None, False


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    best: float = float("inf")
    patience_counter: int = 0

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.patience


def train(args: argparse.Namespace) -> None:
    use_cuda = torch.cuda.is_available() and not args.force_cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    set_seed(args.seed, cuda=use_cuda)
    env_info = log_environment()

    train_loader, train_clean_loader, val_loader = create_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mean=args.normalize_mean,
        std=args.normalize_std,
        random_erasing_p=args.random_erasing_p,
        disable_augmentations=args.disable_augmentations,
        device=device,
    )

    model = CNN(dropout=args.dropout).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = use_cuda and args.use_amp
    scaler = torch.amp.GradScaler(enabled=use_amp)

    scheduler, scheduler_step_per_batch = create_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        base_lr=args.lr,
        max_lr=args.max_lr,
        pct_start=args.onecycle_pct_start,
    )

    early_stopper = EarlyStopping(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta) if args.early_stop_patience > 0 else None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"emnist_cnn_{timestamp}"
    checkpoints_dir = DEFAULT_CHECKPOINTS
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_checkpoints_dir = checkpoints_dir / run_id
    run_checkpoints_dir.mkdir(exist_ok=True)
    logs_dir = DEFAULT_LOGS
    logs_dir.mkdir(exist_ok=True)
    reports_dir = DEFAULT_REPORTS / f"run_{timestamp}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    epoch_records: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_val_acc = -float("inf")
    best_checkpoint_path = checkpoints_dir / "best.pt"
    best_acc_checkpoint_path = run_checkpoints_dir / "best_val_acc.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss_aug, train_acc_aug = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            scheduler=scheduler,
            scheduler_step_per_batch=scheduler_step_per_batch,
        )

        train_clean_loss, train_clean_acc, _, _, _, _, _ = evaluate(
            model=model,
            loader=train_clean_loader,
            loss_fn=loss_fn,
            device=device,
            return_details=False,
        )

        val_loss, val_acc, val_correct, val_total, val_images, val_labels, val_preds = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            return_details=True,
        )

        if scheduler and not scheduler_step_per_batch:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        val_ci_lo, val_ci_hi = wilson_ci(val_correct, val_total)

        print(
            (
                f"[Epoch {epoch:03d}] train_loss_aug={train_loss_aug:.4f} "
                f"train_clean_loss={train_clean_loss:.4f} train_clean_acc={train_clean_acc * 100:.2f}% "
                f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}% "
                f"val_ci=[{val_ci_lo:.2f}, {val_ci_hi:.2f}] "
                f"time={epoch_time:.1f}s"
            )
        )

        record = {
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": train_clean_loss,
            "train_loss_aug": train_loss_aug,
            "train_acc_clean": train_clean_acc * 100.0,
            "val_loss": val_loss,
            "val_acc": val_acc * 100.0,
            "val_ci_lo": val_ci_lo,
            "val_ci_hi": val_ci_hi,
            "n_val": val_total,
            "epoch_time": epoch_time,
        }
        epoch_records.append(record)

        current_checkpoint = run_checkpoints_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if use_amp else None,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_clean_loss,
            },
            current_checkpoint,
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            torch.save(model.state_dict(), run_checkpoints_dir / "best_val_loss.pt")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_acc_checkpoint_path)

        if early_stopper and early_stopper.step(val_loss):
            print(f"Rano zaustavljanje: val loss nije poboljšan za najmanje {args.early_stop_min_delta:.6f} tokom {args.early_stop_patience} epoha.")
            break

        if args.max_epochs and epoch >= args.max_epochs:
            break

    metrics_csv_path = reports_dir / "metrics.csv"
    _write_metrics_csv(epoch_records, metrics_csv_path)
    _append_global_log(epoch_records, logs_dir / "metrics.csv")

    class_names = [chr(ord("A") + i) for i in range(26)]

    if val_labels is not None and val_preds is not None:
        cm = compute_confusion_matrix(val_labels, val_preds, num_classes=26, normalize=True)
        plot_confusion(cm, class_names, reports_dir / "confusion_matrix.png")

        if val_images is not None and val_images.numel() > 0:
            val_images_display = unnormalize_images(val_images, args.normalize_mean, args.normalize_std).clamp(0, 1)
            plot_top_errors(
                images=val_images_display,
                y_true=val_labels,
                y_pred=val_preds,
                out_png=reports_dir / "top_errors.png",
                max_items=20,
                class_names=class_names,
            )

    plot_loss_acc(metrics_csv_path, reports_dir / "loss_acc.png", ema_alpha=args.ema_alpha)

    args_json_path = reports_dir / "args.json"
    args_serializable = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(args_json_path, "w", encoding="utf-8") as f:
        json.dump({**args_serializable, **env_info, "run_id": run_id}, f, indent=2)

    print(f"Sačuvan best checkpoint (val loss) u: {best_checkpoint_path}")
    print(f"Izveštaji: {reports_dir}")


def _write_metrics_csv(records: List[Dict[str, float]], out_path: Path) -> None:
    import csv

    if not records:
        return
    fieldnames = [
        "run_id",
        "epoch",
        "train_loss",
        "train_loss_aug",
        "train_acc_clean",
        "val_loss",
        "val_acc",
        "val_ci_lo",
        "val_ci_hi",
        "n_val",
        "epoch_time",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _append_global_log(records: List[Dict[str, float]], global_log_path: Path) -> None:
    import csv
    import os

    if not records:
        return
    fieldnames = [
        "run_id",
        "epoch",
        "train_loss",
        "train_loss_aug",
        "train_acc_clean",
        "val_loss",
        "val_acc",
        "val_ci_lo",
        "val_ci_hi",
        "n_val",
        "epoch_time",
    ]
    write_header = not global_log_path.exists()
    os.makedirs(global_log_path.parent, exist_ok=True)
    with open(global_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stabilizovan trening CNN modela nad EMNIST Letters datasetom.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV, help="Putanja do train CSV datoteke.")
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV, help="Putanja do val CSV datoteke.")
    parser.add_argument("--batch-size", type=int, default=128, help="Veličina batch-a.")
    parser.add_argument("--epochs", type=int, default=30, help="Broj epoha treniranja.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Osnovni learning rate.")
    parser.add_argument("--max-lr", type=float, default=None, help="Maksimalni LR za OneCycle scheduler (ako nije naveden koristi 3x lr).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay za optimizer.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout posle FC sloja.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing koeficijent za CrossEntropyLoss.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max norm za clip_grad_norm_ (<=0 onemogućava).")
    parser.add_argument("--scheduler", choices=["none", "onecycle", "cosine"], default="onecycle", help="Scheduler politika.")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.1, help="pct_start parametar za OneCycleLR.")
    parser.add_argument("--num-workers", type=int, default=4, help="Broj worker-a za DataLoader.")
    parser.add_argument("--seed", type=int, default=1337, help="Globalni random seed.")
    parser.add_argument("--use-amp", action="store_true", help="Koristi mixed precision (samo kada je CUDA dostupna).")
    parser.add_argument("--force-cpu", action="store_true", help="Forsiraj korišćenje CPU-a čak i ako je CUDA dostupna.")
    parser.add_argument("--early-stop-patience", type=int, default=6, help="Patience za rano zaustavljanje po val loss-u (0 onemogućava).")
    parser.add_argument("--early-stop-min-delta", type=float, default=5e-4, help="Minimalno poboljšanje val loss-a za resetovanje patience-a.")
    parser.add_argument("--disable-augmentations", action="store_true", help="Isključi augmentacije u treningu.")
    parser.add_argument("--random-erasing-p", type=float, default=0.1, help="Verovatnoća RandomErasing augmentacije.")
    parser.add_argument("--normalize-mean", type=float, default=DEFAULT_NORMALIZE_MEAN, help="Normalizacija mean vrednost.")
    parser.add_argument("--normalize-std", type=float, default=DEFAULT_NORMALIZE_STD, help="Normalizacija std vrednost.")
    parser.add_argument("--ema-alpha", type=float, default=0.3, help="EMA faktor za val tačnost na grafiku.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Pomoćni limit epoha (za debug).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
