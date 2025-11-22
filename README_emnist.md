# EMNIST CNN Trening

Ovaj dokument opisuje podešavanje i pokretanje stabilizovanog treninga konvolucione mreže za EMNIST **Letters** dataset u ovom repozitorijumu.

## Dataset i transformacije

- Korišćena je EMNIST Letters varijanta (26 klasa, oznake 1–26). Labela se pomera u opseg `[0, 25]`.
- Svaka slika (`28×28`) se rotira za 90° u smeru suprotnom kazaljke na satu i zatim vertikalno flipuje, kako bi orijentacija odgovarala segmentisanim slovima iz hand-tracking pipeline-a.
- Normalizacija: `(mean=0.1736, std=0.3249)`. Vrednosti su empirijski izračunate nad transformisanim CSV datasetom.
- Trening augmentacije (može se isključiti argumentom `--disable-augmentations`):
  - `RandomAffine(degrees=8, translate=(0.05, 0.05), shear=5)`
  - `RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3))`
- Evaluacioni loaderi (train-clean i validation) koriste samo `ToTensor()` + `Normalize`.

## Reproduktivnost

- Globalni seed: `--seed` (podrazumevano `1337`) postavlja `random`, `numpy`, `torch` i `torch.cuda` generator.
- `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`.
- Loguju se verzije PyTorch/CUDA i naziv CUDA uređaja (ili CPU) u `reports/run_<timestamp>/args.json`.

## Pokretanje

```bash
# CPU (test/dry run)
python models/train_emnist_cnn.py --force-cpu --epochs 5

# Tipičan GPU trening sa AMP-om
python models/train_emnist_cnn.py --use-amp --epochs 30 --batch-size 256

# Onemogućene augmentacije (npr. za ablation)
python models/train_emnist_cnn.py --disable-augmentations
```

Argumenti uključuju kontrolu nad schedulerom (`--scheduler {onecycle,cosine,none}`, `--max-lr`, `--onecycle-pct-start`), regularizacijom (`--dropout`, `--weight-decay`, `--label-smoothing`, `--grad-clip`), ranim zaustavljanjem (`--early-stop-patience`, `--early-stop-min-delta`) i normalizacijom (`--normalize-mean`, `--normalize-std`).

## Izlazni artefakti

Posle svake vožnje kreiraju se sledeći fajlovi i direktorijumi:

- `logs/metrics.csv` – kumulativni log svih epoha i run-ova (kolone: `run_id, epoch, train_loss, train_loss_aug, train_acc_clean, val_loss, val_acc, val_ci_lo, val_ci_hi, n_val, epoch_time`).
- `checkpoints/best.pt` – najbolji model po najnižem validation loss-u (prepisuje se svakim run-om). U `checkpoints/<run_id>/` se čuvaju svi epoch checkpoint-i, plus `best_val_loss.pt` i `best_val_acc.pt`.
- `reports/run_<timestamp>/` – materijali za analizu:
  - `metrics.csv` – metrika za konkretan run.
  - `loss_acc.png` – loss (train clean, val) i accuracy (train clean, val, EMA val) + 95% Wilson CI senka.
  - `confusion_matrix.png` – normirana po true klasi (redovima).
  - `top_errors.png` – mreža 5×4 najčešćih pogrešnih parova (prikaz slike + T/P oznaka).
  - `args.json` – svi argumenti + informacije o okruženju i run_id.

## Napomene

- Model: CNN sa tri konvoluciona bloka (32/64/128 filtera) i jednim FC slojem (128 neurona) + dropout (`--dropout`, podrazumevano 0.3).
- Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`), scheduler OneCycleLR (max_lr=3× osnovni LR) kao podrazumevani.
- Loss: `CrossEntropyLoss` sa `label_smoothing=0.05`.
- Gradient clipping: `clip_grad_norm_=1.0`.
- Early stopping: patience 6 epoha, min_delta 5e-4 na validation loss-u.

Svi grafovi i metrički logovi namenjeni su "akademskoj" prezentaciji – provera čistoće tren/val metrika, prikaz statističke pouzdanosti (Wilson CI), kao i vizuelna analiza grešaka preko confusion matrice i top-errors kolaža.
