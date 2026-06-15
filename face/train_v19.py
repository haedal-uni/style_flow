import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFile, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 하이퍼파라미터
# =========================

SEED               = 42
IMG_SIZE           = 384        # EfficientNetV2-S native
BATCH_SIZE         = 8          
ACCUM_STEPS        = 4          # BATCH_SIZE × ACCUM_STEPS = 32
NUM_EPOCHS         = 250

PHASE1_EPOCHS      = 10         # backbone 동결, head만 학습
EARLY_STOP_PATIENCE = 25        # Phase 2 EarlyStopping
SWA_EPOCHS         = 20         # Phase 3: SWA 수집 기간
CKPT_INTERVAL      = 10

MIXUP_ALPHA        = 0.4
CUTMIX_ALPHA       = 1.0

PHASE1_LR          = 1e-3
PHASE2_BACKBONE_LR = 2e-5
PHASE2_HEAD_LR     = 2e-4
SWA_LR             = 5e-6       # Phase 3 고정 학습률

# v1.8 per-class F1 기반 클래스 가중치 (어려운 클래스 상향)
# Heart=0.8779, Oblong=0.9015, Oval=0.8082, Round=0.8633, Square=0.9177
_raw_w = [1 / 0.8779, 1 / 0.9015, 1 / 0.8082, 1 / 0.8633, 1 / 0.9177]
_mean_w = sum(_raw_w) / len(_raw_w)
CLASS_WEIGHTS = [w / _mean_w for w in _raw_w]   # 정규화 → 평균=1

TRAIN_PATH  = Path("../dataset/training_set")
TEST_PATH   = Path("../dataset/testing_set")
DATASET_DIR = Path("../dataset")
BROKEN_DIR  = Path("../broken_images")

CKPT_DIR    = Path(".")
CKPT_BEST   = CKPT_DIR / "best_model.pth"
CKPT_SWA    = CKPT_DIR / "swa_model.pth"
CKPT_LATEST = CKPT_DIR / "checkpoint_latest.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# Seed
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


set_seed(SEED)
print(f"On device : {DEVICE}")
print(f"Input size: {IMG_SIZE}×{IMG_SIZE}")
print(f"Eff. batch: {BATCH_SIZE} × {ACCUM_STEPS} = {BATCH_SIZE * ACCUM_STEPS}")


# =========================
# 경로 검증
# =========================

def validate_path(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} 경로 없음: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} 폴더 아님: {path}")


validate_path(TRAIN_PATH, "TRAIN_PATH")
validate_path(TEST_PATH,  "TEST_PATH")


# =========================
# 깨진 이미지 이동
# =========================

def scan_and_move_broken_images(dataset_dir: Path, broken_dir: Path):
    broken_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    broken = []

    for p in dataset_dir.rglob("*"):
        if p.suffix.lower() not in exts:
            continue
        try:
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:
                img.convert("RGB")
        except Exception as e:
            print(f"[깨진 이미지] {p} | {e}")
            broken.append(p)

    for p in broken:
        dst = broken_dir / p.relative_to(dataset_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))

    print(f"깨진 이미지 {len(broken)}개 이동 완료")


scan_and_move_broken_images(DATASET_DIR, BROKEN_DIR)


# =========================
# 이미지 로더
# =========================

def safe_pil_loader(path: str):
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()
            return img.convert("RGB")
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as e:
        raise RuntimeError(f"이미지 로드 실패: {path}") from e


# =========================
# Transforms
# =========================

weights       = EfficientNet_V2_S_Weights.DEFAULT
imagenet_mean = weights.transforms().mean
imagenet_std  = weights.transforms().std

_normalize = T.Normalize(mean=imagenet_mean, std=imagenet_std)

train_transforms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    # 다양한 촬영 각도 대응 — 얼굴형은 perspective 변화에 민감
    T.RandomPerspective(distortion_scale=0.2, p=0.3),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.ToTensor(),
    _normalize,
    T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

# 평가용 기본 transform (TTA base)
def make_eval_transform(resize_factor: float = 1.1, hflip: bool = False) -> T.Compose:
    transforms = [
        T.Resize(int(IMG_SIZE * resize_factor)),
        T.CenterCrop(IMG_SIZE),
    ]
    if hflip:
        transforms.append(T.RandomHorizontalFlip(p=1.0))
    transforms += [T.ToTensor(), _normalize]
    return T.Compose(transforms)

# 5-view TTA transform 리스트
TTA_TRANSFORMS = [
    make_eval_transform(1.10, False),   # 기본
    make_eval_transform(1.10, True),    # 수평 반전
    make_eval_transform(1.15, False),   # 약간 더 넓게
    make_eval_transform(1.15, True),    # 넓게 + 반전
    make_eval_transform(1.05, False),   # 가장 타이트
]

# 기본 평가 transform
base_eval_transform = TTA_TRANSFORMS[0]


# =========================
# Dataset / DataLoader
# =========================

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_PATH), transform=train_transforms, loader=safe_pil_loader,
)
test_dataset = datasets.ImageFolder(
    root=str(TEST_PATH), transform=base_eval_transform, loader=safe_pil_loader,
)

num_classes = len(train_dataset.classes)
if num_classes < 2:
    raise ValueError(f"클래스 수 부족: {num_classes}")

NUM_WORKERS = max(0, (os.cpu_count() or 1) - 1)
PIN_MEMORY  = DEVICE == "cuda"

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
)

print(f"Classes : {train_dataset.classes}")
print(f"Train   : {len(train_dataset)}")
print(f"Test    : {len(test_dataset)}")


# =========================
# 모델
# =========================

model = efficientnet_v2_s(weights=weights)

in_features = model.classifier[1].in_features   # 1280
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, num_classes),
)
model = model.to(DEVICE)


# =========================
# Phase 1: backbone 동결
# =========================

def freeze_backbone(m: nn.Module):
    for name, param in m.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(m: nn.Module):
    for param in m.parameters():
        param.requires_grad = True


freeze_backbone(model)

phase1_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=PHASE1_LR, weight_decay=0.01,
)

# 클래스 가중치 적용 손실
cw_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=0.05)

USE_AMP = DEVICE == "cuda"
scaler  = torch.amp.GradScaler("cuda") if USE_AMP else None


# =========================
# CutMix / Mixup
# =========================

def mixup_data(x, y, alpha: float = 0.4):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha: float = 1.0):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)

    H, W = x.size(2), x.size(3)
    cut_r = np.sqrt(1 - lam)
    cut_h, cut_w = int(H * cut_r), int(W * cut_r)

    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)

    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)   # 실제 비율로 재계산

    return mixed, y, y[idx], lam


def aug_criterion(pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =========================
# Train / Validate
# =========================

def train_one_epoch(model, loader, optimizer, scaler, device,
                    use_aug: bool, accum_steps: int):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        y_a = y_b = labels
        lam = 1.0

        if use_aug:
            if random.random() < 0.5:
                inputs, y_a, y_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
            else:
                inputs, y_a, y_b, lam = cutmix_data(inputs, labels, CUTMIX_ALPHA)

        ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()
        with ctx:
            outputs = model(inputs)
            loss = aug_criterion(outputs, y_a, y_b, lam) / accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_steps

    return running_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    losses, all_preds, all_labels = [], [], []

    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()
            with ctx:
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            losses.append(loss.item())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return sum(losses) / len(losses), accuracy_score(all_labels, all_preds)


def evaluate_with_tta(model, test_path: Path, device, tta_transforms):
    """5-view TTA로 최종 정확도 평가."""
    model.eval()
    all_probs  = None
    all_labels = None

    for transform in tta_transforms:
        ds = datasets.ImageFolder(
            root=str(test_path), transform=transform, loader=safe_pil_loader,
        )
        dl = DataLoader(
            ds, batch_size=BATCH_SIZE * 2, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )

        probs_list, labels_list = [], []
        with torch.inference_mode():
            for inputs, labels in dl:
                inputs = inputs.to(device, non_blocking=True)
                ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()
                with ctx:
                    logits = model(inputs)
                probs_list.append(F.softmax(logits, dim=1).cpu())
                labels_list.append(labels)

        batch_probs = torch.cat(probs_list)
        if all_probs is None:
            all_probs  = batch_probs
            all_labels = torch.cat(labels_list).numpy()
        else:
            all_probs += batch_probs

    all_probs /= len(tta_transforms)
    all_preds = all_probs.argmax(dim=1).numpy()
    return accuracy_score(all_labels, all_preds), all_preds, all_labels


# =========================
# Phase 2 옵티마이저 / 스케줄러
# =========================

def build_phase2_optimizer_scheduler(model):
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params     = [p for n, p in model.named_parameters() if "classifier" in n]

    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": PHASE2_BACKBONE_LR},
        {"params": head_params,     "lr": PHASE2_HEAD_LR},
    ], weight_decay=0.01)

    # T_0=25: 25 epoch 주기로 재시작, T_mult=2: 이후 주기 2배
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=25, T_mult=2, eta_min=1e-7,
    )
    return opt, sch


# =========================
# 체크포인트
# =========================

def _safe_ckpt(path: Path) -> bool:
    try:
        path.resolve().relative_to(CKPT_DIR.resolve())
        return True
    except ValueError:
        return False


def save_checkpoint(path: Path, epoch: int, phase: int, model, optimizer,
                    scheduler, scaler, best_acc: float, early_ctr: int,
                    train_losses: list, val_losses: list, accuracies: list):
    if not _safe_ckpt(path):
        raise ValueError(f"안전하지 않은 경로: {path}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "phase": phase,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_acc": best_acc, "early_ctr": early_ctr,
        "train_losses": train_losses, "val_losses": val_losses,
        "accuracies": accuracies,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device):
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return (
        int(ckpt["epoch"]) + 1,
        int(ckpt.get("phase", 1)),
        float(ckpt.get("best_acc", 0.0)),
        int(ckpt.get("early_ctr", 0)),
        ckpt.get("train_losses", []),
        ckpt.get("val_losses", []),
        ckpt.get("accuracies", []),
    )


# =========================
# 학습 준비
# =========================

train_loss_hist, val_loss_hist, acc_hist = [], [], []
best_acc       = 0.0
early_ctr      = 0
start_epoch    = 0
current_phase  = 1
swa_epochs_done = 0

optimizer = phase1_optimizer
scheduler = None

# 최신 체크포인트 이어 학습
ckpt_res = load_checkpoint(CKPT_LATEST, model, optimizer, scheduler, scaler, DEVICE)
if ckpt_res is None:
    print("[체크포인트] 없음 → epoch 1부터 새로 학습")
else:
    start_epoch, current_phase, best_acc, early_ctr, \
        train_loss_hist, val_loss_hist, acc_hist = ckpt_res
    if current_phase >= 2:
        unfreeze_all(model)
        optimizer, scheduler = build_phase2_optimizer_scheduler(model)
        ckpt_res2 = load_checkpoint(CKPT_LATEST, model, optimizer, scheduler, scaler, DEVICE)
        if ckpt_res2:
            start_epoch, current_phase, best_acc, early_ctr, \
                train_loss_hist, val_loss_hist, acc_hist = ckpt_res2
    print(f"[체크포인트] 재개 — epoch {start_epoch + 1}, phase {current_phase}")

# SWA 모델 (Phase 3에서 사용)
swa_model = None


# =========================
# 학습 루프
# =========================

for epoch in range(start_epoch, NUM_EPOCHS):

    # ── Phase 1 → 2 전환 ──
    if current_phase == 1 and epoch >= PHASE1_EPOCHS:
        print(f"\n[Phase 전환] epoch {epoch + 1}: backbone unfreeze → Phase 2 시작")
        unfreeze_all(model)
        optimizer, scheduler = build_phase2_optimizer_scheduler(model)
        current_phase = 2
        early_ctr = 0

    # ── Phase 2 EarlyStopping → Phase 3(SWA) 전환 ──
    if current_phase == 2 and early_ctr >= EARLY_STOP_PATIENCE:
        print(f"\n[Phase 전환] epoch {epoch + 1}: EarlyStopping → Phase 3 SWA 시작")
        current_phase   = 3
        swa_epochs_done = 0
        early_ctr       = 0
        # SWA 모델 초기화
        swa_model = AveragedModel(model)
        # SWA 전용 스케줄러 (고정 낮은 LR)
        swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=5)

    # ── Phase 3 종료 ──
    if current_phase == 3 and swa_epochs_done >= SWA_EPOCHS:
        print(f"\nPhase 3 완료: {SWA_EPOCHS} epoch SWA 수집 완료")
        break

    # ── 학습 ──
    use_aug   = (current_phase >= 2)
    accum     = ACCUM_STEPS if current_phase >= 2 else 1

    avg_train_loss = train_one_epoch(
        model, train_loader, optimizer, scaler, DEVICE, use_aug, accum,
    )

    if current_phase == 3:
        # SWA 가중치 수집
        swa_model.update_parameters(model)
        swa_scheduler.step()
        swa_epochs_done += 1

    avg_val_loss, accuracy = validate(model, test_loader, DEVICE)

    if scheduler is not None and current_phase == 2:
        scheduler.step(epoch - PHASE1_EPOCHS)

    train_loss_hist.append(avg_train_loss)
    val_loss_hist.append(avg_val_loss)
    acc_hist.append(accuracy)

    phase_label = f"P{current_phase}"
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} [{phase_label}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )

    # ── Best 모델 저장 (Phase 1/2 only) ──
    if current_phase <= 2:
        if accuracy > best_acc:
            best_acc  = accuracy
            early_ctr = 0
            save_checkpoint(
                CKPT_BEST, epoch, current_phase, model, optimizer, scheduler,
                scaler, best_acc, early_ctr, train_loss_hist, val_loss_hist, acc_hist,
            )
            print(f"  -> [Best] Accuracy {best_acc:.4f} — {CKPT_BEST} 저장")
        else:
            early_ctr += 1
            suffix = f" (SWA 전환 대기)" if early_ctr >= EARLY_STOP_PATIENCE else ""
            print(f"  -> EarlyStopping count: {early_ctr}/{EARLY_STOP_PATIENCE}{suffix}")
    else:
        print(f"  -> [SWA] {swa_epochs_done}/{SWA_EPOCHS} epoch 수집")

    # 최신 체크포인트 저장
    save_checkpoint(
        CKPT_LATEST, epoch, current_phase, model, optimizer, scheduler,
        scaler, best_acc, early_ctr, train_loss_hist, val_loss_hist, acc_hist,
    )

    if (epoch + 1) % CKPT_INTERVAL == 0:
        periodic = CKPT_DIR / f"checkpoint_epoch_{epoch + 1:04d}.pth"
        save_checkpoint(
            periodic, epoch, current_phase, model, optimizer, scheduler,
            scaler, best_acc, early_ctr, train_loss_hist, val_loss_hist, acc_hist,
        )
        print(f"  -> [주기] {periodic} 저장")


# =========================
# SWA BN 업데이트 & SWA 모델 저장
# =========================

if swa_model is not None:
    print("\n[SWA] BatchNorm 통계 업데이트 중...")
    update_bn(train_loader, swa_model, device=DEVICE)
    torch.save({"model_state_dict": swa_model.module.state_dict(),
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx},
               CKPT_SWA)
    print(f"[SWA] {CKPT_SWA} 저장 완료")


# =========================
# 최종 평가 (Best 모델 + TTA)
# =========================

print("\n" + "=" * 60)
print("최종 평가")
print("=" * 60)

# ── Best 모델 단독 평가 ──
if CKPT_BEST.exists():
    print("\n[1] Best 모델 단독 평가 (TTA 없음)")
    ckpt = torch.load(CKPT_BEST, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, acc_best = validate(model, test_loader, DEVICE)
    print(f"  Accuracy: {acc_best:.4f}")

# ── Best 모델 + TTA ──
print("\n[2] Best 모델 + TTA (5-view 앙상블)")
acc_tta, preds_tta, labels_tta = evaluate_with_tta(model, TEST_PATH, DEVICE, TTA_TRANSFORMS)
print(f"  TTA Accuracy: {acc_tta:.4f}")
print("\n  Classification Report:")
print(classification_report(
    labels_tta, preds_tta,
    target_names=train_dataset.classes, digits=4,
))

# ── SWA 모델 + TTA ──
if swa_model is not None:
    print("\n[3] SWA 모델 + TTA (5-view 앙상블)")
    swa_inner = swa_model.module
    acc_swa, preds_swa, labels_swa = evaluate_with_tta(swa_inner, TEST_PATH, DEVICE, TTA_TRANSFORMS)
    print(f"  SWA TTA Accuracy: {acc_swa:.4f}")
    print("\n  Classification Report (SWA):")
    print(classification_report(
        labels_swa, preds_swa,
        target_names=train_dataset.classes, digits=4,
    ))

print(f"\n학습 완료 — Best Val Accuracy (단독): {best_acc:.4f}")


# =========================
# 학습 곡선 저장
# =========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

p2_start = PHASE1_EPOCHS 
ax1.plot(train_loss_hist, label="Train Loss", alpha=0.8)
ax1.plot(val_loss_hist,   label="Val Loss",   alpha=0.8)
ax1.axvline(x=p2_start - 1, color="gray",   ls="--", label="Phase 2 시작")
ax1.set_title("Loss"), ax1.legend()

ax2.plot(acc_hist, label="Accuracy", color="green", alpha=0.8)
ax2.axvline(x=p2_start - 1, color="gray", ls="--", label="Phase 2 시작")
ax2.axhline(y=0.90, color="red",  ls=":", alpha=0.5, label="90% 목표")
ax2.set_title("Validation Accuracy"), ax2.legend()

fig.tight_layout()
curve_path = CKPT_DIR / "training_curve.png"
fig.savefig(curve_path, dpi=120)
print(f"학습 곡선 저장: {curve_path}")
