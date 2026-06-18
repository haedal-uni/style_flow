"""
얼굴형 분류 v1.9 남성 전이학습 — EfficientNetV2-S · 여성 SWA 가중치 기반

전이학습 전략:
  1. 여성 데이터로 학습된 female/model/swa_model.pth를 초기 가중치로 사용
  2. EfficientNetV2-S features[0~4] 동결 (에지·질감·기본 구조 — 성별 공통)
     features[5~7] + classifier 언프리즈 → 남성 데이터로 파인튜닝
  3. 배경 제거: rembg AI 모델 (없으면 GrabCut 폴백)
  4. 클래스 매핑: rectangular → oblong, ovale → oval

남성 데이터셋 구조:
  dataset/men/
  ├── training_set/ { ovale/, rectangular/, round/, square/ }
  └── testing_set/  { ovale/, rectangular/, round/, square/ }

산출물:
  male/best_model.pth
  male/swa_model.pth
  male/training_curve.png

실행:
  python male/train.py
  python male/train.py --no-rembg        # 배경 제거 비활성화
  python male/train.py --epochs 50       # 파인튜닝 에폭 수 조정
"""

import os
import random
import argparse
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
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Dataset
import cv2
from sklearn.metrics import accuracy_score, classification_report

# rembg 선택적 임포트 — onnxruntime-gpu가 설치된 환경에서도 CPU로 고정
# (CUDA 초기화 실패 방지: rembg는 CPU로, 학습은 PyTorch CUDA로 분리)
try:
    from rembg import remove as _rembg_remove_raw, new_session as _rembg_new_session
    from PIL import Image as PILImage
    _REMBG_SESSION = _rembg_new_session(providers=["CPUExecutionProvider"])
    def rembg_remove(img):
        return _rembg_remove_raw(img, session=_REMBG_SESSION)
    _REMBG_AVAILABLE = True
except ImportError:
    _REMBG_AVAILABLE = False

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# 경로
# =========================

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))  # face/male/
_ROOT = _HERE.parent.parent                                # dataset 상위 디렉토리

FEMALE_SWA_MODEL = _HERE.parent / "female" / "model" / "swa_model.pth"

MALE_TRAIN_DIR = _ROOT / "dataset" / "men" / "training_set"
MALE_TEST_DIR  = _ROOT / "dataset" / "men" / "testing_set"

CKPT_DIR    = _HERE / "model"
CKPT_BEST   = CKPT_DIR / "best_model.pth"
CKPT_SWA    = CKPT_DIR / "swa_model.pth"
CKPT_LATEST = CKPT_DIR / "checkpoint_latest.pth"


# =========================
# 클래스 설정
# =========================

# 여성 모델과 동일한 5클래스 (heart는 남성 데이터 없음)
CLASSES    = ["Heart", "Oblong", "Oval", "Round", "Square"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# 남성 폴더명 → 정규 클래스명 매핑
MALE_CLASS_MAP = {
    "rectangular": "Oblong",
    "ovale":       "Oval",
    "round":       "Round",
    "square":      "Square",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# =========================
# 하이퍼파라미터
# =========================

SEED           = 42
IMG_SIZE       = 384
BATCH_SIZE     = 8
ACCUM_STEPS    = 2          # 파인튜닝: 실효 배치 16
FINETUNE_EPOCHS = 50
SWA_EPOCHS     = 10
CKPT_INTERVAL  = 10

FINETUNE_LR_HEAD     = 1e-4
FINETUNE_LR_BACKBONE = 1e-5
SWA_LR               = 1e-6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
print(f"Device: {DEVICE}")
print(f"rembg: {'사용 가능' if _REMBG_AVAILABLE else '없음 (GrabCut 폴백)'}")


# =========================
# 배경 제거
# =========================

def remove_background(img_bgr: np.ndarray, use_rembg: bool = True) -> np.ndarray:
    """얼굴+머리카락 보존, 배경만 흰색으로 교체."""
    if use_rembg and _REMBG_AVAILABLE:
        try:
            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img  = PILImage.fromarray(img_rgb)
            result   = rembg_remove(pil_img)
            result_np = np.array(result)
            alpha    = result_np[:, :, 3]
            fg_mask  = (alpha > 10).astype(np.uint8)
            white_bg = np.ones_like(img_bgr) * 255
            mask_3ch = np.stack([fg_mask] * 3, axis=-1)
            return np.where(
                mask_3ch,
                cv2.cvtColor(result_np[:, :, :3], cv2.COLOR_RGB2BGR),
                white_bg,
            ).astype(np.uint8)
        except Exception:
            pass

    # GrabCut 폴백
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.10), int(h * 0.05), int(w * 0.80), int(h * 0.90))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        white_bg = np.ones_like(img_bgr) * 255
        return np.where(np.stack([fg] * 3, axis=-1), img_bgr, white_bg).astype(np.uint8)
    except Exception:
        return img_bgr


# =========================
# 데이터셋 수집
# =========================

def collect_male_pairs(root: Path) -> list:
    """남성 폴더명을 정규 클래스명으로 매핑하여 (경로, 클래스명) 쌍 수집."""
    pairs = []
    for folder_name, cls_name in MALE_CLASS_MAP.items():
        for variant in [folder_name, folder_name.capitalize(), folder_name.upper()]:
            cls_dir = root / variant
            if cls_dir.is_dir():
                for p in sorted(cls_dir.iterdir()):
                    if p.suffix.lower() in IMG_EXTS:
                        pairs.append((p, cls_name))
                break
    return pairs


# =========================
# PyTorch Dataset
# =========================

weights       = EfficientNet_V2_S_Weights.DEFAULT
imagenet_mean = weights.transforms().mean
imagenet_std  = weights.transforms().std
_normalize    = T.Normalize(mean=imagenet_mean, std=imagenet_std)

train_transform = T.Compose([
    T.Resize(int(IMG_SIZE * 1.15)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
    T.ToTensor(),
    _normalize,
    T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

eval_transform = T.Compose([
    T.Resize(int(IMG_SIZE * 1.1)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    _normalize,
])

TTA_TRANSFORMS = [
    T.Compose([T.Resize(int(IMG_SIZE * 1.10)), T.CenterCrop(IMG_SIZE), T.ToTensor(), _normalize]),
    T.Compose([T.Resize(int(IMG_SIZE * 1.10)), T.CenterCrop(IMG_SIZE), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _normalize]),
    T.Compose([T.Resize(int(IMG_SIZE * 1.15)), T.CenterCrop(IMG_SIZE), T.ToTensor(), _normalize]),
    T.Compose([T.Resize(int(IMG_SIZE * 1.15)), T.CenterCrop(IMG_SIZE), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _normalize]),
    T.Compose([T.Resize(int(IMG_SIZE * 1.05)), T.CenterCrop(IMG_SIZE), T.ToTensor(), _normalize]),
]


class MaleFaceDataset(Dataset):
    def __init__(self, pairs: list, transform, use_rembg: bool = True):
        self.pairs      = pairs
        self.transform  = transform
        self.use_rembg  = use_rembg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, cls_name = self.pairs[idx]
        label = CLASS_TO_IDX[cls_name]

        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise RuntimeError("cv2.imread 실패")
            img_bgr = remove_background(img_bgr, self.use_rembg)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(img_rgb)
            tensor  = self.transform(pil_img)
        except Exception:
            tensor = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            label  = -1

        return tensor, label


def collate_skip_invalid(batch):
    batch = [(img, lbl) for img, lbl in batch if lbl >= 0]
    if not batch:
        return None
    imgs, lbls = zip(*batch)
    return torch.stack(imgs), torch.tensor(lbls)


# =========================
# 모델 로드 (여성 SWA 가중치)
# =========================

def load_female_model(swa_path: Path) -> nn.Module:
    """여성 SWA 모델을 전이학습 시작점으로 로드."""
    if not swa_path.exists():
        raise FileNotFoundError(
            f"여성 SWA 모델 없음: {swa_path}\n"
            "female/train.py 를 먼저 실행하세요."
        )
    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    ckpt = torch.load(swa_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[전이학습] 여성 SWA 모델 로드: {swa_path}")
    return model


def apply_freeze_strategy(model: nn.Module):
    """
    features[0~4]: 동결 (에지·질감·기본 얼굴 구조 — 성별 공통)
    features[5~7] + classifier: 파인튜닝 (고수준 윤곽 비율 — 성별 차이)
    """
    for name, param in model.named_parameters():
        frozen = any(f"features.{i}." in name for i in range(5))
        param.requires_grad = not frozen

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  동결: features[0~4] | 파인튜닝: features[5~7]+classifier")
    print(f"  파인튜닝 파라미터: {trainable:,} / {total:,} ({trainable/total:.1%})")


# =========================
# 학습 / 검증
# =========================

USE_AMP = DEVICE == "cuda"

def train_one_epoch(model, loader, optimizer, scaler, accum_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    for i, batch in enumerate(loader):
        if batch is None:
            continue
        imgs, lbls = batch
        imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)

        ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()
        with ctx:
            out  = model(imgs)
            loss = criterion(out, lbls) / accum_steps

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

        total_loss += loss.item() * accum_steps

    return total_loss / max(len(loader), 1)


@torch.inference_mode()
def validate(model, loader):
    model.eval()
    losses, preds, targets = [], [], []
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        if batch is None:
            continue
        imgs, lbls = batch
        imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()
        with ctx:
            out  = model(imgs)
            loss = criterion(out, lbls)
        losses.append(loss.item())
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(lbls.cpu().numpy())

    avg_loss = sum(losses) / max(len(losses), 1)
    acc      = accuracy_score(targets, preds) if targets else 0.0
    return avg_loss, acc


# =========================
# 메인 학습
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="얼굴형 분류 v1.9 남성 전이학습")
    parser.add_argument("--epochs",   type=int, default=FINETUNE_EPOCHS, help="파인튜닝 에폭 수")
    parser.add_argument("--no-rembg", action="store_true", help="배경 제거 비활성화")
    parser.add_argument("--resume",   action="store_true", help="checkpoint_latest.pth에서 이어서 학습")
    return parser.parse_args()


def main():
    args     = parse_args()
    use_rembg = not args.no_rembg

    if use_rembg and not _REMBG_AVAILABLE:
        print("[안내] rembg 미설치 → GrabCut 폴백")
        print("       pip install rembg onnxruntime  으로 설치 가능")

    # 데이터 수집
    if not MALE_TRAIN_DIR.exists():
        raise FileNotFoundError(f"남성 학습 데이터 없음: {MALE_TRAIN_DIR}")
    if not MALE_TEST_DIR.exists():
        raise FileNotFoundError(f"남성 테스트 데이터 없음: {MALE_TEST_DIR}")

    train_pairs = collect_male_pairs(MALE_TRAIN_DIR)
    test_pairs  = collect_male_pairs(MALE_TEST_DIR)
    print(f"\n남성 학습 데이터: {len(train_pairs)}장")
    print(f"남성 테스트 데이터: {len(test_pairs)}장")

    cls_dist = {}
    for _, cls in train_pairs:
        cls_dist[cls] = cls_dist.get(cls, 0) + 1
    print(f"클래스 분포: {cls_dist}")

    # DataLoader
    NUM_WORKERS = max(0, (os.cpu_count() or 1) - 1)
    PIN_MEMORY  = DEVICE == "cuda"

    train_ds = MaleFaceDataset(train_pairs, train_transform, use_rembg=use_rembg)
    test_ds  = MaleFaceDataset(test_pairs,  eval_transform,  use_rembg=use_rembg)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              drop_last=True, collate_fn=collate_skip_invalid)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              collate_fn=collate_skip_invalid)

    # 모델 준비
    model = load_female_model(FEMALE_SWA_MODEL)
    apply_freeze_strategy(model)
    model = model.to(DEVICE)

    # 옵티마이저 (head/backbone 차등 LR)
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "classifier" not in n]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and "classifier" in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": FINETUNE_LR_BACKBONE},
        {"params": head_params,     "lr": FINETUNE_LR_HEAD},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7,
    )
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    # SWA 설정
    swa_model     = AveragedModel(model)
    swa_scheduler = None

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_acc        = 0.0
    early_ctr       = 0
    patience        = 10
    swa_start       = max(args.epochs - SWA_EPOCHS, int(args.epochs * 0.55))
    swa_active      = False
    train_loss_hist, val_loss_hist, acc_hist = [], [], []

    start_epoch = 0
    if args.resume and CKPT_LATEST.exists():
        print(f"[Resume] {CKPT_LATEST} 로드 중...")
        res = torch.load(CKPT_LATEST, map_location=DEVICE, weights_only=False)
        model.load_state_dict(res["model_state_dict"])
        optimizer.load_state_dict(res["optimizer_state_dict"])
        scheduler.load_state_dict(res["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in res:
            scaler.load_state_dict(res["scaler_state_dict"])
        best_acc        = res["best_acc"]
        early_ctr       = res.get("early_ctr", 0)
        train_loss_hist = res.get("train_loss_hist", [])
        val_loss_hist   = res.get("val_loss_hist", [])
        acc_hist        = res.get("acc_hist", [])
        swa_active      = res.get("swa_active", False)
        start_epoch     = res["epoch"] + 1
        if swa_active:
            swa_model.load_state_dict(res["swa_model_state_dict"])
            swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=5)
            if "swa_scheduler_state_dict" in res:
                swa_scheduler.load_state_dict(res["swa_scheduler_state_dict"])
        print(f"[Resume] epoch {start_epoch + 1}/{args.epochs}부터 이어서 학습 (Best Acc: {best_acc:.4f})")
    elif args.resume:
        print("[Resume] checkpoint_latest.pth 없음 — 처음부터 학습")

    print(f"\n파인튜닝 시작 (총 {args.epochs} epoch, SWA는 {swa_start} epoch부터)")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, scaler, ACCUM_STEPS)
        avg_val_loss, accuracy = validate(model, test_loader)

        if epoch >= swa_start:
            if not swa_active:
                swa_active    = True
                swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=5)
                print(f"\n  [SWA 시작] epoch {epoch + 1}")
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step(epoch)

        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)
        acc_hist.append(accuracy)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | Acc {accuracy:.4f}",
              end="")

        if accuracy > best_acc:
            best_acc  = accuracy
            early_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "classes": CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "male_class_map": MALE_CLASS_MAP,
            }, CKPT_BEST)
            print("  ← Best", end="")
        else:
            early_ctr += 1

        print()

        if (epoch + 1) % CKPT_INTERVAL == 0:
            periodic = CKPT_DIR / f"checkpoint_epoch_{epoch+1:04d}.pth"
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "best_acc": best_acc, "classes": CLASSES,
            }, periodic)

        # 이어서 학습을 위한 최신 체크포인트 저장 (매 epoch)
        latest_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "early_ctr": early_ctr,
            "train_loss_hist": train_loss_hist,
            "val_loss_hist": val_loss_hist,
            "acc_hist": acc_hist,
            "swa_active": swa_active,
        }
        if scaler:
            latest_state["scaler_state_dict"] = scaler.state_dict()
        if swa_active:
            latest_state["swa_model_state_dict"] = swa_model.state_dict()
            if swa_scheduler:
                latest_state["swa_scheduler_state_dict"] = swa_scheduler.state_dict()
        torch.save(latest_state, CKPT_LATEST)

        if not swa_active and early_ctr >= patience:
            if epoch + 1 < swa_start:
                # SWA 구간에 아직 못 들어갔으면 멈추지 않고 SWA까지 버팀
                if early_ctr == patience:
                    print(f"  [EarlyStopping 대기] {patience}연속 개선 없음, SWA 구간(epoch {swa_start+1})까지 유지")
            else:
                print(f"\n[EarlyStopping] {patience} epoch 개선 없음 → 종료")
                break

    # SWA BN 업데이트 & 저장
    if swa_active:
        print("\n[SWA] BatchNorm 통계 업데이트...")
        update_bn(train_loader, swa_model, device=DEVICE)
        torch.save({
            "model_state_dict": swa_model.module.state_dict(),
            "classes": CLASSES,
            "class_to_idx": CLASS_TO_IDX,
            "male_class_map": MALE_CLASS_MAP,
        }, CKPT_SWA)
        print(f"[SWA] 저장: {CKPT_SWA}")

        # SWA 최종 정확도 평가
        swa_inner = swa_model.module
        _, swa_acc = validate(swa_inner, test_loader)
        print(f"[SWA] 테스트 정확도: {swa_acc:.4f}")

    print(f"\n학습 완료 — Best Accuracy: {best_acc:.4f}")
    print(f"  Best 모델: {CKPT_BEST}")

    # 최종 분류 리포트 (Best 모델)
    if CKPT_BEST.exists():
        ckpt = torch.load(CKPT_BEST, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        all_preds, all_targets = [], []
        with torch.inference_mode():
            for batch in test_loader:
                if batch is None:
                    continue
                imgs, lbls = batch
                imgs = imgs.to(DEVICE)
                out  = model(imgs)
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_targets.extend(lbls.numpy())
        active_classes = [c for c in CLASSES if c != "Heart"]
        print("\n분류 리포트 (Best 모델):")
        print(classification_report(
            [CLASSES[i] for i in all_targets],
            [CLASSES[i] for i in all_preds],
            labels=active_classes,
            digits=4,
        ))

    # 학습 곡선 저장
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_loss_hist, label="Train Loss", alpha=0.8)
    ax1.plot(val_loss_hist,   label="Val Loss",   alpha=0.8)
    ax1.axvline(x=swa_start - 1, color="gray", ls="--", label="SWA 시작")
    ax1.set_title("Loss"), ax1.legend()
    ax2.plot(acc_hist, color="green", label="Accuracy", alpha=0.8)
    ax2.axvline(x=swa_start - 1, color="gray", ls="--", label="SWA 시작")
    ax2.axhline(y=0.85, color="red", ls=":", alpha=0.5, label="85% 목표")
    ax2.set_title("Validation Accuracy"), ax2.legend()
    fig.tight_layout()
    curve_path = CKPT_DIR / "training_curve.png"
    fig.savefig(curve_path, dpi=120)
    print(f"학습 곡선 저장: {curve_path}")


if __name__ == "__main__":
    main()
