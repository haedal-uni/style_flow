"""
v22 여성 얼굴형 학습 — MediaPipe 기하학 + EfficientNetV2-S CNN 하이브리드 + Heart 게이트

클래스: Heart / Oblong / Oval / Round / Square (5클래스)
데이터: dataset/training_set/, dataset/testing_set/

v21 대비 변경점:
  1. HeartGate 추가 — 기하학적으로 Heart 신호가 강한 표본은 geo 정보에
     더 의존하도록 모델이 스스로 가중치를 조절 (common.py 참고)
  2. 3단계 앞머리 분류 — 완전 가림만 제외, 부분 가림은 가중치 0.5로 포함
  3. 표본 가중치를 반영한 weighted CrossEntropy 적용
  4. 클래스 분포 기반 class_weight는 그대로 유지 (Heart/Oval 상향)

실행 (uv 사용):
  uv run python v22/female/train.py
  uv run python v22/female/train.py --resume
  uv run python v22/female/train.py --no-rembg
  uv run python v22/female/train.py --epochs 80
"""

import os, sys, random, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))         # v22/ 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # 프로젝트 루트 추가

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import torch
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from common import (
    DEVICE, USE_AMP, IMG_EXTS, GEO_TOTAL, FEATURE_NAMES,
    _REMBG_AVAILABLE, get_landmarker,
    build_cache, FaceShapeNet, FaceDataset, collate_fn,
    TRAIN_TRANSFORM, EVAL_TRANSFORM,
    train_one_epoch, validate, update_bn_two_input,
)

# ── 경로 ──────────────────────────────────────────────────────
_HERE = Path(__file__).parent          # v22/female/
_ROOT = _HERE.parent.parent            # 프로젝트 루트

TRAIN_DIR = _ROOT / "dataset" / "training_set"
TEST_DIR  = _ROOT / "dataset" / "testing_set"
CACHE_DIR = _HERE / ".cache"

CKPT_BEST   = _HERE / "best_model.pth"
CKPT_SWA    = _HERE / "swa_model.pth"
CKPT_LATEST = _HERE / "checkpoint_latest.pth"
GEO_SCALER  = _HERE / "geo_scaler.pkl"
GEO_MEDIANS = _HERE / "class_geo_medians.pkl"
CURVE_PNG   = _HERE / "training_curve.png"

# ── 클래스 ────────────────────────────────────────────────────
CLASSES      = ["Heart", "Oblong", "Oval", "Round", "Square"]
NUM_CLASSES  = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Heart/Oval 상향 가중치 (v19/v21 실험 기반 — 가장 혼동이 잦은 클래스)
_raw_w  = [1 / 0.75, 1 / 0.91, 1 / 0.83, 1 / 0.90, 1 / 0.92]
_mean_w = sum(_raw_w) / len(_raw_w)
CLASS_WEIGHTS = [w / _mean_w for w in _raw_w]

# ── 하이퍼파라미터 ─────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 8
ACCUM_STEPS  = 4        # 실효 배치 32
TOTAL_EPOCHS = 120
SWA_EPOCHS   = 20
PHASE1_END   = 10       # Phase1: head + geo_enc + heart_gate만 학습
PHASE3_START = 60       # Phase3: 전체 backbone 언프리즈
PATIENCE     = 15
MIXUP_ALPHA  = 0.2      # Mixup augmentation (Heart/Oval 경계 강화)

LR_HEAD     = 1e-3
LR_BACKBONE = 2e-5
LR_PHASE1   = 1e-3
LR_PHASE3   = 5e-6      # 전체 backbone 미세조정용 (매우 낮게)
SWA_LR      = 5e-6

NUM_WORKERS = max(0, (os.cpu_count() or 1) - 1)
PIN_MEMORY  = DEVICE == "cuda"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── 데이터 수집 ────────────────────────────────────────────────
def collect_pairs(root: Path) -> list:
    pairs = []
    for cls in CLASSES:
        for name in [cls, cls.lower(), cls.upper()]:
            d = root / name
            if d.is_dir():
                for p in sorted(d.iterdir()):
                    if p.suffix.lower() in IMG_EXTS:
                        pairs.append((p, cls))
                break
    return pairs


# ── Phase1: head + geo_enc + heart_gate만 학습 ────────────────
def set_phase1(model: FaceShapeNet):
    for name, param in model.named_parameters():
        param.requires_grad = (
            "classifier" in name or "geo_enc" in name or "heart_gate" in name
        )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Phase1] head+geo_enc+heart_gate만 학습: {n:,}파라미터")


# ── Phase2: features[5-7] + classifier + geo_enc + heart_gate ─
def set_phase2(model: FaceShapeNet):
    for name, param in model.named_parameters():
        frozen = any(f"backbone.features.{i}." in name for i in range(5))
        param.requires_grad = not frozen
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Phase2] features[5-7]+head+geo_enc+heart_gate: {n:,}파라미터")


# ── Phase3: 전체 backbone 언프리즈 (매우 낮은 LR) ─────────────
def set_phase3(model: FaceShapeNet):
    for param in model.parameters():
        param.requires_grad = True
    n = sum(p.numel() for p in model.parameters())
    print(f"  [Phase3] 전체 backbone 언프리즈: {n:,}파라미터")


# ── 클래스별 기하학 중앙값 계산 (추론 설명용) ─────────────────
def compute_class_geo_medians(records: list, classes: list) -> dict:
    """
    학습 후 infer.py의 explain_prediction()이 참고할 클래스별 기하학 중앙값을
    저장한다. 정규화 전 원본 스케일로 저장해야 사람이 읽을 때 의미가 있으므로
    geo_scaler.inverse_transform으로 역변환한다.
    """
    by_class = {c: [] for c in classes}
    for _, cls, geo18, _w in records:
        if geo18[-1] > 0.5:   # 유효 플래그가 1인(geo 측정 성공) 표본만
            by_class[cls].append(geo18[:-1])  # 마지막 유효플래그 제외
    medians = {}
    for c, arr in by_class.items():
        if arr:
            medians[c] = np.median(np.array(arr), axis=0)
    return medians


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",   type=int, default=TOTAL_EPOCHS)
    p.add_argument("--no-rembg", action="store_true")
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    use_rembg = not args.no_rembg
    _HERE.mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"rembg  : {'사용' if (_REMBG_AVAILABLE and use_rembg) else 'GrabCut 폴백'}")

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"학습 데이터 없음: {TRAIN_DIR}\n"
            f"dataset/training_set/{{Heart,Oblong,Oval,Round,Square}}/ 구조로 배치해주세요."
        )
    train_pairs = collect_pairs(TRAIN_DIR)
    test_pairs  = collect_pairs(TEST_DIR)
    print(f"여성 학습: {len(train_pairs)}장  |  테스트: {len(test_pairs)}장")
    dist = {}
    for _, c in train_pairs:
        dist[c] = dist.get(c, 0) + 1
    print(f"클래스 분포: {dist}")

    # 전처리 캐시 빌드 (배경 제거 + 기하학 특징 + 3단계 앞머리 가중치)
    print("\n[전처리] 배경 제거 + 기하학 특징 추출 + 앞머리 3단계 분류...")
    get_landmarker()

    train_records, geo_scaler, tr_stats = build_cache(
        train_pairs, CACHE_DIR / "train", use_rembg, fit_scaler=True, cache_tag="v22",
    )
    print(f"  학습: ok={tr_stats['ok']} (정상={tr_stats['normal']}, 부분가림={tr_stats['partial']}) "
          f"완전가림제외={tr_stats['bangs']} 얼굴미탐지={tr_stats['no_face']} 오류={tr_stats['error']}")

    test_records, _, te_stats = build_cache(
        test_pairs, CACHE_DIR / "test", use_rembg, geo_scaler=geo_scaler, cache_tag="v22",
    )
    print(f"  테스트: ok={te_stats['ok']} (정상={te_stats['normal']}, 부분가림={te_stats['partial']}) "
          f"완전가림제외={te_stats['bangs']}")

    joblib.dump(geo_scaler, GEO_SCALER)

    # 추론 설명용 클래스별 기하학 중앙값 저장
    medians = compute_class_geo_medians(train_records, CLASSES)
    # 정규화된 값이므로 원본 스케일로 역변환
    medians_raw = {}
    for c, m in medians.items():
        medians_raw[c] = dict(zip(FEATURE_NAMES, geo_scaler.inverse_transform([m])[0]))
    joblib.dump(medians_raw, GEO_MEDIANS)
    print(f"  클래스별 기하학 중앙값 저장: {GEO_MEDIANS}")

    # DataLoader
    train_ds = FaceDataset(train_records, CLASS_TO_IDX, TRAIN_TRANSFORM)
    test_ds  = FaceDataset(test_records,  CLASS_TO_IDX, EVAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             collate_fn=collate_fn)

    # 모델
    model = FaceShapeNet(NUM_CLASSES).to(DEVICE)
    swa_model     = AveragedModel(model)
    swa_scheduler = None

    cw_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)

    # Phase1 옵티마이저
    set_phase1(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1, weight_decay=0.01,
    )
    scaler_amp = torch.amp.GradScaler("cuda") if USE_AMP else None

    swa_start = max(args.epochs - SWA_EPOCHS, int(args.epochs * 0.75))

    best_acc, early_ctr = 0.0, 0
    swa_active, phase2_started, phase3_started = False, False, False
    train_loss_hist, val_loss_hist, acc_hist = [], [], []
    start_epoch = 0

    # Resume
    if args.resume and CKPT_LATEST.exists():
        print(f"\n[Resume] {CKPT_LATEST} 로드...")
        ck = torch.load(CKPT_LATEST, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if scaler_amp and "scaler_state_dict" in ck:
            scaler_amp.load_state_dict(ck["scaler_state_dict"])
        best_acc        = ck["best_acc"]
        early_ctr       = ck.get("early_ctr", 0)
        train_loss_hist = ck.get("train_loss_hist", [])
        val_loss_hist   = ck.get("val_loss_hist", [])
        acc_hist        = ck.get("acc_hist", [])
        swa_active      = ck.get("swa_active", False)
        phase2_started  = ck.get("phase2_started", False)
        phase3_started  = ck.get("phase3_started", False)
        start_epoch     = ck["epoch"] + 1
        if swa_active:
            swa_model.load_state_dict(ck["swa_model_state_dict"])
            swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=5)
            if "swa_scheduler_state_dict" in ck:
                swa_scheduler.load_state_dict(ck["swa_scheduler_state_dict"])
        if phase3_started and not swa_active:
            set_phase3(model)
        elif phase2_started and not swa_active:
            set_phase2(model)
        print(f"  epoch {start_epoch + 1}부터 이어서 (Best Acc: {best_acc:.4f})")
    elif args.resume:
        print("[Resume] checkpoint_latest.pth 없음 — 처음부터 시작")

    phase2_scheduler = None

    print(f"\n학습 시작 (총 {args.epochs} epoch | SWA: epoch {swa_start+1}~)")
    print("=" * 62)

    for epoch in range(start_epoch, args.epochs):

        # Phase 전환
        if epoch == PHASE1_END and not phase2_started:
            print(f"\n  [Phase2 시작] epoch {epoch+1} — features[5-7]+head+geo_enc+heart_gate 언프리즈")
            set_phase2(model)
            phase2_started = True
            backbone_params = [p for n, p in model.named_parameters()
                               if p.requires_grad and "backbone" in n]
            head_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and "backbone" not in n]
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": LR_BACKBONE},
                {"params": head_params,     "lr": LR_HEAD},
            ], weight_decay=0.01)
            phase2_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=25, T_mult=2, eta_min=1e-7,
            )
            scaler_amp = torch.amp.GradScaler("cuda") if USE_AMP else None

        if epoch == PHASE3_START and not phase3_started and not swa_active:
            print(f"\n  [Phase3 시작] epoch {epoch+1} — 전체 backbone 언프리즈 (LR={LR_PHASE3})")
            set_phase3(model)
            phase3_started = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR_PHASE3, weight_decay=0.01,
            )
            phase2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=swa_start - PHASE3_START, eta_min=1e-7,
            )

        # SWA 전환
        if epoch >= swa_start:
            if not swa_active:
                swa_active    = True
                swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=5)
                print(f"\n  [SWA 시작] epoch {epoch+1}")
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif phase2_started and phase2_scheduler:
            phase2_scheduler.step()

        # 학습 / 검증
        avg_train = train_one_epoch(
            model, train_loader, optimizer, scaler_amp, ACCUM_STEPS,
            class_weight_tensor=cw_tensor,
            mixup_alpha=MIXUP_ALPHA if phase2_started else 0.0,
        )
        avg_val, acc = validate(model, test_loader)

        train_loss_hist.append(avg_train)
        val_loss_hist.append(avg_val)
        acc_hist.append(acc)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train {avg_train:.4f} | Val {avg_val:.4f} | Acc {acc:.4f}", end="")

        if acc > best_acc:
            best_acc, early_ctr = acc, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "classes": CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "geo_total": GEO_TOTAL,
            }, CKPT_BEST)
            print("  ← Best", end="")
        else:
            early_ctr += 1
        print()

        # Checkpoint (latest)
        latest = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc, "early_ctr": early_ctr,
            "train_loss_hist": train_loss_hist,
            "val_loss_hist": val_loss_hist, "acc_hist": acc_hist,
            "swa_active": swa_active, "phase2_started": phase2_started,
            "phase3_started": phase3_started,
        }
        if scaler_amp:
            latest["scaler_state_dict"] = scaler_amp.state_dict()
        if swa_active:
            latest["swa_model_state_dict"] = swa_model.state_dict()
            if swa_scheduler:
                latest["swa_scheduler_state_dict"] = swa_scheduler.state_dict()
        torch.save(latest, CKPT_LATEST)

        # Early stopping (SWA 전 구간에서만)
        if not swa_active and early_ctr >= PATIENCE:
            if epoch < swa_start:
                if early_ctr == PATIENCE:
                    print(f"  [EarlyStopping 대기] {PATIENCE}연속 개선 없음, SWA(epoch {swa_start+1})까지 유지")
            else:
                print(f"\n[EarlyStopping] {PATIENCE} epoch 개선 없음 → 종료")
                break

    # SWA BN 업데이트
    if swa_active:
        print("\n[SWA] BatchNorm 업데이트...")
        update_bn_two_input(train_loader, swa_model, DEVICE)
        torch.save({
            "model_state_dict": swa_model.module.state_dict(),
            "classes": CLASSES,
            "class_to_idx": CLASS_TO_IDX,
            "geo_total": GEO_TOTAL,
        }, CKPT_SWA)
        print(f"  저장: {CKPT_SWA}")
        _, swa_acc = validate(swa_model.module, test_loader)
        print(f"  SWA 정확도: {swa_acc:.4f}")

    print(f"\n학습 완료 — Best Accuracy: {best_acc:.4f}")

    # 분류 리포트
    if CKPT_BEST.exists():
        ck = torch.load(CKPT_BEST, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        all_preds, all_targets = [], []
        with torch.inference_mode():
            for batch in test_loader:
                if batch is None:
                    continue
                imgs, geos, lbls, _w = batch
                out, _gate = model(imgs.to(DEVICE), geos.to(DEVICE))
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_targets.extend(lbls.numpy())
        print("\n분류 리포트 (Best 모델):")
        print(classification_report(
            [CLASSES[i] for i in all_targets],
            [CLASSES[i] for i in all_preds],
            labels=CLASSES, digits=4,
        ))

    # 학습 곡선
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_loss_hist, label="Train Loss", alpha=0.8)
    ax1.plot(val_loss_hist,   label="Val Loss",   alpha=0.8)
    ax1.axvline(x=PHASE1_END - 1, color="blue", ls=":", alpha=0.6, label="Phase2 Start")
    ax1.axvline(x=swa_start - 1,  color="gray", ls="--",            label="SWA Start")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(acc_hist, color="green", label="Accuracy")
    ax2.axvline(x=swa_start - 1, color="gray", ls="--", label="SWA Start")
    ax2.axhline(y=0.90, color="red", ls=":", alpha=0.5, label="90% Target")
    ax2.set_title("Validation Accuracy"); ax2.legend()
    fig.tight_layout()
    fig.savefig(CURVE_PNG, dpi=120)
    print(f"학습 곡선: {CURVE_PNG}")
    print(f"\n다음 단계: 남성 전이학습 실행")
    print(f"  uv run python v22/male/train.py")


if __name__ == "__main__":
    main()
