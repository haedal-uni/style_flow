"""
v22/male/train.py
남성 얼굴형 전이학습 — HeadOnly 안정형 + exclude_list.txt 자동 제외

목적
- 여성 모델 전이학습 유지
- 여성 5클래스 구조 유지
- backbone은 freeze
- classifier + geo_enc + heart_gate만 학습
- 오분류 분석 후 exclude_list.txt에 적은 이미지 자동 제외
- 제외 후 다시 학습해서 라벨 애매한 샘플 영향을 줄임

사용 흐름
1. 오분류 이미지 추출
   uv run python v22/male/export_misclassified.py --clear

2. misclassified 폴더 확인 후 제외할 이미지 경로/파일명을 exclude_list.txt에 작성
   예:
     Oval_to_Oblong/0032_true-Oval_pred-Oblong_conf-0.821_xxx.png
     xxx.png
     /home/ai/v20/v22/male/misclassified/Oval_to_Oblong/0032_true-Oval_pred-Oblong_conf-0.821_xxx.png

3. 다시 학습
   uv run python v22/male/train.py --no-cache

exclude_list.txt 위치
- 기본: v22/male/exclude_list.txt
- 옵션으로 변경 가능:
  uv run python v22/male/train.py --exclude-list v22/male/exclude_list.txt

실행
  uv run python v22/male/train.py --no-cache

비교 실험
  uv run python v22/male/train.py --no-cache --mixup 0
  uv run python v22/male/train.py --no-cache --epochs 25 --patience 6
  uv run python v22/male/train.py --no-cache --fresh-scaler
"""

import os
import sys
import csv
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))         # v22/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from common import (
    DEVICE, USE_AMP, IMG_EXTS, GEO_TOTAL, FEATURE_NAMES,
    _REMBG_AVAILABLE, get_landmarker,
    build_cache, FaceShapeNet, FaceDataset, collate_fn,
    TRAIN_TRANSFORM, EVAL_TRANSFORM,
    train_one_epoch,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 경로
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent

MALE_TRAIN_DIR = _ROOT / "dataset" / "men" / "training_set"
MALE_TEST_DIR  = _ROOT / "dataset" / "men" / "testing_set"
CACHE_DIR      = _HERE / ".cache_headonly_exclude_safe"

FEMALE_DIR    = _HERE.parent / "female"
FEMALE_SWA    = FEMALE_DIR / "swa_model.pth"
FEMALE_BEST   = FEMALE_DIR / "best_model.pth"
FEMALE_SCALER = FEMALE_DIR / "geo_scaler.pkl"

DEFAULT_EXCLUDE_LIST = _HERE / "exclude_list.txt"

CKPT_BEST   = _HERE / "best_model.pth"
CKPT_LATEST = _HERE / "checkpoint_latest.pth"
GEO_SCALER  = _HERE / "geo_scaler.pkl"
GEO_MEDIANS = _HERE / "class_geo_medians.pkl"
CURVE_PNG   = _HERE / "training_curve.png"
CONF_PNG    = _HERE / "confusion_matrix.png"
TRAIN_SUMMARY_JSON = _HERE / "train_summary.json"
EXCLUDED_REPORT_CSV = _HERE / "excluded_applied_report.csv"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLASSES = ["Heart", "Oblong", "Oval", "Round", "Square"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

MALE_CLASSES = ["Oblong", "Oval", "Round", "Square"]

MALE_CLASS_MAP = {
    "rectangular": "Oblong",
    "ovale": "Oval",
    "round": "Round",
    "square": "Square",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 하이퍼파라미터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEED = 42

BATCH_SIZE = 8
ACCUM_STEPS = 4

TOTAL_EPOCHS = 40
PATIENCE = 8

LR_HEAD = 8e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.10

NUM_WORKERS = max(0, (os.cpu_count() or 1) - 1)
PIN_MEMORY = DEVICE == "cuda"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 기본 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_key(s: str) -> str:
    return str(s).replace("\\", "/").strip().lower()


def load_exclude_rules(exclude_path: Path):
    """
    안전한 exclude 규칙 로더.

    지원 형식:
      stem:원본파일stem      ← 권장
      name:원본파일명.jpg
      path:/full/path/file.jpg
      원본파일명.jpg          ← name으로 처리
      원본파일stem           ← stem으로 처리

    중요:
    부분 문자열 포함 매칭을 하지 않는다.
    이전처럼 후보 19장인데 128장이 제외되는 문제를 막기 위해
    path/name/stem의 정확 일치만 사용한다.
    """
    rules = {
        "stem": set(),
        "name": set(),
        "path": set(),
    }

    if not exclude_path.exists():
        print(f"[Exclude] 파일 없음 — 제외 없이 진행: {exclude_path}")
        return rules

    raw_count = 0

    with exclude_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            raw_count += 1

            if "," in raw:
                raw = raw.split(",", 1)[0].strip()

            raw_norm = normalize_key(raw)

            if raw_norm.startswith("stem:"):
                value = raw_norm.split("stem:", 1)[1].strip()
                if value:
                    rules["stem"].add(value)

            elif raw_norm.startswith("name:"):
                value = raw_norm.split("name:", 1)[1].strip()
                if value:
                    p = Path(value)
                    rules["name"].add(p.name.lower())
                    rules["stem"].add(p.stem.lower())

            elif raw_norm.startswith("path:"):
                value = raw_norm.split("path:", 1)[1].strip()
                if value:
                    p = Path(value)
                    rules["path"].add(value)
                    rules["name"].add(p.name.lower())
                    rules["stem"].add(p.stem.lower())

            else:
                # 확장자가 있으면 파일명으로, 없으면 stem으로 처리
                p = Path(raw_norm)
                if p.suffix.lower() in IMG_EXTS:
                    rules["name"].add(p.name.lower())
                    rules["stem"].add(p.stem.lower())
                else:
                    rules["stem"].add(raw_norm)

    print(f"[Exclude] 원본 규칙 {raw_count}줄 로드: {exclude_path}")
    print(
        f"[Exclude] parsed: "
        f"stem={len(rules['stem'])}, "
        f"name={len(rules['name'])}, "
        f"path={len(rules['path'])}"
    )

    return rules


def is_excluded(path: Path, rules: dict) -> bool:
    """
    정확 매칭만 수행.
    - absolute path exact
    - filename exact
    - stem exact
    """
    try:
        abs_path = normalize_key(str(path.resolve()))
    except Exception:
        abs_path = normalize_key(str(path))

    name = path.name.lower()
    stem = path.stem.lower()

    if abs_path in rules["path"]:
        return True
    if name in rules["name"]:
        return True
    if stem in rules["stem"]:
        return True

    return False


def apply_exclude(pairs: list, rules: dict, split_name: str):
    kept = []
    excluded = []

    for path, cls in pairs:
        if is_excluded(path, rules):
            excluded.append((path, cls, split_name))
        else:
            kept.append((path, cls))

    return kept, excluded


def save_excluded_report(excluded_all: list):
    with EXCLUDED_REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "path", "filename"])
        writer.writeheader()
        for path, cls, split in excluded_all:
            writer.writerow({
                "split": split,
                "class": cls,
                "path": str(path),
                "filename": path.name,
            })

    print(f"[Exclude] 적용 리포트 저장: {EXCLUDED_REPORT_CSV}")


def collect_male_pairs(root: Path) -> list:
    pairs = []

    for folder_name, class_name in MALE_CLASS_MAP.items():
        found_dir = None

        for candidate in [
            folder_name,
            folder_name.lower(),
            folder_name.upper(),
            folder_name.capitalize(),
        ]:
            d = root / candidate
            if d.is_dir():
                found_dir = d
                break

        if found_dir is None:
            print(f"[경고] 폴더 없음: {root / folder_name}")
            continue

        for p in sorted(found_dir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                pairs.append((p, class_name))

    return pairs


def print_distribution(pairs: list, title: str):
    dist = {c: 0 for c in MALE_CLASSES}
    for _, cls in pairs:
        dist[cls] = dist.get(cls, 0) + 1
    print(f"{title}: {dist}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 / 학습 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_female_transfer_weights(model: FaceShapeNet):
    ckpt_path = FEMALE_SWA if FEMALE_SWA.exists() else FEMALE_BEST

    if not ckpt_path.exists():
        raise FileNotFoundError(
            "여성 전이학습 가중치를 찾을 수 없습니다.\n"
            f"확인 경로 1: {FEMALE_SWA}\n"
            f"확인 경로 2: {FEMALE_BEST}\n\n"
            "먼저 여성 모델을 학습하세요:\n"
            "  uv run python v22/female/train.py"
        )

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model.load_state_dict(state, strict=True)

    print(f"[전이학습] 여성 모델 로드: {ckpt_path}")


def set_head_only(model: FaceShapeNet):
    for name, param in model.named_parameters():
        param.requires_grad = (
            "classifier" in name or
            "geo_enc" in name or
            "heart_gate" in name
        )

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  [HeadOnly] classifier + geo_enc + heart_gate 학습: {n_train:,} / {n_total:,}")


def make_class_weights(train_pairs: list) -> list:
    counts = {c: 0 for c in CLASSES}

    for _, cls in train_pairs:
        counts[cls] += 1

    valid_counts = [counts[c] for c in MALE_CLASSES if counts[c] > 0]
    mean_count = float(np.mean(valid_counts)) if valid_counts else 1.0

    weights = []
    for c in CLASSES:
        if c == "Heart":
            weights.append(1.0)
        elif counts[c] > 0:
            weights.append(mean_count / counts[c])
        else:
            weights.append(1.0)

    male_mean = float(np.mean([weights[CLASS_TO_IDX[c]] for c in MALE_CLASSES]))
    weights = [float(w / male_mean) for w in weights]

    print("Class weights:")
    for c, w in zip(CLASSES, weights):
        print(f"  {c:7s}: {w:.4f}")

    return weights


def make_weighted_sampler(records: list):
    counts = {c: 0 for c in MALE_CLASSES}
    for _, cls, _, _ in records:
        if cls in counts:
            counts[cls] += 1

    sample_weights = []
    for _, cls, _, sample_weight in records:
        cls_count = max(counts.get(cls, 1), 1)
        sample_weights.append((1.0 / cls_count) * float(sample_weight))

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_class_geo_medians(records: list, geo_scaler) -> dict:
    if geo_scaler is None:
        return {}

    by_class = {c: [] for c in MALE_CLASSES}

    for _, cls, geo18, _w in records:
        if cls in by_class and geo18[-1] > 0.5:
            by_class[cls].append(geo18[:-1])

    medians_raw = {}
    for c, arr in by_class.items():
        if arr:
            median_scaled = np.median(np.array(arr), axis=0)
            median_raw = geo_scaler.inverse_transform([median_scaled])[0]
            medians_raw[c] = dict(zip(FEATURE_NAMES, median_raw))

    return medians_raw


@torch.inference_mode()
def validate_male(model, loader):
    import torch.nn.functional as F

    model.eval()
    losses = []
    preds = []
    targets = []

    for batch in loader:
        if batch is None:
            continue

        imgs, geos, lbls, _weights = batch
        imgs = imgs.to(DEVICE, non_blocking=True)
        geos = geos.to(DEVICE, non_blocking=True)
        lbls = lbls.to(DEVICE, non_blocking=True)

        ctx = torch.autocast("cuda", torch.float16) if USE_AMP else torch.no_grad()

        with ctx:
            out, _gate = model(imgs, geos)
            loss = F.cross_entropy(out, lbls)

        losses.append(loss.item())
        preds.extend(out.argmax(1).detach().cpu().numpy())
        targets.extend(lbls.detach().cpu().numpy())

    avg_loss = sum(losses) / max(len(losses), 1)
    acc = accuracy_score(targets, preds) if targets else 0.0
    return avg_loss, acc


@torch.inference_mode()
def collect_predictions(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        if batch is None:
            continue

        imgs, geos, lbls, _weights = batch
        imgs = imgs.to(DEVICE, non_blocking=True)
        geos = geos.to(DEVICE, non_blocking=True)

        out, _gate = model(imgs, geos)

        all_preds.extend(out.argmax(1).detach().cpu().numpy())
        all_targets.extend(lbls.numpy())

    y_true = [CLASSES[i] for i in all_targets]
    y_pred = [CLASSES[i] for i in all_preds]

    return y_true, y_pred


def save_training_curve(train_hist, val_hist, acc_hist, best_epoch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_hist, label="Train Loss", alpha=0.85)
    ax1.plot(val_hist, label="Val Loss", alpha=0.85)
    ax1.axvline(x=best_epoch, color="green", ls=":", alpha=0.7, label="Best")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(acc_hist, color="green", label="Accuracy")
    ax2.axvline(x=best_epoch, color="green", ls=":", alpha=0.7, label="Best")
    ax2.axhline(y=0.75, color="orange", ls=":", alpha=0.5, label="75%")
    ax2.axhline(y=0.85, color="red", ls=":", alpha=0.5, label="85% Target")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(CURVE_PNG, dpi=120)
    plt.close(fig)

    print(f"학습 곡선 저장: {CURVE_PNG}")


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=MALE_CLASSES)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_xticks(range(len(MALE_CLASSES)))
    ax.set_yticks(range(len(MALE_CLASSES)))
    ax.set_xticklabels(MALE_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(MALE_CLASSES)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Male Face Shape Confusion Matrix")

    for i in range(len(MALE_CLASSES)):
        for j in range(len(MALE_CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CONF_PNG, dpi=120)
    plt.close(fig)

    print(f"혼동행렬 저장: {CONF_PNG}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# args
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-rembg", action="store_true")
    p.add_argument("--no-cache", action="store_true")

    p.add_argument(
        "--fresh-scaler",
        action="store_true",
        help="여성 geo_scaler를 쓰지 않고 남성 train set으로 새 scaler fit",
    )

    p.add_argument(
        "--no-sampler",
        action="store_true",
        help="WeightedRandomSampler 비활성화",
    )

    p.add_argument("--lr", type=float, default=LR_HEAD)
    p.add_argument("--mixup", type=float, default=MIXUP_ALPHA)

    p.add_argument(
        "--exclude-list",
        type=str,
        default=str(DEFAULT_EXCLUDE_LIST),
        help="제외할 이미지 목록 txt 경로",
    )

    p.add_argument(
        "--no-exclude",
        action="store_true",
        help="exclude_list.txt를 무시하고 전체 데이터 사용",
    )

    return p.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    args = parse_args()
    set_seed(SEED)

    use_rembg = not args.no_rembg
    _HERE.mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"rembg  : {'사용' if (_REMBG_AVAILABLE and use_rembg) else 'GrabCut 폴백'}")

    if not MALE_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"남성 학습 데이터 없음: {MALE_TRAIN_DIR}\n"
            "dataset/men/training_set/{ovale,rectangular,round,square}/ 구조로 배치하세요."
        )

    if not MALE_TEST_DIR.exists():
        raise FileNotFoundError(
            f"남성 테스트 데이터 없음: {MALE_TEST_DIR}\n"
            "dataset/men/testing_set/{ovale,rectangular,round,square}/ 구조로 배치하세요."
        )

    train_pairs_raw = collect_male_pairs(MALE_TRAIN_DIR)
    test_pairs_raw = collect_male_pairs(MALE_TEST_DIR)

    print(f"남성 학습 원본: {len(train_pairs_raw)}장 | 테스트 원본: {len(test_pairs_raw)}장")
    print_distribution(train_pairs_raw, "학습 원본 클래스 분포")
    print_distribution(test_pairs_raw, "테스트 원본 클래스 분포")

    # exclude 적용
    excluded_all = []
    if args.no_exclude:
        print("[Exclude] --no-exclude 지정됨 — 제외 없이 진행")
        train_pairs = train_pairs_raw
        test_pairs = test_pairs_raw
    else:
        exclude_rules = load_exclude_rules(Path(args.exclude_list))
        train_pairs, excluded_train = apply_exclude(train_pairs_raw, exclude_rules, "train")
        test_pairs, excluded_test = apply_exclude(test_pairs_raw, exclude_rules, "test")
        excluded_all = excluded_train + excluded_test

        print(f"[Exclude] 학습 제외: {len(excluded_train)}장")
        print(f"[Exclude] 테스트 제외: {len(excluded_test)}장")
        print(f"[Exclude] 총 제외: {len(excluded_all)}장")

        if excluded_all:
            save_excluded_report(excluded_all)

    print(f"남성 학습 사용: {len(train_pairs)}장 | 테스트 사용: {len(test_pairs)}장")
    print_distribution(train_pairs, "학습 사용 클래스 분포")
    print_distribution(test_pairs, "테스트 사용 클래스 분포")

    if not train_pairs:
        raise RuntimeError("exclude 적용 후 남성 학습 이미지가 0장입니다.")
    if not test_pairs:
        raise RuntimeError("exclude 적용 후 남성 테스트 이미지가 0장입니다.")

    # scaler
    if args.fresh_scaler:
        geo_scaler = None
        fit_scaler = True
        print("[Scaler] 남성 train set으로 geo_scaler 새로 fit")
    else:
        if not FEMALE_SCALER.exists():
            raise FileNotFoundError(
                f"여성 geo_scaler가 없습니다: {FEMALE_SCALER}\n"
                "남성 scaler로 새로 fit하려면 --fresh-scaler 옵션을 사용하세요."
            )
        geo_scaler = joblib.load(FEMALE_SCALER)
        fit_scaler = False
        print(f"[Scaler] 여성 geo_scaler 재사용: {FEMALE_SCALER}")

    # cache 삭제
    if args.no_cache:
        import shutil
        shutil.rmtree(CACHE_DIR / "train", ignore_errors=True)
        shutil.rmtree(CACHE_DIR / "test", ignore_errors=True)
        print("[Cache] 기존 exclude cache 삭제")

    print("\n[전처리] 배경 제거 + 기하학 특징 추출 + 앞머리 3단계 분류...")
    get_landmarker()

    cache_tag = "v22_male_exclude_safe"

    train_records, geo_scaler, tr_stats = build_cache(
        train_pairs,
        CACHE_DIR / "train",
        use_rembg,
        geo_scaler=geo_scaler,
        fit_scaler=fit_scaler,
        cache_tag=cache_tag,
    )

    print(
        f"  학습: ok={tr_stats['ok']} "
        f"(정상={tr_stats['normal']}, 부분가림={tr_stats['partial']}) "
        f"완전가림제외={tr_stats['bangs']} "
        f"얼굴미탐지={tr_stats['no_face']} 오류={tr_stats['error']}"
    )

    test_records, _, te_stats = build_cache(
        test_pairs,
        CACHE_DIR / "test",
        use_rembg,
        geo_scaler=geo_scaler,
        fit_scaler=False,
        cache_tag=cache_tag,
    )

    print(
        f"  테스트: ok={te_stats['ok']} "
        f"(정상={te_stats['normal']}, 부분가림={te_stats['partial']}) "
        f"완전가림제외={te_stats['bangs']} "
        f"얼굴미탐지={te_stats['no_face']} 오류={te_stats['error']}"
    )

    if not train_records:
        raise RuntimeError("전처리 후 학습 가능한 이미지가 0장입니다.")
    if not test_records:
        raise RuntimeError("전처리 후 테스트 가능한 이미지가 0장입니다.")

    joblib.dump(geo_scaler, GEO_SCALER)
    medians_raw = compute_class_geo_medians(train_records, geo_scaler)
    joblib.dump(medians_raw, GEO_MEDIANS)

    print(f"  geo_scaler 저장: {GEO_SCALER}")
    print(f"  클래스별 기하학 중앙값 저장: {GEO_MEDIANS}")

    # dataloader
    train_ds = FaceDataset(train_records, CLASS_TO_IDX, TRAIN_TRANSFORM)
    test_ds = FaceDataset(test_records, CLASS_TO_IDX, EVAL_TRANSFORM)

    sampler = None if args.no_sampler else make_weighted_sampler(train_records)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )

    # model
    model = FaceShapeNet(NUM_CLASSES).to(DEVICE)

    if not args.resume:
        load_female_transfer_weights(model)

    set_head_only(model)

    class_weights = make_class_weights(train_pairs)
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    scaler_amp = torch.amp.GradScaler("cuda") if USE_AMP else None

    best_acc = 0.0
    best_epoch = 0
    early_ctr = 0
    start_epoch = 0

    train_hist = []
    val_hist = []
    acc_hist = []

    # resume
    if args.resume and CKPT_LATEST.exists():
        print(f"\n[Resume] {CKPT_LATEST} 로드...")
        ckpt = torch.load(CKPT_LATEST, map_location=DEVICE, weights_only=False)

        model.load_state_dict(ckpt["model_state_dict"])
        set_head_only(model)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=WEIGHT_DECAY,
        )

        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError:
                print("[Resume] optimizer 구조가 달라 새 optimizer로 재시작합니다.")

        best_acc = ckpt.get("best_acc", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        early_ctr = ckpt.get("early_ctr", 0)
        start_epoch = ckpt["epoch"] + 1

        train_hist = ckpt.get("train_hist", [])
        val_hist = ckpt.get("val_hist", [])
        acc_hist = ckpt.get("acc_hist", [])

        if scaler_amp and "scaler_state_dict" in ckpt:
            scaler_amp.load_state_dict(ckpt["scaler_state_dict"])

        print(f"  epoch {start_epoch + 1}부터 이어서 시작 (Best Acc: {best_acc:.4f})")

    elif args.resume:
        print("[Resume] checkpoint_latest.pth 없음 -> 처음부터 시작")
        load_female_transfer_weights(model)

    print(
        f"\n남성 HeadOnly 전이학습 시작 "
        f"(총 {args.epochs} epoch | patience={args.patience} | "
        f"lr={args.lr} | mixup={args.mixup} | "
        f"excluded={len(excluded_all)})"
    )
    print("=" * 90)

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler_amp,
            ACCUM_STEPS,
            class_weight_tensor=class_weight_tensor,
            label_smoothing=LABEL_SMOOTHING,
            mixup_alpha=args.mixup,
        )

        scheduler.step()

        val_loss, acc = validate_male(model, test_loader)

        train_hist.append(train_loss)
        val_hist.append(val_loss)
        acc_hist.append(acc)

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | Acc {acc:.4f}",
            end="",
        )

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            early_ctr = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "best_epoch": best_epoch,
                    "classes": CLASSES,
                    "male_classes": MALE_CLASSES,
                    "class_to_idx": CLASS_TO_IDX,
                    "male_class_map": MALE_CLASS_MAP,
                    "geo_total": GEO_TOTAL,
                    "model_type": "male_headonly_5class_transfer_with_safe_exclude",
                    "exclude_list": str(args.exclude_list),
                    "excluded_count": len(excluded_all),
                },
                CKPT_BEST,
            )

            print("  ← Best", end="")
        else:
            early_ctr += 1

        print()

        latest = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "early_ctr": early_ctr,
            "train_hist": train_hist,
            "val_hist": val_hist,
            "acc_hist": acc_hist,
            "classes": CLASSES,
            "male_classes": MALE_CLASSES,
            "class_to_idx": CLASS_TO_IDX,
            "model_type": "male_headonly_5class_transfer_with_safe_exclude",
            "exclude_list": str(args.exclude_list),
            "excluded_count": len(excluded_all),
        }

        if scaler_amp:
            latest["scaler_state_dict"] = scaler_amp.state_dict()

        torch.save(latest, CKPT_LATEST)

        if early_ctr >= args.patience:
            print(f"\n[EarlyStopping] {args.patience} epoch 개선 없음 -> 종료")
            break

    print(f"\n학습 완료 — Best Accuracy: {best_acc:.4f} at epoch {best_epoch + 1}")

    # report
    if CKPT_BEST.exists():
        ckpt = torch.load(CKPT_BEST, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        y_true, y_pred = collect_predictions(model, test_loader)

        print("\n분류 리포트 (Best 모델, 남성 4클래스 기준):")
        report = classification_report(
            y_true,
            y_pred,
            labels=MALE_CLASSES,
            digits=4,
            zero_division=0,
        )
        print(report)

        heart_count = sum(1 for p in y_pred if p == "Heart")
        if heart_count > 0:
            print(f"[참고] Heart로 잘못 예측된 테스트 샘플: {heart_count}장")

        save_confusion_matrix(y_true, y_pred)

    save_training_curve(train_hist, val_hist, acc_hist, best_epoch)

    summary = {
        "best_acc": float(best_acc),
        "best_epoch": int(best_epoch + 1),
        "train_raw_count": len(train_pairs_raw),
        "test_raw_count": len(test_pairs_raw),
        "train_used_count": len(train_pairs),
        "test_used_count": len(test_pairs),
        "excluded_count": len(excluded_all),
        "exclude_list": str(args.exclude_list),
        "fresh_scaler": bool(args.fresh_scaler),
        "lr": float(args.lr),
        "mixup": float(args.mixup),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
    }

    TRAIN_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n저장 파일")
    print(f"  Best model      : {CKPT_BEST}")
    print(f"  Latest          : {CKPT_LATEST}")
    print(f"  Curve           : {CURVE_PNG}")
    print(f"  Confusion       : {CONF_PNG}")
    print(f"  Train summary   : {TRAIN_SUMMARY_JSON}")
    if excluded_all:
        print(f"  Excluded report : {EXCLUDED_REPORT_CSV}")

    print("\n추천 실행")
    print("  uv run python v22/male/train.py --no-cache")
    print("  uv run python v22/male/train.py --no-cache --mixup 0")
    print("  uv run python v22/male/train.py --no-cache --epochs 25 --patience 6")
    print("  uv run python v22/male/train.py --no-exclude")


if __name__ == "__main__":
    main()
