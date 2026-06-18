"""
남성 얼굴형 추론 스크립트 — EfficientNetV2-S · 여성 SWA 전이학습 모델

사용법 (face/ 디렉토리에서 실행):
  uv run male/infer.py                                       # 기본 (image/photo.jpg)
  uv run male/infer.py --image image/photo.jpg --tta        # TTA 5-view 앙상블 (권장)
  uv run male/infer.py --image image/photo.jpg --model best # best_model 사용
"""

import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch.nn as nn


_HERE = Path(os.path.dirname(os.path.abspath(__file__)))

IMG_SIZE   = 384
CLASSES    = ["Heart", "Oblong", "Oval", "Round", "Square"]
CLASSES_KO = ["역삼각형", "장방형(긴형)", "달걀형(Oval)", "둥근형", "각진형"]

# 남성 결과에서 Heart는 학습 데이터 없음 (참고용)
MALE_NOTE = {
    "Heart":  "(남성 학습 데이터 없음 — 참고치)",
    "Oblong": "(rectangular 포함)",
    "Oval":   "(ovale 포함)",
    "Round":  "",
    "Square": "",
}

MODEL_DIR  = _HERE / "model"
BEST_MODEL = MODEL_DIR / "best_model.pth"
SWA_MODEL  = MODEL_DIR / "swa_model.pth"

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

# =========================
# Transforms 
# =========================

weights       = EfficientNet_V2_S_Weights.DEFAULT
imagenet_mean = weights.transforms().mean
imagenet_std  = weights.transforms().std
_normalize    = T.Normalize(mean=imagenet_mean, std=imagenet_std)


def make_transform(resize_factor: float = 1.1, hflip: bool = False) -> T.Compose:
    ops = [T.Resize(int(IMG_SIZE * resize_factor)), T.CenterCrop(IMG_SIZE)]
    if hflip:
        ops.append(T.RandomHorizontalFlip(p=1.0))
    ops += [T.ToTensor(), _normalize]
    return T.Compose(ops)


BASE_TRANSFORM = make_transform(1.1, False)
TTA_TRANSFORMS = [
    make_transform(1.10, False),
    make_transform(1.10, True),
    make_transform(1.15, False),
    make_transform(1.15, True),
    make_transform(1.05, False),
]


# =========================
# 모델 로드
# =========================

def load_model(model_type: str = "swa") -> nn.Module:
    ckpt_path = SWA_MODEL if model_type == "swa" else BEST_MODEL
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"모델 파일 없음: {ckpt_path}\n"
            "male/train.py 를 먼저 실행하여 남성 모델을 학습해주세요."
        )
    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, len(CLASSES)),
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"모델 로드 완료: {ckpt_path.name}")
    return model


# =========================
# 예측
# =========================

@torch.inference_mode()
def predict_single(model: nn.Module, img: Image.Image) -> torch.Tensor:
    tensor = BASE_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()
    with ctx:
        logits = model(tensor)
    return F.softmax(logits, dim=1).squeeze(0).cpu()


@torch.inference_mode()
def predict_tta(model: nn.Module, img: Image.Image) -> torch.Tensor:
    total = torch.zeros(len(CLASSES))
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()
    for tf in TTA_TRANSFORMS:
        tensor = tf(img).unsqueeze(0).to(DEVICE)
        with ctx:
            logits = model(tensor)
        total += F.softmax(logits, dim=1).squeeze(0).cpu()
    return total / len(TTA_TRANSFORMS)


# =========================
# 결과 출력
# =========================

def print_result(probs: torch.Tensor, use_tta: bool, model_type: str):
    sorted_idx = probs.argsort(descending=True)
    mode_label = "SWA" if model_type == "swa" else "Best"
    tta_label  = " + TTA" if use_tta else ""

    print()
    print("=" * 50)
    print(f"  남성 얼굴형 분석 결과  [{mode_label}{tta_label}]")
    print("=" * 50)

    for rank, idx in enumerate(sorted_idx[:2], 1):
        cls_en = CLASSES[idx]
        cls_ko = CLASSES_KO[idx]
        note   = MALE_NOTE[cls_en]
        prob   = probs[idx].item() * 100
        bar    = "█" * int(prob / 3)
        print(f"  {rank}위  {cls_en:<8} ({cls_ko})  {prob:5.1f}%  {bar}")
        if note:
            print(f"       {note}")

    print("=" * 50)
    print("\n  [전체 확률]")
    for idx in sorted_idx:
        cls_en = CLASSES[idx]
        cls_ko = CLASSES_KO[idx]
        prob   = probs[idx].item() * 100
        bar    = "█" * int(prob / 3)
        print(f"    {cls_en:<8} ({cls_ko:<20})  {prob:5.1f}%")
    print()


# =========================
# 메인
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="남성 얼굴형 분류 추론")
    parser.add_argument("--image", "-i", type=str, default="image/photo.jpg", help="추론할 이미지 경로 (기본: image/photo.jpg)")
    parser.add_argument("--model", "-m", type=str, default="swa",
                        choices=["swa", "best"], help="swa(기본) 또는 best")
    parser.add_argument("--tta", action="store_true", help="TTA 5-view 앙상블")
    return parser.parse_args()


def main():
    args  = parse_args()
    p     = Path(args.image)

    if not p.exists():
        print(f"[오류] 이미지 파일 없음: {p}")
        return

    print(f"\n이미지: {p}")
    print(f"모델  : {args.model}_model.pth")
    print(f"TTA   : {'ON (5-view)' if args.tta else 'OFF'}")

    model = load_model(args.model)
    img   = Image.open(p).convert("RGB")

    probs = predict_tta(model, img) if args.tta else predict_single(model, img)
    print_result(probs, args.tta, args.model)


if __name__ == "__main__":
    main()
