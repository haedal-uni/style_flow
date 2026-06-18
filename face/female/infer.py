"""
infer.py — 이미지 한 장 넣으면 얼굴형 Top-2 알려주는 추론 스크립트

사용법 (face/ 디렉토리에서 실행):
  uv run female/infer.py                                       # 기본 (image/photo.jpg)
  uv run female/infer.py --image image/photo.jpg --tta        # TTA 5-view 앙상블 (권장)
  uv run female/infer.py --image image/photo.jpg --model best # best_model 사용
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch.nn as nn


# = = = = = = = = = = = = = = = = = = = = = 
# 설정
# = = = = = = = = = = = = = = = = = = = = = 
IMG_SIZE   = 384
CLASSES    = ["Heart", "Oblong", "Oval", "Round", "Square"]
CLASSES_KO = ["하트형", "긴형", "달걀형", "둥근형", "각진형"]

MODEL_DIR  = Path(__file__).parent / "model"
BEST_MODEL = MODEL_DIR / "best_model.pth"
SWA_MODEL  = MODEL_DIR / "swa_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"


# = = = = = = = = = = = = = = = = = = = = = 
# Transform 정의
# = = = = = = = = = = = = = = = = = = = = = 
weights       = EfficientNet_V2_S_Weights.DEFAULT
imagenet_mean = weights.transforms().mean
imagenet_std  = weights.transforms().std
_normalize    = T.Normalize(mean=imagenet_mean, std=imagenet_std)

def make_transform(resize_factor: float = 1.1, hflip: bool = False) -> T.Compose:
    ops = [
        T.Resize(int(IMG_SIZE * resize_factor)),
        T.CenterCrop(IMG_SIZE),
    ]
    if hflip:
        ops.append(T.RandomHorizontalFlip(p=1.0))
    ops += [T.ToTensor(), _normalize]
    return T.Compose(ops)

# 기본 변환 (단일 예측용)
BASE_TRANSFORM = make_transform(1.1, False)

# TTA 5-view
TTA_TRANSFORMS = [
    make_transform(1.10, False),
    make_transform(1.10, True),
    make_transform(1.15, False),
    make_transform(1.15, True),
    make_transform(1.05, False),
]


# = = = = = = = = = = = = = = = = = = = = = 
# 모델 로드
# = = = = = = = = = = = = = = = = = = = = = 
def load_model(model_type: str = "swa") -> nn.Module:
    """
    model_type: "best" → best_model.pth
                "swa"  → swa_model.pth (권장)
    """
    ckpt_path = SWA_MODEL if model_type == "swa" else BEST_MODEL

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없어요: {ckpt_path}\n"
            f"{MODEL_DIR.name}/ 폴더 안에 {ckpt_path.name} 파일이 있는지 확인해주세요."
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


# = = = = = = = = = = = = = = = = = = = = = 
# 이미지 로드
# = = = = = = = = = = = = = = = = = = = = = 
def load_image(image_path: str) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없어요: {image_path}")
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        raise RuntimeError(f"이미지를 열 수 없어요: {image_path}\n오류: {e}")


# = = = = = = = = = = = = = = = = = = = = = 
# 단일 예측
# = = = = = = = = = = = = = = = = = = = = = 
@torch.inference_mode()
def predict_single(model: nn.Module, img: Image.Image) -> torch.Tensor:
    """기본 변환 1회 예측 → 클래스별 확률 반환"""
    tensor = BASE_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()
    with ctx:
        logits = model(tensor)

    return F.softmax(logits, dim=1).squeeze(0).cpu()


# = = = = = = = = = = = = = = = = = = = = = 
# TTA 예측 (5-view 앙상블)
# = = = = = = = = = = = = = = = = = = = = = 
@torch.inference_mode()
def predict_tta(model: nn.Module, img: Image.Image) -> torch.Tensor:
    """5-view TTA 앙상블 → 클래스별 확률 반환"""
    total_probs = torch.zeros(len(CLASSES))

    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()
    for transform in TTA_TRANSFORMS:
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with ctx:
            logits = model(tensor)
        total_probs += F.softmax(logits, dim=1).squeeze(0).cpu()

    return total_probs / len(TTA_TRANSFORMS)


# = = = = = = = = = = = = = = = = = = = = = 
# 결과 출력
# = = = = = = = = = = = = = = = = = = = = = 
def print_result(probs: torch.Tensor, use_tta: bool, model_type: str):
    # 확률 높은 순 정렬
    sorted_idx = probs.argsort(descending=True)

    mode_label = "SWA" if model_type == "swa" else "Best"
    tta_label  = " + TTA" if use_tta else ""

    print()
    print("=" * 45)
    print(f"  얼굴형 분석 결과  [{mode_label}{tta_label}]")
    print("=" * 45)

    for rank, idx in enumerate(sorted_idx[:2], start=1):
        cls_en = CLASSES[idx]
        cls_ko = CLASSES_KO[idx]
        prob   = probs[idx].item() * 100
        bar    = "█" * int(prob / 3)
        print(f"  {rank}위  {cls_en:<8} ({cls_ko})  {prob:5.1f}%  {bar}")
    print("=" * 45)

    # 전체 확률 상세 보기
    print("\n  [전체 확률]")
    for idx in sorted_idx:
        cls_en = CLASSES[idx]
        cls_ko = CLASSES_KO[idx]
        prob   = probs[idx].item() * 100
        print(f"    {cls_en:<8} ({cls_ko})  {prob:5.1f}%")
    print()


# = = = = = = = = = = = = = = = = = = = = = 
# 메인
# = = = = = = = = = = = = = = = = = = = = = 
def parse_args():
    parser = argparse.ArgumentParser(
        description="얼굴형 분류 추론 스크립트"
    )
    parser.add_argument(
        "--image", "-i", type=str, default="image/photo.jpg",
        help="추론할 이미지 경로 (기본: image/photo.jpg)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="swa",
        choices=["swa", "best"],
        help="사용할 모델: swa(기본, 더 정확) 또는 best"
    )
    parser.add_argument(
        "--tta", action="store_true",
        help="TTA(5-view 앙상블) 적용 여부"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n 이미지: {args.image}")
    print(f"모델  : {args.model}_model.pth")
    print(f"TTA   : {'ON (5-view)' if args.tta else 'OFF'}")

    # 모델 + 이미지 로드
    model = load_model(args.model)
    img   = load_image(args.image)

    # 예측
    if args.tta:
        probs = predict_tta(model, img)
    else:
        probs = predict_single(model, img)
    # 결과 출력
    print_result(probs, args.tta, args.model)


if __name__ == "__main__":
    main()
