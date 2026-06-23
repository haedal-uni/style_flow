"""
v22 얼굴형 추론 — 이미지 한 장을 넣으면 얼굴형 + 설명 가능한 근거를 출력

성별 모델 자동/수동 선택:
  --gender female (기본값) → v22/female/swa_model.pth 사용, 5클래스(Heart 포함)
  --gender male            → v22/male/swa_model.pth 사용, 4클래스(전이학습)

동작 방식:
  1. 배경 제거 (rembg, 머리카락 보존)
  2. MediaPipe로 17개 기하학 특징 + 3단계 앞머리 상태 측정
  3. EfficientNetV2-S(CNN) + GeoEncoder + HeartGate 융합 모델로 TTA 5-view 추론
  4. 삼정 비율·하악각·턱끝각도·Heart 게이트 값을 사람이 읽는 설명으로 변환해 출력

실행 (uv 사용):
  uv run python v22/infer.py path/to/photo.jpg
  uv run python v22/infer.py path/to/photo.jpg --gender male
  uv run python v22/infer.py path/to/photo.jpg --no-rembg
  uv run python v22/infer.py path/to/photo.jpg --json   # JSON 형식으로만 출력 (파이프라인 연동용)
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import joblib
import torch

from common import (
    DEVICE, FEATURE_NAMES,
    remove_background, get_landmarker,
    compute_upper_r, classify_bangs,
    _crop_face, _compute_geo,
    FaceShapeNet, GEO_TOTAL,
    predict_tta, explain_prediction,
)

_HERE = Path(__file__).parent

FEMALE_DIR = _HERE / "female"
MALE_DIR   = _HERE / "male"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델/스케일러 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _resolve_ckpt(model_dir: Path) -> Path:
    """SWA 모델을 우선 사용하고, 없으면 best_model로 폴백."""
    swa = model_dir / "swa_model.pth"
    best = model_dir / "best_model.pth"
    if swa.exists():
        return swa
    if best.exists():
        print(f"[안내] swa_model.pth가 없어 best_model.pth를 사용합니다: {best}")
        return best
    raise FileNotFoundError(
        f"학습된 모델이 없습니다: {model_dir}\n"
        f"먼저 학습을 실행하세요. "
        f"(예: uv run python v22/{model_dir.name}/train.py)"
    )


def load_model_bundle(gender: str):
    """
    gender ('female' | 'male')에 맞는 모델·geo_scaler·클래스 목록·기하학 중앙값을 로드한다.
    """
    model_dir = FEMALE_DIR if gender == "female" else MALE_DIR
    ckpt_path = _resolve_ckpt(model_dir)

    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    classes = ck["classes"]
    num_classes = len(classes)

    model = FaceShapeNet(num_classes).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # geo_scaler: 남성 모델도 여성 scaler를 그대로 재사용하는 구조이므로
    # female 폴더의 scaler를 우선 찾고, male 폴더에 별도로 있으면 그것을 사용
    scaler_candidates = [model_dir / "geo_scaler.pkl", FEMALE_DIR / "geo_scaler.pkl"]
    geo_scaler = None
    for sc in scaler_candidates:
        if sc.exists():
            geo_scaler = joblib.load(sc)
            break
    if geo_scaler is None:
        raise FileNotFoundError(
            "geo_scaler.pkl을 찾을 수 없습니다. female 모델을 먼저 학습해주세요."
        )

    medians_path = model_dir / "class_geo_medians.pkl"
    class_geo_medians = joblib.load(medians_path) if medians_path.exists() else None

    # 남성 모델은 4클래스만 유효 — Heart는 학습 데이터가 없어 표시에서 제외
    valid_classes = ck.get("male_classes", classes) if gender == "male" else classes

    return {
        "model": model,
        "classes": classes,
        "valid_classes": valid_classes,
        "geo_scaler": geo_scaler,
        "class_geo_medians": class_geo_medians,
        "ckpt_path": ckpt_path,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단일 이미지 전처리 (학습 파이프라인과 100% 동일한 경로)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess_for_inference(image_path: Path, use_rembg: bool = True) -> dict:
    """
    추론용 전처리. common.preprocess_one과 동일한 단계를 거치되,
    원본 스케일 geo(설명용)와 정규화된 geo18(모델 입력용)을 모두 반환한다.

    Returns
    -------
    dict with keys:
      status        : 'ok' | 'no_face' | 'bangs' | 'error'
      bangs_status  : 'normal' | 'partial' | None
      img_bgr       : 배경 제거 + 크롭된 BGR 이미지 (status='ok'일 때만)
      geo_raw       : 원본 스케일 17차원 특징 (측정 실패 시 None)
      message       : 사람이 읽는 안내 메시지
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {
            "status": "error", "bangs_status": None, "img_bgr": None, "geo_raw": None,
            "message": f"이미지를 읽을 수 없습니다: {image_path}",
        }

    img_bgr = remove_background(img_bgr, use_rembg)
    h, w = img_bgr.shape[:2]

    import mediapipe as mp
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    res = get_landmarker().detect(mp_img)
    if not res.face_landmarks:
        return {
            "status": "no_face", "bangs_status": None, "img_bgr": img_bgr, "geo_raw": None,
            "message": "얼굴을 인식하지 못했습니다. 정면을 향한 선명한 사진을 사용해주세요.",
        }

    lms = res.face_landmarks[0]
    upper_r_raw = compute_upper_r(img_bgr, lms, w, h)
    bangs_status = classify_bangs(upper_r_raw)

    if bangs_status == "full":
        return {
            "status": "bangs", "bangs_status": "full", "img_bgr": img_bgr, "geo_raw": None,
            "message": (
                "앞머리가 이마를 완전히 가리고 있어 정확한 분석이 어렵습니다. "
                "이마가 보이는 사진을 사용해주세요."
            ),
        }

    # 2-pass 크롭 재탐지
    crop = _crop_face(img_bgr, lms, w, h)
    if crop.size > 0:
        ch, cw = crop.shape[:2]
        mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res2 = get_landmarker().detect(mp_crop)
        if res2.face_landmarks:
            lms, img_bgr, w, h = res2.face_landmarks[0], crop, cw, ch

    geo_raw = _compute_geo(img_bgr, lms, w, h)

    message = "정상적으로 분석되었습니다."
    if bangs_status == "partial":
        message = (
            "이마가 머리카락에 일부 가려져 있어 측정값의 신뢰도가 다소 낮을 수 있습니다. "
            "참고용으로 확인해주세요."
        )

    return {
        "status": "ok", "bangs_status": bangs_status,
        "img_bgr": img_bgr, "geo_raw": geo_raw, "message": message,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 추론 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def infer_image(image_path: str, gender: str = "female", use_rembg: bool = True) -> dict:
    """
    이미지 한 장에 대해 얼굴형 예측 + 설명 가능한 근거를 반환한다.

    Returns
    -------
    dict — status, predictions(Top-2), explanation, raw_features 등을 포함
    """
    p = Path(image_path)
    if not p.exists():
        return {"status": "error", "message": f"파일이 존재하지 않습니다: {p}"}

    bundle = load_model_bundle(gender)
    model        = bundle["model"]
    classes      = bundle["classes"]
    valid_classes = bundle["valid_classes"]
    geo_scaler   = bundle["geo_scaler"]
    class_geo_medians = bundle["class_geo_medians"]

    pre = preprocess_for_inference(p, use_rembg=use_rembg)
    if pre["status"] in ("error", "bangs"):
        return {"status": pre["status"], "message": pre["message"]}

    # geo18 구성 (모델 입력용 — 정규화 + 유효 플래그)
    if pre["geo_raw"] is not None:
        scaled = geo_scaler.transform([pre["geo_raw"]])[0]
        geo18 = np.append(scaled, 1.0).astype(np.float32)
    else:
        geo18 = np.zeros(GEO_TOTAL, dtype=np.float32)

    # TTA 5-view 추론
    probs, gate_value = predict_tta(model, pre["img_bgr"], geo18)
    probs_np = probs.numpy()

    # 남성 모델은 Heart를 학습하지 않았으므로 결과에서 제외하고 재정규화
    if gender == "male":
        mask = np.array([c in valid_classes for c in classes])
        probs_np = probs_np * mask
        if probs_np.sum() > 0:
            probs_np = probs_np / probs_np.sum()

    ranked = sorted(zip(classes, probs_np.tolist()), key=lambda x: x[1], reverse=True)
    ranked = [r for r in ranked if r[0] in valid_classes][:2]

    pred_class = ranked[0][0]
    explanation = explain_prediction(
        pre["geo_raw"], pred_class, gate_value, class_geo_medians,
    )

    return {
        "status": "ok",
        "gender_model": gender,
        "bangs_status": pre["bangs_status"],
        "preprocessing_message": pre["message"],
        "predictions": [{"class": c, "probability": prob} for c, prob in ranked],
        "heart_gate_value": gate_value,
        "explanation": explanation,
        "model_checkpoint": str(bundle["ckpt_path"]),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 출력 포맷
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_human_readable(result: dict, image_path: str):
    print(f"\n[얼굴형 분석] {Path(image_path).name}")
    print("=" * 50)

    if result["status"] != "ok":
        print(f"  분석 실패: {result['message']}")
        return

    if result["bangs_status"] == "partial":
        print(f"  ※ {result['preprocessing_message']}")

    print(f"\n  사용 모델: {result['gender_model']} "
          f"({'5클래스, Heart 포함' if result['gender_model']=='female' else '4클래스, 전이학습'})")

    print("\n  [Top-2 예측]")
    for i, p in enumerate(result["predictions"], 1):
        bar = "█" * int(p["probability"] * 30)
        print(f"   {i}위  {p['class']:<8} {p['probability']*100:5.1f}%  {bar}")

    exp = result["explanation"]
    if not exp.get("available"):
        print(f"\n  {exp['message']}")
        return

    print("\n  [삼정 비율]")
    for k, v in exp["삼정비율"].items():
        print(f"   - {k}: {v}")

    print("\n  [핵심 수치]")
    for k, v in exp["핵심수치"].items():
        print(f"   - {k}: {v}")

    print(f"\n  [Heart 게이트 해설]\n   {exp['heart_gate']}")

    if "reference_comparison" in exp:
        print("\n  [동일 클래스 평균과 비교]")
        for k, v in exp["reference_comparison"].items():
            print(f"   - {k}: {v}")

    print(f"\n  [종합 설명]\n   {exp['summary']}")
    print("=" * 50)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    p = argparse.ArgumentParser(
        description="v22 얼굴형 추론 — 이미지 한 장으로 얼굴형 + 설명 가능한 근거 출력"
    )
    p.add_argument("image", type=str, help="분석할 이미지 경로")
    p.add_argument("--gender", choices=["female", "male"], default="female",
                  help="사용할 모델 (기본값: female)")
    p.add_argument("--no-rembg", action="store_true", help="배경 제거 비활성화")
    p.add_argument("--json", action="store_true", help="JSON 형식으로만 출력")
    return p.parse_args()


def main():
    args = parse_args()
    result = infer_image(args.image, gender=args.gender, use_rembg=not args.no_rembg)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human_readable(result, args.image)


if __name__ == "__main__":
    main()