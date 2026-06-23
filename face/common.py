"""
v22 공통 모듈 — MediaPipe 기하학 + EfficientNetV2-S CNN 하이브리드 융합 모델

v21 대비 핵심 변경:
  1. Heart 게이트(HeartGate) 추가
       CNN은 이마가 머리카락에 가려진 Heart형을 윤곽만으로 잘 구분하지 못합니다.
       반면 기하학 특징(R2=이마/턱 비율, chin_angle=턱끝 각도, face_taper_r)은
       머리카락에 가려도 랜드마크 좌표만으로 안정적으로 측정됩니다.
       그래서 두 신호를 단순히 이어붙이는 late fusion 대신,
       geo 특징이 Heart를 강하게 가리킬 때 CNN 출력보다 geo 출력에
       더 큰 가중치를 주는 작은 게이트 네트워크를 추가했습니다.
       (Mixture-of-Experts의 단순화 버전 — 입력에 따라 전문가 비중을 조절)

  2. 3단계 앞머리 분류 (v21의 2단계 → 3단계)
       - 완전 가림 (upper_r < 0.08): 제외
       - 부분 가림 (0.08 ~ 0.15): 포함하되 표본 가중치를 낮춤 (신뢰도 페널티)
       - 정상 (upper_r >= 0.15): 정상 가중치로 포함
       단일 임계값으로 버려지던 데이터를 더 많이 활용하면서도,
       측정이 불안정한 표본이 학습을 과도하게 흔들지 않도록 균형을 잡습니다.

  3. 설명 가능성 함수 explain_prediction() 추가
       추론 시 "왜 이 얼굴형으로 분류됐는지"를 삼정 비율·각도·게이트 가중치
       기준으로 사람이 읽을 수 있게 정리해서 반환합니다.

  4. 배경 제거는 v21과 동일 (rembg 우선, GrabCut 폴백) — 머리카락 보존, 배경만 흰색 치환
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── rembg 선택적 임포트 (지연 세션 초기화) ─────────────────────
# new_session()은 첫 호출 시 모델 가중치(u2net.onnx, ~170MB)를 다운로드합니다.
# import 시점에 즉시 호출하면 네트워크가 느리거나 막혀 있을 때
# common.py를 import하는 것만으로 전체 스크립트가 멈추거나 죽습니다.
# 그래서 세션 생성을 실제로 배경 제거가 필요한 첫 호출 시점까지 지연시킵니다.
try:
    from rembg import remove as _rembg_raw, new_session as _rembg_sess
    _REMBG_IMPORTABLE = True
except ImportError:
    _REMBG_IMPORTABLE = False

_REMBG_SESSION = None
_REMBG_AVAILABLE = _REMBG_IMPORTABLE  # 실제 가용 여부는 첫 사용 시 재확인됨


def _get_rembg_session():
    """rembg 세션을 지연 생성한다. 모델 다운로드 실패 시 None을 반환."""
    global _REMBG_SESSION, _REMBG_AVAILABLE
    if not _REMBG_IMPORTABLE:
        return None
    if _REMBG_SESSION is None:
        try:
            _REMBG_SESSION = _rembg_sess(providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"[rembg] 세션 생성 실패 — GrabCut 폴백으로 전환합니다: {e}")
            _REMBG_AVAILABLE = False
            return None
    return _REMBG_SESSION


def _rembg_fn(img):
    session = _get_rembg_session()
    if session is None:
        raise RuntimeError("rembg 세션을 사용할 수 없습니다.")
    return _rembg_raw(img, session=session)


# ── libGLES 프리로드 (Linux 헤드리스 서버용 — MediaPipe 의존성) ──
def _preload_libgles():
    import ctypes, glob, subprocess
    try:
        ctypes.CDLL("libGLESv2.so.2")
        return
    except OSError:
        pass
    candidates = []
    try:
        out = subprocess.check_output(["ldconfig", "-p"], stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            if "libGLESv2" in line and "=>" in line:
                candidates.append(line.split("=>")[-1].strip())
    except Exception:
        pass
    for pat in [
        "/usr/lib/x86_64-linux-gnu/libGLESv2.so*",
        "/usr/lib/x86_64-linux-gnu/mesa-egl/libGLESv2.so*",
        "/usr/lib/aarch64-linux-gnu/libGLESv2.so*",
    ]:
        candidates += glob.glob(pat)
    for p in candidates:
        try:
            ctypes.CDLL(p)
            return
        except OSError:
            continue


if sys.platform != "win32":
    _preload_libgles()

import mediapipe as mp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMG_SIZE = 384
GEO_DIM  = 17
GEO_TOTAL = GEO_DIM + 1   # 17 특징 + 유효 플래그(1)

# 3단계 앞머리 임계값
BANGS_FULL_THRESHOLD    = 0.08   # 이 미만 → 완전 가림 → 학습 제외
BANGS_PARTIAL_THRESHOLD = 0.15   # 이 미만(완전가림 이상) → 부분 가림 → 포함하되 가중치 낮춤
PARTIAL_SAMPLE_WEIGHT   = 0.5    # 부분 가림 표본의 학습 가중치 (정상=1.0 대비 절반)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

FEATURE_NAMES = [
    "R1", "R2", "R3", "R4",
    "upper_r", "mid_r", "lower_r", "golden_flag",
    "jaw_angle", "lower_area_r",
    "eye_w_r", "nose_w_r",
    "chin_angle", "jaw_taper_r", "mouth_w_r", "brow_w_r", "face_taper_r",
]

# 설명 가능성 리포트에서 사람이 읽는 한글 설명
FEATURE_DESC = {
    "R1": "얼굴 길이 / 광대 너비 (세로 길이감)",
    "R2": "이마 너비 / 턱 너비 (상하 균형, Heart는 높음)",
    "R3": "광대 너비 / 평균(이마,턱) (광대 돌출도)",
    "R4": "광대 너비 / 턱 너비",
    "upper_r": "상안부 비율 (이마)",
    "mid_r": "중안부 비율 (눈·코)",
    "lower_r": "하안부 비율 (입·턱)",
    "golden_flag": "삼정 균등 여부",
    "jaw_angle": "하악각 (Square≈90°, Round≈140°)",
    "lower_area_r": "하악 삼각형 면적 비율",
    "eye_w_r": "눈 간격 / 광대 너비",
    "nose_w_r": "코 너비 / 광대 너비",
    "chin_angle": "턱끝 각도 (Heart는 작음=뾰족)",
    "jaw_taper_r": "턱 너비 / 광대 너비 (Heart는 작음)",
    "mouth_w_r": "입 너비 / 광대 너비",
    "brow_w_r": "눈썹 너비 / 광대 너비",
    "face_taper_r": "턱 너비 / 이마 너비 (Heart는 낮음)",
}

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 배경 제거 — 머리카락 보존, 배경만 흰색 치환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def remove_background(img_bgr: np.ndarray, use_rembg: bool = True) -> np.ndarray:
    """
    배경만 흰색으로 치환한다. 머리카락은 보존.
    1순위: rembg (AI 세그멘테이션, 머리카락 경계 보존 우수)
    2순위: GrabCut (rembg 미설치 시 폴백)
    """
    if use_rembg and _REMBG_AVAILABLE:
        try:
            img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(img_rgb)
            result    = _rembg_fn(pil_img)
            result_np = np.array(result)
            alpha     = result_np[:, :, 3]
            fg        = (alpha > 10).astype(np.uint8)
            white     = np.ones_like(img_bgr) * 255
            fg3       = np.stack([fg] * 3, axis=-1)
            return np.where(
                fg3,
                cv2.cvtColor(result_np[:, :, :3], cv2.COLOR_RGB2BGR),
                white,
            ).astype(np.uint8)
        except Exception:
            pass
    # GrabCut 폴백
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.10), int(h * 0.05), int(w * 0.80), int(h * 0.90))
    bgd  = np.zeros((1, 65), np.float64)
    fgd  = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        white = np.ones_like(img_bgr) * 255
        return np.where(np.stack([fg] * 3, axis=-1), img_bgr, white).astype(np.uint8)
    except Exception:
        return img_bgr


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MediaPipe (지연 초기화)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_LANDMARKER = None


def _ensure_mp_model() -> Path:
    candidates = [
        Path(__file__).parent.parent / "face_landmarker.task",
        Path(__file__).parent / "face_landmarker.task",
    ]
    for p in candidates:
        if p.exists():
            return p
    target = Path(__file__).parent.parent / "face_landmarker.task"
    print("[MediaPipe] face_landmarker.task 다운로드 중...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        target,
    )
    return target


def get_landmarker():
    global _LANDMARKER
    if _LANDMARKER is None:
        model_path = _ensure_mp_model()
        opt = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _LANDMARKER = mp.tasks.vision.FaceLandmarker.create_from_options(opt)
    return _LANDMARKER


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 기하학 유틸리티
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _angle_at(a, b, c) -> float:
    ba  = np.array(a) - np.array(b)
    bc  = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _polygon_area(pts) -> float:
    n, area = len(pts), 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _detect_hairline_y(img_bgr, lms, w, h, brow_y_norm) -> float:
    ycrcb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cx       = w // 2
    brow_row = max(0, int(brow_y_norm * h) - 5)
    ref_row  = min(brow_row + 10, h - 1)
    y_ref    = float(ycrcb[ref_row, cx, 0])
    cr_ref   = float(ycrcb[ref_row, cx, 1])
    hairline_y = float(lms[10].y)
    for row in range(brow_row, -1, -1):
        py = ycrcb[row, cx, 0]
        if abs(int(py) - y_ref) > 40 or abs(int(ycrcb[row, cx, 1]) - cr_ref) > 20:
            if py < 230:  # 흰색 배경(255) 픽셀 제외
                hairline_y = row / h
                break
    return hairline_y


def compute_upper_r(img_bgr, lms, w, h) -> float:
    """이마 비율(upper_r) 단독 계산 — 3단계 앞머리 분류에 사용."""
    brow_ys    = [lms[i].y for i in list(range(33, 42)) + list(range(46, 53))]
    brow_y     = float(np.mean(brow_ys))
    hairline_y = _detect_hairline_y(img_bgr, lms, w, h, brow_y)
    chin_y     = lms[152].y
    face_len   = abs(chin_y - hairline_y)
    if face_len < 1e-6:
        return 1.0  # 측정 불가 시 정상으로 간주 (보수적)
    return abs(brow_y - hairline_y) / face_len


def classify_bangs(upper_r: float) -> str:
    """
    3단계 앞머리 분류.
    'full'    : 완전 가림 → 학습/추론 제외
    'partial' : 부분 가림 → 포함하되 표본 가중치 낮춤
    'normal'  : 정상 → 정상 가중치
    """
    if upper_r < BANGS_FULL_THRESHOLD:
        return "full"
    if upper_r < BANGS_PARTIAL_THRESHOLD:
        return "partial"
    return "normal"


def _crop_face(img_bgr, lms, w, h, pad=0.20):
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    fw, fh  = x2 - x1, y2 - y1
    return img_bgr[
        max(0, y1 - int(fh * pad * 2)):min(h, y2 + int(fh * pad)),
        max(0, x1 - int(fw * pad))    :min(w, x2 + int(fw * pad)),
    ]


def _compute_geo(img_bgr, lms, w, h) -> "np.ndarray | None":
    """랜드마크에서 17개 기하학 특징을 계산한다."""
    def lm(i):  return (lms[i].x * w, lms[i].y * h)
    def lmy(i): return lms[i].y

    brow_ys     = [lms[i].y for i in list(range(33, 42)) + list(range(46, 53))]
    brow_y_norm = float(np.mean(brow_ys))
    hairline_y  = _detect_hairline_y(img_bgr, lms, w, h, brow_y_norm)
    chin_y      = lmy(152)
    face_len    = abs(chin_y - hairline_y)
    if face_len < 1e-6:
        return None

    cheek_w    = abs(lms[234].x - lms[454].x) * w
    forehead_w = abs(lms[127].x - lms[356].x) * w
    jaw_w      = abs(lms[172].x - lms[397].x) * w
    if min(cheek_w, jaw_w, forehead_w) < 1e-6:
        return None

    face_px = face_len * h
    R1 = face_px / cheek_w
    R2 = forehead_w / jaw_w          # Heart: 높음 (넓은 이마, 좁은 턱)
    R3 = cheek_w / ((forehead_w + jaw_w) / 2.0)
    R4 = cheek_w / jaw_w

    nose_y   = lmy(2)
    upper_px = abs(brow_y_norm - hairline_y) * h
    mid_px   = abs(nose_y - brow_y_norm) * h
    lower_px = abs(chin_y - nose_y) * h
    total_px = upper_px + mid_px + lower_px + 1e-9
    upper_r  = upper_px / total_px
    mid_r    = mid_px   / total_px
    lower_r  = lower_px / total_px
    golden   = float(all(abs(v - 1 / 3) < 0.05 for v in [upper_r, mid_r, lower_r]))

    jaw_angle = (_angle_at(lm(58), lm(172), lm(152)) +
                 _angle_at(lm(288), lm(397), lm(152))) / 2
    chin_angle   = _angle_at(lm(172), lm(152), lm(397))  # Heart: 작음(뾰족한 턱)
    lower_area   = _polygon_area([lm(172), lm(152), lm(397)])
    lower_area_r = lower_area / (cheek_w * face_px + 1e-9)

    eye_w   = abs(lms[33].x  - lms[263].x) * w
    nose_w  = abs(lms[129].x - lms[358].x) * w
    mouth_w = abs(lms[61].x  - lms[291].x) * w
    brow_w  = abs(lms[70].x  - lms[300].x) * w

    return np.array([
        R1, R2, R3, R4,
        upper_r, mid_r, lower_r, golden,
        jaw_angle, lower_area_r,
        eye_w / (cheek_w + 1e-9), nose_w / (cheek_w + 1e-9),
        chin_angle,
        jaw_w / (cheek_w + 1e-9),            # jaw_taper_r
        mouth_w / (cheek_w + 1e-9),
        brow_w / (cheek_w + 1e-9),
        jaw_w / (forehead_w + 1e-9),          # face_taper_r: Heart는 낮음
    ], dtype=np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 이미지 1장 전처리 (배경 제거 + 기하학 추출 + 3단계 앞머리 판정)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess_one(path: Path, use_rembg: bool = True):
    """
    Returns
    -------
    (bg_removed_bgr, geo_17d, bangs_status, sample_weight, status)

    status:
      'ok'      — 정상 처리 완료 (bangs_status는 'normal' 또는 'partial')
      'no_face' — 얼굴 미탐지, CNN은 사용 가능하나 geo는 None
      'bangs'   — 완전 가림, 학습/추론 제외
      'error'   — 이미지 로드 실패
    """
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return None, None, None, 0.0, "error"

    img_bgr = remove_background(img_bgr, use_rembg)
    h, w    = img_bgr.shape[:2]

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    res = get_landmarker().detect(mp_img)
    if not res.face_landmarks:
        return img_bgr, None, None, 1.0, "no_face"

    lms = res.face_landmarks[0]
    upper_r_raw  = compute_upper_r(img_bgr, lms, w, h)
    bangs_status = classify_bangs(upper_r_raw)

    if bangs_status == "full":
        return None, None, bangs_status, 0.0, "bangs"

    sample_weight = PARTIAL_SAMPLE_WEIGHT if bangs_status == "partial" else 1.0

    # 2-pass 크롭 재탐지 (정밀도 향상)
    crop = _crop_face(img_bgr, lms, w, h)
    if crop.size > 0:
        ch, cw = crop.shape[:2]
        mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res2 = get_landmarker().detect(mp_crop)
        if res2.face_landmarks:
            lms, img_bgr, w, h = res2.face_landmarks[0], crop, cw, ch

    geo = _compute_geo(img_bgr, lms, w, h)
    return img_bgr, geo, bangs_status, sample_weight, "ok"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 전처리 캐시 빌드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_cache(
    pairs: list,
    cache_dir: Path,
    use_rembg: bool = True,
    geo_scaler: "StandardScaler | None" = None,
    fit_scaler: bool = False,
    cache_tag: str = "v22",
) -> tuple:
    """
    배경 제거 이미지를 cache_dir에 저장하고 geo 특징 + 표본 가중치를 수집한다.

    Returns
    -------
    records    : [(cached_img_path, class_name, geo_18d, sample_weight)]
    geo_scaler : 학습 시 fit된 StandardScaler (테스트는 학습 scaler를 그대로 사용)
    stats      : dict(total, ok, normal, partial, bangs, no_face, error)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    records  = []
    all_geos = []
    stats = dict(total=len(pairs), ok=0, normal=0, partial=0,
                 bangs=0, no_face=0, error=0)

    for path, cls in tqdm(pairs, desc=f"  전처리 ({cache_dir.name})"):
        cached = cache_dir / f"{path.stem}_{cache_tag}.png"

        if cached.exists():
            # 캐시 hit: 저장된 배경제거 이미지를 재사용, geo/가중치만 재계산
            img_bgr = cv2.imread(str(cached))
            if img_bgr is None:
                cached.unlink(missing_ok=True)
                img_bgr, geo, bangs_status, weight, status = preprocess_one(path, use_rembg)
            else:
                h, w   = img_bgr.shape[:2]
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                res = get_landmarker().detect(mp_img)
                if not res.face_landmarks:
                    geo, bangs_status, weight, status = None, None, 1.0, "no_face"
                else:
                    lms = res.face_landmarks[0]
                    upper_r_raw  = compute_upper_r(img_bgr, lms, w, h)
                    bangs_status = classify_bangs(upper_r_raw)
                    if bangs_status == "full":
                        cached.unlink(missing_ok=True)
                        geo, weight, status = None, 0.0, "bangs"
                    else:
                        weight = PARTIAL_SAMPLE_WEIGHT if bangs_status == "partial" else 1.0
                        crop = _crop_face(img_bgr, lms, w, h)
                        if crop.size > 0:
                            mp2 = mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            r2 = get_landmarker().detect(mp2)
                            if r2.face_landmarks:
                                lms, img_bgr, w, h = (
                                    r2.face_landmarks[0], crop,
                                    crop.shape[1], crop.shape[0],
                                )
                        geo    = _compute_geo(img_bgr, lms, w, h)
                        status = "ok"
        else:
            img_bgr, geo, bangs_status, weight, status = preprocess_one(path, use_rembg)

        stats[status] = stats.get(status, 0) + 1
        if status == "ok" and bangs_status in ("normal", "partial"):
            stats[bangs_status] += 1

        if status in ("bangs", "error"):
            continue

        # 캐시 저장 (최초 1회)
        if not cached.exists() and img_bgr is not None:
            cv2.imwrite(str(cached), img_bgr)

        if geo is not None:
            all_geos.append(geo)

        records.append((cached, cls, geo, weight))

    # Scaler fit/transform
    if fit_scaler and all_geos:
        geo_scaler = StandardScaler()
        geo_scaler.fit(np.array(all_geos))

    # geo_18d 생성 (스케일 + 유효 플래그)
    final_records = []
    for cached_path, cls, geo, weight in records:
        if geo is not None and geo_scaler is not None:
            scaled = geo_scaler.transform([geo])[0]
            geo_18 = np.append(scaled, 1.0).astype(np.float32)
        else:
            geo_18 = np.zeros(GEO_TOTAL, dtype=np.float32)
        final_records.append((cached_path, cls, geo_18, weight))

    return final_records, geo_scaler, stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 아키텍처
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_weights_def   = EfficientNet_V2_S_Weights.DEFAULT
_imagenet_mean = _weights_def.transforms().mean
_imagenet_std  = _weights_def.transforms().std
_normalize     = T.Normalize(mean=_imagenet_mean, std=_imagenet_std)


class GeoEncoder(nn.Module):
    """17개 기하학 특징 + 유효 플래그(18차원) → 64차원 임베딩."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GEO_TOTAL, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 64),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class HeartGate(nn.Module):
    """
    [v22 신규] Heart 인식 보강 게이트.

    문제 의식:
      EfficientNetV2-S는 '윤곽'으로 얼굴형을 보는데, 머리카락이 이마 일부를
      가리면 Heart(이마 넓고 턱 좁음)와 Oval을 혼동하기 쉽습니다.
      반면 R2(이마/턱 비율), chin_angle(턱끝 각도), face_taper_r은
      랜드마크 좌표만으로 계산되므로 머리카락 차폐에 비교적 강건합니다.

    동작 방식:
      geo 임베딩에서 "이 표본이 기하학적으로 Heart에 가까운 정도"를
      스칼라 게이트 값(0~1)으로 추정합니다.
      최종 분류기에는 (CNN 임베딩, geo 임베딩, 게이트 값)을 모두 넣어서,
      게이트가 1에 가까우면 모델이 자연히 geo 신호에 더 의존하도록
      학습 과정에서 가중치를 스스로 조절하게 합니다.
      (하드 라우팅이 아니라 추가 특징으로 제공하는 soft gate 방식이라
       학습이 불안정해지지 않습니다.)
    """

    def __init__(self):
        super().__init__()
        # R2, chin_angle, face_taper_r 등 Heart 핵심 지표 중심으로 작은 MLP
        self.net = nn.Sequential(
            nn.Linear(GEO_TOTAL, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, geo_raw: torch.Tensor) -> torch.Tensor:
        return self.net(geo_raw)   # (B, 1), 0~1


class FaceShapeNet(nn.Module):
    """
    EfficientNetV2-S(1280) + GeoEncoder(64) + HeartGate(1) → 분류기.

    v21과의 차이:
      classifier 입력에 HeartGate 출력(스칼라)을 추가로 concat.
      게이트 자체가 학습 가능한 파라미터이므로, 역전파를 통해
      "이 게이트 값이 분류에 도움이 되는 패턴"을 모델이 스스로 찾습니다.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        backbone = efficientnet_v2_s(weights=_weights_def)
        cnn_out  = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone   = backbone
        self.geo_enc    = GeoEncoder()
        self.heart_gate = HeartGate()
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(cnn_out + 64 + 1, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, img: torch.Tensor, geo: torch.Tensor):
        v = self.backbone(img)            # (B, 1280)
        g = self.geo_enc(geo)             # (B, 64)
        gate = self.heart_gate(geo)       # (B, 1)
        out = self.classifier(torch.cat([v, g, gate], dim=1))
        return out, gate.squeeze(1)       # 게이트 값도 반환 (설명 가능성용)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Transforms
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_tf(scale=1.1, hflip=False, train=False) -> T.Compose:
    ops = [T.Resize(int(IMG_SIZE * scale))]
    if train:
        ops += [
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(8),
            T.ColorJitter(0.25, 0.25, 0.2, 0.05),
        ]
    else:
        ops.append(T.CenterCrop(IMG_SIZE))
        if hflip:
            ops.append(T.RandomHorizontalFlip(p=1.0))
    ops += [T.ToTensor(), _normalize]
    if train:
        ops.append(T.RandomErasing(p=0.15, scale=(0.02, 0.1)))
    return T.Compose(ops)


TRAIN_TRANSFORM = _make_tf(1.15, train=True)
EVAL_TRANSFORM  = _make_tf(1.10, train=False)
TTA_TRANSFORMS = [
    _make_tf(1.10, False), _make_tf(1.10, True),
    _make_tf(1.15, False), _make_tf(1.15, True),
    _make_tf(1.05, False),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FaceDataset(Dataset):
    def __init__(self, records: list, class_to_idx: dict, transform):
        """
        records: [(cached_img_path, class_name, geo_18d_array, sample_weight)]
        """
        self.records      = records
        self.class_to_idx = class_to_idx
        self.transform    = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, cls, geo18, weight = self.records[idx]
        label = self.class_to_idx[cls]
        try:
            img_bgr = cv2.imread(str(path))
            img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            img_t   = self.transform(img_pil)
        except Exception:
            img_t  = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            label  = -1
            weight = 0.0
        geo_t = torch.tensor(geo18, dtype=torch.float32)
        return img_t, geo_t, label, weight


def collate_fn(batch):
    batch = [(i, g, l, w) for i, g, l, w in batch if l >= 0]
    if not batch:
        return None
    imgs, geos, lbls, weights = zip(*batch)
    return (
        torch.stack(imgs),
        torch.stack(geos),
        torch.tensor(lbls),
        torch.tensor(weights, dtype=torch.float32),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SWA BatchNorm 업데이트 (두 입력 모델용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def update_bn_two_input(loader, model, device):
    """
    torch.optim.swa_utils.update_bn 대체 함수.
    기본 update_bn은 model(input) 단일 입력을 가정하지만
    FaceShapeNet은 (img, geo) 두 입력이 필요하므로 별도 구현.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            momenta[module] = module.momentum
            module.momentum = None
    if not momenta:
        return
    was_training = model.training
    model.train()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, geos, _, _ = batch
            model(imgs.to(device), geos.to(device))
    model.train(was_training)
    for module, momentum in momenta.items():
        module.momentum = momentum


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 학습 / 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _weighted_ce(logits, targets, sample_weights, class_weight_tensor, label_smoothing=0.05):
    """
    표본별 가중치(앞머리 부분가림=0.5) × 클래스 가중치를 함께 반영한 CE.
    reduction='none'으로 개별 손실을 구한 뒤 표본 가중치로 평균.
    """
    per_sample = F.cross_entropy(
        logits, targets,
        weight=class_weight_tensor,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    weighted = per_sample * sample_weights
    denom = sample_weights.sum().clamp_min(1e-6)
    return weighted.sum() / denom


def train_one_epoch(model, loader, optimizer, scaler_amp, accum_steps,
                    class_weight_tensor=None, label_smoothing=0.05, mixup_alpha=0.0):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        if batch is None:
            continue
        imgs, geos, lbls, weights = batch
        imgs    = imgs.to(DEVICE, non_blocking=True)
        geos    = geos.to(DEVICE, non_blocking=True)
        lbls    = lbls.to(DEVICE, non_blocking=True)
        weights = weights.to(DEVICE, non_blocking=True)

        ctx = torch.autocast("cuda", torch.float16) if USE_AMP else nullcontext()

        if mixup_alpha > 0 and np.random.random() < 0.5:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(imgs.size(0), device=imgs.device)
            imgs_m  = lam * imgs  + (1 - lam) * imgs[idx]
            geos_m  = lam * geos  + (1 - lam) * geos[idx]
            lbls_b  = lbls[idx]
            w_b     = weights[idx]
            with ctx:
                out, _gate = model(imgs_m, geos_m)
                loss_a = _weighted_ce(out, lbls,   weights, class_weight_tensor, label_smoothing)
                loss_b = _weighted_ce(out, lbls_b, w_b,     class_weight_tensor, label_smoothing)
                loss   = (lam * loss_a + (1 - lam) * loss_b) / accum_steps
        else:
            with ctx:
                out, _gate = model(imgs, geos)
                loss = _weighted_ce(out, lbls, weights, class_weight_tensor, label_smoothing) / accum_steps

        if scaler_amp:
            scaler_amp.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            if scaler_amp:
                scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler_amp:
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def validate(model, loader):
    from sklearn.metrics import accuracy_score
    model.eval()
    losses, preds, targets = [], [], []

    for batch in loader:
        if batch is None:
            continue
        imgs, geos, lbls, weights = batch
        imgs = imgs.to(DEVICE, non_blocking=True)
        geos = geos.to(DEVICE, non_blocking=True)
        lbls = lbls.to(DEVICE, non_blocking=True)
        ctx  = torch.autocast("cuda", torch.float16) if USE_AMP else nullcontext()
        with ctx:
            out, _gate = model(imgs, geos)
            loss = F.cross_entropy(out, lbls)
        losses.append(loss.item())
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(lbls.cpu().numpy())

    avg_loss = sum(losses) / max(len(losses), 1)
    acc      = accuracy_score(targets, preds) if targets else 0.0
    return avg_loss, acc


@torch.inference_mode()
def predict_tta(model, img_bgr: np.ndarray, geo18: np.ndarray):
    """
    TTA 5-view 앙상블 확률을 반환한다.

    Returns
    -------
    probs : (num_classes,) 텐서 — 5-view 평균 softmax 확률
    gate  : float — Heart 게이트 평균값 (설명 가능성에 사용)
    """
    model.eval()
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    geo_t   = torch.tensor(geo18, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    total   = torch.zeros(model.num_classes)
    gates   = []
    ctx     = torch.autocast("cuda", torch.float16) if USE_AMP else nullcontext()
    for tf in TTA_TRANSFORMS:
        img_t = tf(img_pil).unsqueeze(0).to(DEVICE)
        with ctx:
            logits, gate = model(img_t, geo_t)
        total += F.softmax(logits, dim=1).squeeze(0).cpu()
        gates.append(gate.item())
    return total / len(TTA_TRANSFORMS), float(np.mean(gates))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v22 신규] 설명 가능성 — 왜 이 얼굴형으로 분류됐는지 사람이 읽는 리포트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def explain_prediction(
    geo_raw: "np.ndarray | None",
    pred_class: str,
    gate_value: float,
    class_geo_medians: "dict | None" = None,
) -> dict:
    """
    추론 결과를 사람이 읽을 수 있는 근거로 변환한다.

    Parameters
    ----------
    geo_raw     : 정규화 전 원본 17개 기하학 특징값 (None이면 측정 불가)
    pred_class  : 모델이 예측한 클래스명
    gate_value  : HeartGate 출력값 (0~1). 1에 가까울수록 기하학적으로
                  Heart 신호가 강했고, 모델이 geo 정보에 더 의존했다는 뜻.
    class_geo_medians : {클래스명: {특징명: 중앙값}} 참고용 기준값 (선택)

    Returns
    -------
    dict — 삼정 비율, 핵심 각도, 게이트 해설, 종합 한 줄 설명을 담은 딕셔너리
    """
    if geo_raw is None:
        return {
            "available": False,
            "message": "얼굴 랜드마크를 정밀하게 측정하지 못해 "
                       "수치 기반 설명을 제공할 수 없습니다. CNN 예측만 사용되었습니다.",
        }

    feat = dict(zip(FEATURE_NAMES, geo_raw.tolist()))

    삼정 = {
        "이마(상안부)": f"{feat['upper_r']*100:.1f}%",
        "중안부(눈코)": f"{feat['mid_r']*100:.1f}%",
        "하안부(입턱)": f"{feat['lower_r']*100:.1f}%",
        "균등 여부": "균등(황금비에 근접)" if feat["golden_flag"] > 0.5 else "불균등",
    }

    핵심수치 = {
        "하악각 (jaw_angle)": f"{feat['jaw_angle']:.1f}°  "
                              f"(square≈90°, round≈140° 기준)",
        "턱끝 각도 (chin_angle)": f"{feat['chin_angle']:.1f}°  "
                                 f"(작을수록 뾰족한 턱)",
        "이마/턱 비율 (R2)": f"{feat['R2']:.3f}  "
                             f"(클수록 이마가 넓고 턱이 좁음 → Heart 경향)",
        "턱/이마 비율 (face_taper_r)": f"{feat['face_taper_r']:.3f}  "
                                       f"(작을수록 Heart 경향)",
        "세로 길이감 (R1)": f"{feat['R1']:.3f}  (클수록 긴 얼굴)",
    }

    gate_pct = gate_value * 100
    if gate_value > 0.6:
        gate_desc = (
            f"Heart 게이트 {gate_pct:.0f}% — 이마는 넓고 턱은 좁은 기하학적 "
            f"패턴이 뚜렷하게 감지되어, 모델이 윤곽(CNN) 정보보다 "
            f"랜드마크 비율(R2·chin_angle) 정보에 더 의존해 판단했습니다."
        )
    elif gate_value > 0.3:
        gate_desc = (
            f"Heart 게이트 {gate_pct:.0f}% — 기하학적 Heart 신호가 일부 감지되어 "
            f"CNN 윤곽 정보와 랜드마크 비율 정보를 균형 있게 반영했습니다."
        )
    else:
        gate_desc = (
            f"Heart 게이트 {gate_pct:.0f}% — Heart 특유의 비율 패턴이 약해, "
            f"모델이 주로 CNN의 윤곽 정보로 판단했습니다."
        )

    summary = (
        f"'{pred_class}'으로 분류된 핵심 근거: "
        f"삼정 비율은 이마 {삼정['이마(상안부)']} · 중안부 {삼정['중안부(눈코)']} · "
        f"하안부 {삼정['하안부(입턱)']}이며, "
        f"하악각은 {feat['jaw_angle']:.1f}°, 턱끝 각도는 {feat['chin_angle']:.1f}°입니다."
    )

    result = {
        "available": True,
        "삼정비율": 삼정,
        "핵심수치": 핵심수치,
        "heart_gate": gate_desc,
        "summary": summary,
        "raw_features": feat,
    }

    if class_geo_medians and pred_class in class_geo_medians:
        ref = class_geo_medians[pred_class]
        result["reference_comparison"] = {
            k: f"이 사진 {feat[k]:.3f}  vs  {pred_class} 평균 {ref.get(k, float('nan')):.3f}"
            for k in ["R1", "R2", "jaw_angle", "chin_angle", "face_taper_r"]
            if k in feat and k in ref
        }

    return result
