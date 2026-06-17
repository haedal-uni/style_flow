"""
Personal Color Inference — v09 Hierarchical Pipeline

학습된 v09 계층형 파이프라인으로 이미지(또는 직접 색상 값)를 받아
봄/여름/가을/겨울 퍼스널 컬러를 예측

Usage:
  python infer.py --image photo.jpg
  python infer.py --image photo.jpg --model-dir outputs
  python infer.py --hex d3af8c
  python infer.py --rgb 210 175 140

Prerequisites:
  Run train_personal_color.py first to generate model files in outputs/
"""

import argparse
import colorsys
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn

# Model (identical to v09 training architecture)
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Color space conversion 
def _linearize(c: float) -> float:
    """sRGB → linear RGB (gamma removal)."""
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """sRGB (0–255) → CIE L*a*b* (D65 white point)."""
    rl, gl, bl = _linearize(r), _linearize(g), _linearize(b)
    X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_lab = 200 * (fy - fz)
    return L, a, b_lab


def compute_all_features(r: int, g: int, b: int) -> Dict[str, float]:
    """RGB → 18 feature dictionary used by v09 models."""
    L, a, b_lab = rgb_to_lab(r, g, b)

    C = math.sqrt(a ** 2 + b_lab ** 2)
    H = math.degrees(math.atan2(b_lab, a)) % 360

    r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
    _, S, V = colorsys.rgb_to_hsv(r_n, g_n, b_n)

    eps = 1e-6
    h_rad = math.radians(H)

    return {
        "L": L,
        "a": a,
        "b": b_lab,
        "C": C,
        "S": S,
        "V": V,
        "H_sin": math.sin(h_rad),
        "H_cos": math.cos(h_rad),
        "R": r_n,
        "G": g_n,
        "B": b_n,
        "C_div_L": C / (L + eps),
        "S_div_V": S / (V + eps),
        "a_div_b": a / (b_lab + eps),
        "warm_yellow_score": b_lab - abs(a),
        "red_yellow_sum": a + b_lab,
        "clarity_score": C + S,
        "darkness_score": 100.0 - L,
    }


# Representative color extraction from image
def extract_representative_color(image_path: str) -> Tuple[int, int, int]:
    """
    Extract a representative skin-tone color from an image.

    Strategy:
      1. Resize to at most 300×300 (speed)
      2. Remove too-bright (white background) and too-dark (black background) pixels
      3. Remove near-achromatic pixels (low saturation = likely background)
      4. Return the median RGB of remaining pixels
    """
    try:
        from PIL import Image
    except ImportError:
        print("[ERROR] Pillow is not installed. Run: pip install Pillow")
        sys.exit(1)

    img = Image.open(image_path).convert("RGB")

    max_dim = 300
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)
    pixels = arr.reshape(-1, 3)

    R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    brightness = (R + G + B) / 3.0
    mask = (brightness >= 30) & (brightness <= 230)

    r_n, g_n, b_n = R / 255.0, G / 255.0, B / 255.0
    max_c = np.maximum(np.maximum(r_n, g_n), b_n)
    min_c = np.minimum(np.minimum(r_n, g_n), b_n)
    saturation = np.where(max_c > 0, np.divide(max_c - min_c, max_c, where=max_c > 0, out=np.zeros_like(max_c)), 0.0)
    mask &= saturation >= 0.05

    filtered = pixels[mask]

    if len(filtered) < 10:
        mask = (brightness >= 20) & (brightness <= 240)
        filtered = pixels[mask]

    if len(filtered) == 0:
        filtered = pixels

    median_rgb = np.median(filtered, axis=0).astype(int)
    r, g, b = int(median_rgb[0]), int(median_rgb[1]), int(median_rgb[2])
    return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))


# Model loading 
def load_bundle(pt_path: Path, scaler_path: Path, device: torch.device):
    """Load a .pt checkpoint + scaler .joblib and return (model, scaler, class_names, feature_cols)."""
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    input_dim: int = ckpt["input_dim"]
    num_classes: int = ckpt["num_classes"]
    class_names: List[str] = ckpt["class_names"]
    feature_cols: List[str] = ckpt["feature_cols"]
    hidden_dim: int = ckpt.get("hidden_dim", 64)

    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    scaler = joblib.load(scaler_path)
    return model, scaler, class_names, feature_cols


# Hierarchical pipeline prediction
def predict(r: int, g: int, b: int, model_dir: Path, device: torch.device) -> Dict:
    """
    RGB → Hierarchical pipeline (Stage 1: Warm/Cool → Stage 2) → season prediction.

    Returns:
      {
        "season": "Spring" | "Summer" | "Autumn" | "Winter",
        "stage1": {"prediction": "Warm"|"Cool", "probabilities": {...}},
        "stage2": {"prediction": "Spring"|..., "probabilities": {...}},
        "input_color": {"R": r, "G": g, "B": b, "hex": "#rrggbb", "L": ..., "a": ..., "b_lab": ...},
      }
    """
    temp_pt = model_dir / "temperature_spring_summer_autumn_winter.pt"
    temp_sc = model_dir / "temperature_spring_summer_autumn_winter_scaler.joblib"
    warm_pt = model_dir / "season_spring_autumn.pt"
    warm_sc = model_dir / "season_spring_autumn_scaler.joblib"
    cool_pt = model_dir / "season_summer_winter.pt"
    cool_sc = model_dir / "season_summer_winter_scaler.joblib"

    for p in [temp_pt, temp_sc, warm_pt, warm_sc, cool_pt, cool_sc]:
        if not p.exists():
            print(f"[ERROR] Model file not found: {p}")
            print("  Run 'python train_personal_color.py --epochs 800' first.")
            sys.exit(1)

    temp_model, temp_scaler, temp_classes, temp_fcols = load_bundle(temp_pt, temp_sc, device)
    warm_model, warm_scaler, warm_classes, warm_fcols = load_bundle(warm_pt, warm_sc, device)
    cool_model, cool_scaler, cool_classes, cool_fcols = load_bundle(cool_pt, cool_sc, device)

    features = compute_all_features(r, g, b)

    def make_x(fcols, scaler):
        x = np.array([[features[c] for c in fcols]], dtype=np.float32)
        return scaler.transform(x)

    # Stage 1: Warm / Cool
    X1 = torch.tensor(make_x(temp_fcols, temp_scaler), device=device)
    with torch.no_grad():
        probs1 = torch.softmax(temp_model(X1), dim=1).cpu().numpy()[0]
    temp_pred = temp_classes[int(probs1.argmax())]
    stage1_probs = dict(zip(temp_classes, [round(float(p), 4) for p in probs1]))

    # Stage 2
    if temp_pred == "Warm":
        X2 = torch.tensor(make_x(warm_fcols, warm_scaler), device=device)
        with torch.no_grad():
            probs2 = torch.softmax(warm_model(X2), dim=1).cpu().numpy()[0]
        season = warm_classes[int(probs2.argmax())]
        stage2_probs = dict(zip(warm_classes, [round(float(p), 4) for p in probs2]))
    else:
        X2 = torch.tensor(make_x(cool_fcols, cool_scaler), device=device)
        with torch.no_grad():
            probs2 = torch.softmax(cool_model(X2), dim=1).cpu().numpy()[0]
        season = cool_classes[int(probs2.argmax())]
        stage2_probs = dict(zip(cool_classes, [round(float(p), 4) for p in probs2]))

    return {
        "season": season,
        "stage1": {"prediction": temp_pred, "probabilities": stage1_probs},
        "stage2": {"prediction": season, "probabilities": stage2_probs},
        "input_color": {
            "R": r, "G": g, "B": b,
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "L": round(features["L"], 2),
            "a": round(features["a"], 2),
            "b_lab": round(features["b"], 2),
        },
    }


# Result display
SEASON_KO = {"Spring": "봄", "Summer": "여름", "Autumn": "가을", "Winter": "겨울"}
SEASON_DESC = {
    "Spring": "밝고 따뜻한 톤 — 노란 기운의 선명한 색상이 어울립니다.",
    "Summer": "밝고 차가운 톤 — 블루/핑크 기운의 부드러운 색상이 어울립니다.",
    "Autumn": "깊고 따뜻한 톤 — 황갈색·올리브 계열의 풍부한 색상이 어울립니다.",
    "Winter": "깊고 차가운 톤 — 선명하고 대비가 강한 색상이 어울립니다.",
}


def print_result(result: Dict) -> None:
    c = result["input_color"]
    s1 = result["stage1"]
    s2 = result["stage2"]
    season = result["season"]

    print()
    print("=" * 50)
    print(f"  퍼스널 컬러 결과: {season} ({SEASON_KO[season]})")
    print("=" * 50)
    print(f"  {SEASON_DESC[season]}")
    print()
    print(f"  입력 색상   : RGB({c['R']}, {c['G']}, {c['B']})  {c['hex']}")
    print(f"               L={c['L']}  a={c['a']}  b*={c['b_lab']}")
    print()
    print(f"  Stage 1  Warm/Cool → {s1['prediction']}")
    for k, v in s1["probabilities"].items():
        bar = "█" * int(v * 20)
        print(f"    {k:6s} {v:.3f}  {bar}")
    print()
    print(f"  Stage 2  계절 분류 → {s2['prediction']}")
    for k, v in s2["probabilities"].items():
        bar = "█" * int(v * 20)
        print(f"    {k:6s} {v:.3f}  {bar}")
    print("=" * 50)
    print()


# Main
def main():
    default_model_dir = Path(__file__).parent / "outputs"

    parser = argparse.ArgumentParser(
        description="Personal color prediction via v09 hierarchical pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py --image photo.jpg
  python infer.py --hex d3af8c
  python infer.py --rgb 210 175 140
  python infer.py --image photo.jpg --model-dir outputs
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, metavar="PATH",
                             help="Image file path (jpg/png). Representative color is extracted automatically.")
    input_group.add_argument("--hex", type=str, metavar="RRGGBB",
                             help="Hex color code (e.g. d3af8c or #d3af8c)")
    input_group.add_argument("--rgb", type=int, nargs=3, metavar=("R", "G", "B"),
                             help="RGB values 0–255. e.g. --rgb 210 175 140")

    parser.add_argument("--model-dir", type=str, default=str(default_model_dir),
                        help=f"Folder containing .pt and .joblib model files (default: {default_model_dir})")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)

    if args.image:
        print(f"[Image] {args.image}")
        r, g, b = extract_representative_color(args.image)
        print(f"  Extracted color: RGB({r}, {g}, {b})  #{r:02x}{g:02x}{b:02x}")
    elif args.hex:
        h = args.hex.strip().lstrip("#")
        if len(h) != 6:
            print(f"[ERROR] Invalid hex value: {args.hex}")
            sys.exit(1)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    else:
        r, g, b = args.rgb
        for val in (r, g, b):
            if not (0 <= val <= 255):
                print(f"[ERROR] RGB values must be in 0–255 range: {r} {g} {b}")
                sys.exit(1)

    result = predict(r, g, b, model_dir, device)
    print_result(result)


if __name__ == "__main__":
    main()
