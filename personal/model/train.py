import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
WARM_COOL_MAP = {"Spring": "Warm", "Autumn": "Warm", "Summer": "Cool", "Winter": "Cool"}
CSV_FILENAME = "personal_color_palette_full.csv"


#  CSV 자동 탐색
def find_csv() -> Path:
    script_dir = Path(__file__).parent
    candidates = [
        script_dir / CSV_FILENAME,
        script_dir.parent / CSV_FILENAME,
        Path.cwd() / CSV_FILENAME,
    ]
    for p in candidates:
        if p.exists():
            print(f"[CSV 자동 탐색] 발견: {p.resolve()}")
            return p
    searched = "\n  ".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(
        f"{CSV_FILENAME} 파일을 찾을 수 없습니다.\n탐색한 경로:\n  {searched}\n"
        f"--csv 옵션으로 경로를 직접 지정하거나, 파일을 위 경로 중 하나에 놓아주세요."
    )


# 유틸
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = str(hex_color).strip().replace("#", "")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    data = df.copy()
    for col in ["L", "a", "b", "C", "H", "S", "V"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    rgb = data["hex"].apply(hex_to_rgb)
    data["R"] = rgb.apply(lambda x: x[0] / 255.0)
    data["G"] = rgb.apply(lambda x: x[1] / 255.0)
    data["B"] = rgb.apply(lambda x: x[2] / 255.0)

    h_rad = np.deg2rad(data["H"].astype(float))
    data["H_sin"] = np.sin(h_rad)
    data["H_cos"] = np.cos(h_rad)

    eps = 1e-6
    data["C_div_L"] = data["C"] / (data["L"] + eps)
    data["S_div_V"] = data["S"] / (data["V"] + eps)
    data["a_div_b"] = data["a"] / (data["b"] + eps)
    data["warm_yellow_score"] = data["b"] - np.abs(data["a"])
    data["red_yellow_sum"] = data["a"] + data["b"]
    data["clarity_score"] = data["C"] + data["S"]
    data["darkness_score"] = 100.0 - data["L"]

    feature_cols = [
        "L", "a", "b", "C", "S", "V",
        "H_sin", "H_cos",
        "R", "G", "B",
        "C_div_L", "S_div_V", "a_div_b",
        "warm_yellow_score", "red_yellow_sum", "clarity_score", "darkness_score",
    ]
    X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X, feature_cols


# 모델
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

@dataclass
class TrainResult:
    model: nn.Module
    history: Dict[str, List[float]]
    best_val_f1: float

@dataclass
class ModelBundle:
    """학습된 모델 + 스케일러 + 메타 정보를 하나로 보관"""
    model: nn.Module
    scaler: StandardScaler
    class_names: List[str]
    feature_cols: List[str]
    best_val_f1: float
    history: Dict = field(default_factory=dict)


# 학습 공통 로직
def make_class_weights(y_train: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)

def train_one_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int, device: torch.device,
    epochs: int = 800, batch_size: int = 32,
    lr: float = 1e-3, patience: int = 80,
) -> TrainResult:
    model = MLPClassifier(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    class_weights = make_class_weights(y_train, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=20)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=(len(train_ds) % batch_size == 1),
    )
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_state, best_val_f1, bad_epochs = None, -1.0, 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_f1": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv).argmax(dim=1).cpu().numpy()

        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, val_pred)
        scheduler.step(val_f1)

        history["train_loss"].append(float(np.mean(losses)))
        history["val_f1"].append(float(val_f1))
        history["val_acc"].append(float(val_acc))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 100 == 0 or epoch == 1:
            print(f"  epoch={epoch:04d}  loss={np.mean(losses):.4f}  val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}")

        if bad_epochs >= patience:
            print(f"  Early stopping at epoch {epoch}. Best val macro F1={best_val_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, history=history, best_val_f1=float(best_val_f1))


def evaluate_model(
    model: nn.Module, X: np.ndarray, y: np.ndarray,
    class_names: List[str], device: torch.device,
) -> Dict:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32, device=device)).argmax(dim=1).cpu().numpy()

    report = classification_report(y, pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred).tolist()
    print(classification_report(y, pred, target_names=class_names, zero_division=0))
    print(pd.DataFrame(cm, index=class_names, columns=class_names))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def _build_and_save_bundle(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    class_names: List[str], feature_cols: List[str],
    scaler: StandardScaler, model_stem: str,
    out_dir: Path, args, device: torch.device,
    epochs: int, patience: int,
) -> Tuple[ModelBundle, Dict]:
    """공통 학습 + 저장 + 평가 루틴."""
    result = train_one_model(
        X_train, y_train, X_test, y_test,
        num_classes=len(class_names), device=device,
        epochs=epochs, batch_size=args.batch_size, lr=args.lr, patience=patience,
    )

    print(f"\n[{model_stem}] Test set 평가:")
    metrics = evaluate_model(result.model, X_test, y_test, class_names, device)

    torch.save(
        {
            "model_state_dict": result.model.state_dict(),
            "input_dim": X_train.shape[1],
            "num_classes": len(class_names),
            "feature_cols": feature_cols,
            "class_names": class_names,
        },
        out_dir / f"{model_stem}.pt",
    )
    joblib.dump(scaler, out_dir / f"{model_stem}_scaler.joblib")

    bundle = ModelBundle(
        model=result.model,
        scaler=scaler,
        class_names=class_names,
        feature_cols=feature_cols,
        best_val_f1=result.best_val_f1,
        history=result.history,
    )
    return bundle, metrics


# 개별 모델 학습 함수
def train_four_class_model(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    out_dir: Path, args, device: torch.device,
) -> Tuple[ModelBundle, Dict]:
    print("\n" + "=" * 60)
    print("Four-Class Model  (Spring / Summer / Autumn / Winter)")
    print("=" * 60)

    X_tr_df, feature_cols = build_features(df_train)
    X_te_df, _ = build_features(df_test)

    le = LabelEncoder().fit(SEASON_ORDER)
    y_tr = le.transform(df_train["season"].astype(str))
    y_te = le.transform(df_test["season"].astype(str))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_df.values.astype(np.float32))
    X_te = scaler.transform(X_te_df.values.astype(np.float32))

    return _build_and_save_bundle(
        X_tr, y_tr, X_te, y_te,
        class_names=list(le.classes_),
        feature_cols=feature_cols,
        scaler=scaler,
        model_stem="best_four_class_model",
        out_dir=out_dir, args=args, device=device,
        epochs=args.epochs, patience=args.patience,
    )


def train_temp_model(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    out_dir: Path, args, device: torch.device,
) -> Tuple[ModelBundle, Dict]:
    """Stage 1: Warm / Cool 분류기 (전체 데이터 사용)"""
    print("\n" + "=" * 60)
    print("Stage 1 — Temperature Model  (Warm / Cool)")
    print("=" * 60)

    df_tr = df_train.copy()
    df_te = df_test.copy()
    df_tr["temperature"] = df_tr["season"].map(WARM_COOL_MAP)
    df_te["temperature"] = df_te["season"].map(WARM_COOL_MAP)

    X_tr_df, feature_cols = build_features(df_tr)
    X_te_df, _ = build_features(df_te)

    le = LabelEncoder().fit(["Cool", "Warm"])
    y_tr = le.transform(df_tr["temperature"])
    y_te = le.transform(df_te["temperature"])

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_df.values.astype(np.float32))
    X_te = scaler.transform(X_te_df.values.astype(np.float32))

    return _build_and_save_bundle(
        X_tr, y_tr, X_te, y_te,
        class_names=list(le.classes_),
        feature_cols=feature_cols,
        scaler=scaler,
        model_stem="temperature_spring_summer_autumn_winter",
        out_dir=out_dir, args=args, device=device,
        epochs=max(300, args.epochs // 2), patience=60,
    )


def train_branch_model(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    season_pair: List[str],
    out_dir: Path, args, device: torch.device,
) -> Tuple[Optional[ModelBundle], Dict]:
    """Stage 2: 두 계절 구별 분류기 (해당 계절 샘플만 사용)"""
    label = "_".join(s.lower() for s in season_pair)
    model_stem = f"season_{label}"

    print(f"\n{'=' * 60}")
    print(f"Stage 2 — Branch Model  ({' / '.join(season_pair)})")
    print("=" * 60)

    df_tr = df_train[df_train["season"].isin(season_pair)].copy()
    df_te = df_test[df_test["season"].isin(season_pair)].copy()

    if df_tr["season"].nunique() < 2 or len(df_tr) < 10:
        print("  [SKIP] 학습 데이터가 부족합니다.")
        return None, {"skipped": True, "reason": "not enough training data"}

    X_tr_df, feature_cols = build_features(df_tr)
    X_te_df, _ = build_features(df_te)

    le = LabelEncoder().fit(season_pair)
    y_tr = le.transform(df_tr["season"].astype(str))
    y_te = le.transform(df_te["season"].astype(str))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_df.values.astype(np.float32))
    X_te = scaler.transform(X_te_df.values.astype(np.float32))

    return _build_and_save_bundle(
        X_tr, y_tr, X_te, y_te,
        class_names=list(le.classes_),
        feature_cols=feature_cols,
        scaler=scaler,
        model_stem=model_stem,
        out_dir=out_dir, args=args, device=device,
        epochs=max(300, args.epochs // 2), patience=60,
    )


# 계층형 파이프라인 전체 평가
def evaluate_hierarchical_pipeline(
    X_test_raw: np.ndarray,
    y_test_seasons: np.ndarray,
    temp_bundle: ModelBundle,
    warm_bundle: ModelBundle,
    cool_bundle: ModelBundle,
    device: torch.device,
) -> Dict:
    """
    Stage 1(Warm/Cool) → Stage 2(Spring/Autumn or Summer/Winter) 파이프라인을
    전체 test set에 통과시켜 최종 4계절 accuracy / macro F1 을 계산

    - Stage 1 오분류가 Stage 2로 전파되는 효과까지 반영한 실제 end-to-end 성능
    - 각 stage 개별 정확도와 달리 실제 서비스 투입 시 기대 성능에 해당
    """
    n = len(X_test_raw)

    # Stage 1: Warm / Cool 예측
    X1 = temp_bundle.scaler.transform(X_test_raw)
    temp_bundle.model.eval()
    with torch.no_grad():
        t_idx = temp_bundle.model(
            torch.tensor(X1, dtype=torch.float32, device=device)
        ).argmax(dim=1).cpu().numpy()
    temp_preds = np.array(temp_bundle.class_names)[t_idx]

    final_preds = np.empty(n, dtype=object)
    warm_mask = temp_preds == "Warm"

    # Stage 2a: Warm → Spring / Autumn
    if warm_mask.any():
        X2w = warm_bundle.scaler.transform(X_test_raw[warm_mask])
        warm_bundle.model.eval()
        with torch.no_grad():
            w_idx = warm_bundle.model(
                torch.tensor(X2w, dtype=torch.float32, device=device)
            ).argmax(dim=1).cpu().numpy()
        final_preds[warm_mask] = np.array(warm_bundle.class_names)[w_idx]

    # Stage 2b: Cool → Summer / Winter
    cool_mask = ~warm_mask
    if cool_mask.any():
        X2c = cool_bundle.scaler.transform(X_test_raw[cool_mask])
        cool_bundle.model.eval()
        with torch.no_grad():
            c_idx = cool_bundle.model(
                torch.tensor(X2c, dtype=torch.float32, device=device)
            ).argmax(dim=1).cpu().numpy()
        final_preds[cool_mask] = np.array(cool_bundle.class_names)[c_idx]

    acc = float(accuracy_score(y_test_seasons, final_preds))
    f1 = float(f1_score(
        y_test_seasons, final_preds,
        average="macro", labels=SEASON_ORDER, zero_division=0,
    ))
    report = classification_report(
        y_test_seasons, final_preds,
        labels=SEASON_ORDER, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test_seasons, final_preds, labels=SEASON_ORDER).tolist()

    print("\n" + "=" * 60)
    print("Hierarchical Pipeline — 최종 4계절 평가 (end-to-end)")
    print("=" * 60)
    print(f"  Warm branch로 라우팅된 샘플: {warm_mask.sum()}개")
    print(f"  Cool branch로 라우팅된 샘플: {cool_mask.sum()}개\n")
    print(classification_report(
        y_test_seasons, final_preds, labels=SEASON_ORDER, zero_division=0,
    ))
    print(pd.DataFrame(cm, index=SEASON_ORDER, columns=SEASON_ORDER))

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "warm_routed": int(warm_mask.sum()),
        "cool_routed": int(cool_mask.sum()),
    }


# Cross Validation
def cross_validate_mlp(
    X: np.ndarray, y: np.ndarray,
    class_names: List[str], device: torch.device,
    epochs: int, folds: int = 5,
) -> Dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])
        result = train_one_model(
            X_tr, y[tr_idx], X_va, y[va_idx],
            num_classes=len(class_names), device=device,
            epochs=epochs, patience=60,
        )
        metrics = evaluate_model(result.model, X_va, y[va_idx], class_names, device)
        rows.append({"fold": fold, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
        print(f"  Fold {fold}: acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")
    return {
        "folds": rows,
        "mean_accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "mean_macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
    }


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Personal color classifier (v09). "
                    "계층형 파이프라인의 end-to-end 4계절 정확도를 측정"
    )
    parser.add_argument("--csv", type=str, default=None,
                        help=f"{CSV_FILENAME}을 자동 탐색. 수동 지정 시 이 옵션 사용.")
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "outputs"))
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", action="store_true", help="5-fold CV 후 최종 학습 진행")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # CSV 로드
    csv_path = Path(args.csv) if args.csv else find_csv()
    df = pd.read_csv(csv_path)

    required_cols = {"season", "hex", "L", "a", "b", "C", "H", "S", "V"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["season"].isin(SEASON_ORDER)].copy()
    df = df.drop_duplicates(subset=["season", "hex", "L", "a", "b", "C", "H", "S", "V"]).reset_index(drop=True)

    print(f"\nData: {len(df)} rows")
    print(df["season"].value_counts().to_string())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 공유 train / test split (모든 모델이 동일한 분할을 사용)
    X_df, _ = build_features(df)
    y_all = LabelEncoder().fit(SEASON_ORDER).transform(df["season"].astype(str))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X_df, y_all))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    print(f"\nTrain: {len(df_train)} / Test: {len(df_test)}")

    # Optional: 5-fold CV
    cv_metrics = None
    if args.cv:
        print("\n" + "=" * 60)
        print("5-Fold Cross Validation (Four-Class)")
        print("=" * 60)
        cv_metrics = cross_validate_mlp(
            X_df.values.astype(np.float32), y_all, SEASON_ORDER, device,
            epochs=max(300, args.epochs // 2), folds=5,
        )
        print(f"\n  CV mean acc={cv_metrics['mean_accuracy']:.4f}  mean_macro_f1={cv_metrics['mean_macro_f1']:.4f}")

    # 모델 학습 (모두 동일한 df_train / df_test 사용)
    four_bundle, four_metrics = train_four_class_model(df_train, df_test, out_dir, args, device)
    temp_bundle, temp_metrics = train_temp_model(df_train, df_test, out_dir, args, device)
    warm_bundle, warm_metrics = train_branch_model(df_train, df_test, ["Spring", "Autumn"], out_dir, args, device)
    cool_bundle, cool_metrics = train_branch_model(df_train, df_test, ["Summer", "Winter"], out_dir, args, device)

    # 계층형 파이프라인 end-to-end 평가
    X_test_raw, _ = build_features(df_test)
    X_test_raw = X_test_raw.values.astype(np.float32)
    y_test_seasons = df_test["season"].values

    pipeline_metrics = evaluate_hierarchical_pipeline(
        X_test_raw, y_test_seasons,
        temp_bundle, warm_bundle, cool_bundle,
        device,
    )

    # 최종 비교 출력
    print("\n" + "=" * 60)
    print("성능 비교 요약")
    print("=" * 60)
    print(f"  Four-Class 직접 분류       acc={four_metrics['accuracy']:.4f}  macro_f1={four_metrics['macro_f1']:.4f}")
    print(f"  Hierarchical 파이프라인    acc={pipeline_metrics['accuracy']:.4f}  macro_f1={pipeline_metrics['macro_f1']:.4f}")
    print()
    print(f"  (Stage 1  Warm/Cool        acc={temp_metrics['accuracy']:.4f}  macro_f1={temp_metrics['macro_f1']:.4f})")
    if not warm_metrics.get("skipped"):
        print(f"  (Stage 2a Spring/Autumn    acc={warm_metrics['accuracy']:.4f}  macro_f1={warm_metrics['macro_f1']:.4f})")
    if not cool_metrics.get("skipped"):
        print(f"  (Stage 2b Summer/Winter    acc={cool_metrics['accuracy']:.4f}  macro_f1={cool_metrics['macro_f1']:.4f})")

    # metrics.json 저장
    all_metrics = {
        "device": str(device),
        "csv": str(csv_path.resolve()),
        "data_shape": list(df.shape),
        "class_counts": df["season"].value_counts().to_dict(),
        "train_size": len(df_train),
        "test_size": len(df_test),
        "cross_validation": cv_metrics,
        "four_class": four_metrics,
        "hierarchical": {
            "stage1_warm_cool": temp_metrics,
            "stage2_spring_autumn": warm_metrics,
            "stage2_summer_winter": cool_metrics,
            "pipeline_final": pipeline_metrics,
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
