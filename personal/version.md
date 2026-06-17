# Version History — Personal Color Classifier

v01부터 v10까지 각 버전의 구체적인 설계 결정, 실험 결과, 실패 원인 분석을 기록했다.

<br><br><br>  

--- 

<br>

## v01 — 베이스라인: EfficientNetB0 + 색상 피처

### 설계

**목적:** 얼굴 이미지에서 직접 4계절 퍼스널 컬러를 분류하는 CNN 모델 베이스라인 구축

<br>

**모델 아키텍처**
- 백본: EfficientNetB0 (4.75M params, ImageNet pretrained)
- 색상 피처 브랜치: 13차원 — LAB/HSV 통계(9) + 시즌별 팔레트 최소거리(4)
- 결합: CNN 1280차원 + 색상 브랜치 32차원 → concat → 분류 헤드

<br>

**학습 전략**
- Phase 1 (10 epochs): 백본 동결, 헤드만 학습 / AdamW lr=1e-3
- Phase 2 (35 epochs): EfficientNetB0 block 5-8 언프리즈 / backbone lr=2e-5, head lr=5e-4
- CosineAnnealingLR, Label Smoothing 0.1, Gradient Clipping 1.0
- 데이터: ColorJitter 포함, 검정 배경(0,0,0) 그대로 사용

<br>

**데이터셋:** RGB-M 마스크 얼굴 이미지 / Train 4,008장 / Test 912장

<br><br>

### 실험 결과

| Phase | Epoch | Train Acc | Test Acc |
|-------|-------|-----------|----------|
| P1 | 10/10 | 55.14% | 52.19% |
| P2 | 07/35 ← best | 53.77% | **53.29%** |
| P2 | 35/35 | 70.41% | 50.88% |

Best: Train 53.77% / Test **53.29%**

<br><br>

### 문제점 분석

| 문제 | 원인 | 영향 |
|------|------|------|
| ColorJitter 사용 | 피부톤 색상(분류 신호)을 무작위로 변환 | 봄 웜톤 Recall 22% |
| 검정 배경 그대로 | ImageNet 정규화(mean=[0.485, 0.456, 0.406])와 불일치 | CNN 입력 왜곡 |
| 13차원 피처 단순 | mean/std만 사용, 색상 분포 형태 미반영 | 피처 표현력 부족 |
| Phase 2 과적합 | P2-07 이후 train 계속 상승, test 51~53% 정체 | 일반화 실패 |

<br><br><br>  

--- 

<br>

## v02 — EfficientNetV2-S + 70차원 피처 + 배경 처리

### 설계 변경

**v01 문제 3가지 집중 개선:**

1. **ColorJitter 완전 제거** — 피부톤이 분류 신호이므로 색 변환은 금지
2. **배경 → 중립 회색(128,128,128) + 얼굴 ROI 크롭** — ImageNet 정규화와 더 잘 맞는 입력 분포
3. **70차원 색상 피처** — 색상 모멘트(18: LAB/HSV mean/std/skew) + 색상 히스토그램(48: 8bin×6채널) + 팔레트 거리(4)

<br>

**모델 업그레이드**
- 백본: EfficientNetV2-S (21M params) — v01 대비 4.4배 큰 모델
- 색상 브랜치: Linear(70→256)→BN→GELU → Linear(256→128)→BN→GELU
- 3-Phase 학습 + OneCycleLR + Mixup(α=0.3)

<br>

| Phase | Epochs | 설정 | LR |
|-------|--------|------|----|
| P1 | 1–10 | 백본 동결 | head: 5e-4 |
| P2 | 11–35 | 블록 4–7 언프리즈 | backbone: 5e-5 / head: 5e-4 |
| P3 | 36–60 | 전체 언프리즈 | backbone: 2.5e-5 / head: 2.5e-4 |

<br><br>

### 실험 결과

| Phase | Epoch | Train Acc | Test Acc |
|-------|-------|-----------|----------|
| P1 | 05/10 | 46.38% | 53.84% |
| P2 | 07/25 ← **best** | 59.93% | **55.37%** |
| P2 | 22/25 | 86.68% | 50.33% |

Best: Train 59.93% / Test **55.37%** (+2.08%p vs v01)

P3는 도달하지 못하고 P2-24에서 조기 중단.

<br><br>

### 문제점 분석

```
Phase 2 train/test 정확도 추이:

%
90│                              ████ train (86%)
80│                        ████
70│                    ████
60│               ████
50│████████████████         test (50%)
  └──────────────────────────────
   P2-01   P2-07   P2-13   P2-22
              ↑ best (55.37%)
```

- P2-08부터 train만 급등, test는 55%→50% 역행 → **전형적 과적합**
- 원인: 21M 파라미터 ÷ 4,008장 = 파라미터/데이터 비율 과다
- test set을 validation으로 사용 → 데이터 누출, early stopping 불가
- backbone LR 5e-5가 너무 높아 빠른 암기

<br><br><br>  

--- 

<br>

## v03 — Train/Val/Test 분리 + Early Stopping + EfficientNet-B3

### 설계 변경

**v02 문제 4가지 집중 개선:**

1. **EfficientNet-B3 (12M)** — v02 V2-S(21M)에서 모델 용량 축소
2. **Train 80% / Val 20% 분리** — test는 최종 1회만 평가 (데이터 누출 해결)
3. **patience 기반 Early Stopping** — P2 patience=8, P3 patience=5
4. **backbone LR 대폭 감소** — P2: 1e-5, P3: 5e-6 (v02 5e-5에서 1/5로 감소)

<br>

**추가 강화**
- dropout=0.6 (v02 0.5 → 0.6)
- weight_decay=5e-4 (v02 2e-4 → 5e-4)
- Label Smoothing=0.1 (v02 없음)
- Mixup α=0.4 (v02 0.3 → 0.4)
- RandomAffine 증강 추가

<br><br>

### 실험 결과

| Phase | Epoch | Train | Val | 비고 |
|-------|-------|-------|-----|------|
| P1 | 09/15 | 47.43% | 53.31% | P1 best |
| P2 | 09/30 | 52.01% | 54.93% | P2 best |
| P2 | 17/30 | 53.54% | 53.93% | Early stop |
| P3 | 10/20 | 57.62% | **56.68%** | **전체 best** |
| P3 | 15/20 | 57.69% | 56.43% | Early stop |

최종: Train **64.42%** / Val **56.68%** / Test **54.17%**

> v02 test 55.37%는 test set을 validation으로 쓴 값(데이터 누출).  

> v03 test 54.17%는 진짜 unseen 평가. 공정한 비교 기준.

<br><br>

### 분류 리포트 (Test)

| 시즌 | Precision | Recall | F1 |
|------|-----------|--------|----|
| 봄 웜톤 | 0.48 | 0.41 | 0.44 |
| 여름 쿨톤 | 0.52 | 0.42 | 0.46 |
| 가을 웜톤 | 0.51 | 0.57 | 0.54 |
| 겨울 쿨톤 | 0.62 | 0.70 | 0.66 |

<br><br>

### 문제점

- P2에서 backbone LR 1e-5가 너무 낮아 17에포크 동안 train 50%→53% (backbone 거의 안 움직임)

- P3 patience=5가 너무 짧아 아직 개선 중에 조기 종료

- 봄/여름 Recall 41~42%로 저조

<br><br><br>  

--- 

<br>

## v04 — Backbone LR 조정 + Focal Loss + SWA

### 설계 변경

- backbone LR P2: 1e-5 → **2e-5** (v03 P2 정체 해소)
- backbone LR P3: 5e-6 → **1e-5**
- P3 max 30 epochs, patience=8 (v03 20/5에서 확장)
- **CosineAnnealingWarmRestarts(T₀=10)** — LR 사이클링으로 plateau 탈출
- **Focal Loss(γ=2)** in Phase 3 — 봄/여름 어려운 샘플 자동 업가중
- **SWA(Stochastic Weight Averaging)** — P3 8에포크부터 가중치 평균화
- Val split 15% (v03 20%에서 축소 → 학습 데이터 3,407장으로 증가)
- 정규화 소폭 완화: dropout=0.55, wd=3e-4, Mixup α=0.3, ls=0.05

<br><br>

### 실험 결과

| Phase | Epoch | Train | Val | 비고 |
|-------|-------|-------|-----|------|
| P1 | 13/15 | 51.25% | 54.24% | P1 best |
| P2 | 30/35 ← **best** | 59.55% | **57.57%** | 전체 best |
| P3 | 01/30 | 63.02% | 55.74% | ← P3 진입 즉시 val 급락 |
| P3 | 11/30 | 65.48% | 55.91% | Early stop |

최종: Train **70.56%** / Val **57.57%** / Test **51.97%**

<br><br>

### 문제점 분석

| 문제 | 내용 |
|------|------|
| Focal Loss 역효과 | P3 진입 즉시 val 57.57%→55.74% 급락. 손실함수 전환 시 gradient 방향 변화로 학습 불안정 |
| SWA 불충분 | P3 early stop으로 snapshot 4개뿐, 성능 향상 없어 사용 안 됨 |
| Test < Val | test 51.97% vs val 57.57% — val set이 easier 샘플로 구성된 것으로 추정 |
| 과적합 심화 | v03 +10%p gap → v04 +18.59%p gap으로 확대 |

<br><br><br>  

--- 

<br>

## v05 — 계층형 CNN (Stage1: Warm/Cool + Stage2: 세부 시즌)

### 핵심 변경

**4클래스 단일 헤드 → 3개 헤드 계층 구조**

```
Stage 1: Warm/Cool (이진 분류)
    ↓
Stage 2-W: Spring vs Autumn (Warm 라우팅 시)
Stage 2-C: Summer vs Winter (Cool 라우팅 시)

손실함수: L_total = L_stage1 + 0.7 × (L_stage2w + L_stage2c)
```

- Color Branch: 2레이어 BatchNorm → 3레이어 LayerNorm (소규모 데이터 안정성)

- 색상 피처: 70차원 → **80차원** (ITA, Warmth Index 등 10개 추가)

- backbone 동결: P1~P2 완전 동결, P3에서 blocks 6+만 LR=5e-6으로 미세 해제

- 클래스 가중치: 봄 ×1.3, 여름 ×1.2 (Recall 저조 클래스 업가중)

<br><br><br>  

--- 

<br>

## v06 

v05와 v07 사이의 중간 test로 별도 기록 x

<br><br><br>  

--- 

<br>

## v07 — 이미지 CNN에서 색상 수치 MLP로 전환 (핵심 전환점)

### 설계 철학 전환

v01~v05는 얼굴 이미지를 CNN으로 직접 분류했다. v07은 방향 전환

<br>

**전환 이유**

외국인 얼굴 이미지 데이터로 학습했을 때 test 데이터도 외국인임에도 정확도가 낮았다. 

이미지 기반 CNN의 구조적 문제는 다음과 같다:

- 조명, 카메라 화이트밸런스, 피부 보정 처리가 픽셀 분포를 바꿈
- 메이크업, 의상 색상, 배경이 피부색 신호를 덮음
- 4,000장 데이터에 21M 파라미터 → 얼굴 형태·질감 암기로 이어짐

<br>

**새로운 파이프라인**
```
얼굴 이미지 → 피부 영역 추출 → 조명 보정 → 대표 피부색 RGB
    → LAB / HSV / Chroma 변환 → MLP 분류
```

CNN이 아닌 MLP를 쓰는 이유: 입력이 18차원 수치 피처이므로 공간적 패턴 추출이 필요 없다. 

작은 MLP가 과적합 없이 학습할 수 있다.

<br><br>

### 데이터

`personal_color_palette_full.csv` — 전문가 라벨링 팔레트 데이터  

Spring 60 / Winter 59 / Autumn 45 / Summer 42 / 총 206행

<br><br>

### 피처 엔지니어링 (18개)

원본 컬럼(L, a, b, C, H, S, V, hex) 외에 파생 피처 추가:

<br>

| 피처 | 계산식 | 추가 근거 |
|------|--------|-----------|
| H_sin, H_cos | sin(H_rad), cos(H_rad) | Hue 순환성 보정 (359°와 1°는 유사한 색) |
| R, G, B | hex 변환 | 원본 채널 직접 노출 |
| C_div_L | C / L | 밝기 대비 채도 (봄=밝고선명, 가을=어둡고채도낮음) |
| S_div_V | S / V | 밝기 대비 채도 (여름vs겨울 구분) |
| a_div_b | a / b | 빨강–노랑 비율 (warm/cool 지표) |
| warm_yellow_score | b − \|a\| | 노란 기운 위주 웜톤 점수 |
| red_yellow_sum | a + b | 빨강·노랑 총량 |
| clarity_score | C + S | 선명도·채도 합산 (봄vs가을) |
| darkness_score | 100 − L | 깊이감 (가을·겨울 특징) |

<br><br>

### 계층형 분류 설계

Stage 1 피처 집중: a, b, H_sin, H_cos, a_div_b, warm_yellow_score  

Stage 2-W 피처 집중: L, C, S, V, C_div_L, clarity_score, darkness_score  

Stage 2-C 피처 집중: C, S, V, L, C_div_L, clarity_score

<br><br>

### 실험 결과

5-fold CV (4-class 기준):

| Fold | Accuracy | Macro F1 |
|------|----------|----------|
| 1 | 59.52% | 0.5967 |
| 2 | 63.41% | 0.6411 |
| 3 | 65.85% | 0.6655 |
| 4 | 65.85% | 0.6531 |
| 5 | 70.73% | 0.7104 |
| **평균** | **65.07%** | **0.6534** |

<br>

단일 split 결과 (Train 164 / Test 42):

| 모델 | Accuracy | Macro F1 |
|------|----------|----------|
| 4-class (직접 분류) | 55.0% | 0.543 |
| Warm/Cool | 85.7% | 0.857 |
| Spring/Autumn | 81.0% | 0.798 |
| Summer/Winter | 71.4% | 0.714 |

**핵심 발견:** Warm/Cool 이진 분류는 85.7%로 매우 높은 반면, 4계절 직접 분류는 55%로 낮다. 

계층형 구조가 각 단계를 더 쉬운 문제로 분해해 성능을 높일 수 있다.

<br><br>

### 한계

각 모델이 내부에서 독립적으로 split을 생성 → 서로 다른 test set으로 평가 → 파이프라인 end-to-end 정확도 측정 불가

<br><br><br>  

--- 

<br>

## v08 — CSV 자동 탐색 + 실행 편의성 개선

### 변경 사항

v07에서 실행 방법을 개선했다. 
```bash
# v07: --csv 필수 인자
uv run python train_personal_color.py --csv ../personal_color_palette_full.csv --epochs 800
```

<br>

v08에서 CSV 자동 탐색을 구현 :
```bash
# v08: 인자 없이 실행 가능
uv run train_personal_color.py --epochs 800 --cv
```

**CSV 탐색 순서:**
1. 스크립트와 같은 폴더 (`v08/personal_color_palette_full.csv`)
2. 상위 폴더 (`../personal_color_palette_full.csv`)
3. 현재 작업 디렉터리 (`./personal_color_palette_full.csv`)

<br>

**그 외 변경:**

- `pyproject.toml` 추가 → `uv run` 이 v08 폴더 내에서 독립 작동

- `metrics.json`에 사용된 CSV 절대 경로 기록

<br><br>

### 실험 결과

데이터: Spring 60 / Winter 59 / Autumn 45 / Summer 42 (206행) / Train 164 / Test 42

| 모델 | Accuracy | Macro F1 |
|------|----------|----------|
| Four-Class | 0.55 | 0.55 |
| Warm/Cool | 0.83 | 0.83 |
| Spring/Autumn | 0.86 | 0.85 |
| Summer/Winter | 0.76 | 0.76 |

> **한계:** 각 모델이 독립적으로 split을 생성해 서로 다른 test set 사용 → 파이프라인 end-to-end 정확도 측정 불가 → v09에서 해결

<br><br><br>  

--- 

<br>

## v09 — 공유 Split + 파이프라인 End-to-End 평가

### 핵심 문제 의식

v08까지 측정 방법의 구조적 문제가 있었다:

```
v08 방식:
  Four-Class → 자체 StratifiedShuffleSplit → test_A 평가 (acc=55%)
  Warm/Cool  → 자체 StratifiedShuffleSplit → test_B 평가 (acc=83%)
  Spring/Autumn → 자체 split → test_C 평가 (acc=86%)

문제: test_A, test_B, test_C가 모두 다른 샘플 → 파이프라인 비교 불가
```

Stage 1에서 발생한 오분류가 Stage 2로 전파되기 때문에 각 단계를 개별로 평가하면 실제 전체 성능과 다르다.

<br><br>

### 핵심 변경

**1. 공유 split 도입**

모든 모델이 동일한 StratifiedShuffleSplit을 사용 :

```
v09: 하나의 split 생성 → train_164 / test_42
     Four-Class, Stage1, Stage2a, Stage2b 모두 동일한 test_42로 평가
```

<br>

**2. `ModelBundle` 도입**

```python
@dataclass
class ModelBundle:
    model: nn.Module
    scaler: StandardScaler
    class_names: List[str]
    feature_cols: List[str]
    best_val_f1: float
    history: Dict
```

모델·스케일러·메타 정보를 묶어 파이프라인 평가 함수에 전달한다.

<br>

**3. `evaluate_hierarchical_pipeline()` 함수 추가**

전체 test set을 2단계 파이프라인에 직접 통과시켜 최종 4계절 예측 생성:
```
Test sample
    → Stage 1 Warm/Cool 예측
    → Warm이면 Stage 2a (Spring/Autumn)
    → Cool이면 Stage 2b (Summer/Winter)
    → 최종 계절 결정
```

Stage 1 오분류로 잘못 라우팅된 샘플이 Stage 2에서도 틀리는 효과까지 포함한 실제 성능이다. 

<br>

**4. 함수 구조 개편**

| v08 | v09 |
|-----|-----|
| `train_four_class()` | `train_four_class_model()` |
| `train_binary_head()` | `train_temp_model()` + `train_branch_model()` |
| (없음) | `_build_and_save_bundle()` |
| (없음) | `evaluate_hierarchical_pipeline()` |

<br><br>

### 실험 결과

데이터: Spring 60 / Winter 59 / Autumn 45 / Summer 42 (206행)  

Train: 164 / Test: 42 (모든 모델 동일 split)

<br>

**5-Fold Cross Validation (Four-Class)**

| Fold | Accuracy | Macro F1 |
|------|----------|----------|
| 1 | 0.5952 | 0.5967 |
| 2 | 0.6341 | 0.6411 |
| 3 | 0.6585 | 0.6655 |
| 4 | 0.6585 | 0.6531 |
| 5 | 0.7073 | 0.7104 |
| **평균** | **0.6508** | **0.6533** |

<br>

**개별 모델 성능 (동일 test set 기준)**

| 모델 | Accuracy | Macro F1 |
|------|----------|----------|
| Four-Class | 0.5476 | 0.5432 |
| Stage 1 Warm/Cool | 0.7381 | 0.7368 |
| Stage 2a Spring/Autumn | 0.8095 | 0.7981 |
| Stage 2b Summer/Winter | 0.8095 | 0.8091 |

<br>

**파이프라인 End-to-End (v09 신규 측정)**

- Warm branch 라우팅: 18개 / Cool branch 라우팅: 24개

```
              precision    recall  f1-score   support

      Spring       0.45      0.42      0.43        12
      Summer       0.53      0.89      0.67         9
      Autumn       0.71      0.56      0.62         9
      Winter       0.67      0.50      0.57        12

    accuracy                           0.57        42
   macro avg       0.59      0.59      0.57        42
```

<br>

| 모델 | Accuracy | Macro F1 | 비고 |
|------|----------|----------|------|
| Four-Class | 0.5476 | 0.5432 | 직접 분류 |
| **Pipeline** | **0.5714** | **0.5745** | **end-to-end** |

<br>

**파이프라인 오류 전파 분석:**
- 단순 계산: `0.74 × (0.81+0.81)/2 ≈ 0.60`
- 실제 end-to-end: **0.57**
- 3%p 차이 = Stage 1 오분류 샘플이 Stage 2에서 틀린 계절을 예측하는 효과

<br>

Spring 오류 7/12의 원인 역추적:
- Summer(3개): Stage 1이 Warm→Cool로 오라우팅 → Stage 2b에서 Summer
- Winter(3개): Stage 1이 Warm→Cool로 오라우팅 → Stage 2b에서 Winter
- Autumn(1개): Stage 1은 맞았지만 Stage 2a 오분류

**→ Stage 1이 파이프라인의 핵심 병목**

<br><br><br>  

--- 

<br>

## v10 — 피처 강화 + MLP 확장 시도

### v09 문제점 분석을 바탕으로 한 개선 시도

v09 분석에서 Stage 1이 핵심 병목이고 Spring/Summer 구분 피처가 부족하다는 것을 확인했다.

<br><br>

### 변경 사항

**1. 피처 5개 추가 (18 → 23개)**

| 피처 | 계산식 | 근거 |
|------|--------|------|
| `ITA` | `arctan((L−50)/b)×(180/π)` | 피부과학 warm/cool 지표. Spring(+30~45°) vs Summer(+20~35°) |
| `yellow_red_ratio` | `b / (\|a\|+ε)` | 노랑 대 빨강 비율 |
| `H_sin2` | `sin(2H)` | Hue 2차 조화 성분 |
| `H_cos2` | `cos(2H)` | Hue 2차 조화 성분 |
| `warm_strength` | `(b+0.5×a)/(L+ε)` | 밝기 정규화 복합 웜 신호 |

<br>

**2. MLP 구조 강화**

| 항목 | v09 | v10 |
|------|-----|-----|
| Hidden dim | 64 | 128 |
| 레이어 수 | 2 | 3 (128→128→64→classes) |
| 활성화 | ReLU | GELU |
| Dropout | 0.25 | 0.30 (마지막: 0.15) |

<br>

**3. 학습 정규화 추가**
- Feature Noise: 학습 시 Gaussian noise(std=0.03) 주입
- Label Smoothing: 0.10

<br>

**4. Stage 1 학습 강화**
- Stage 1 epochs: 400 → **800** (four-class와 동일)
- Stage 1 patience: 60 → **100**
- Branch epochs: 400 → **533**

<br><br>

### 실험 결과

| 지표 | v09 | v10 | 차이 |
|------|-----|-----|------|
| CV mean acc | **0.6508** | 0.6215 | **-0.029** |
| Four-Class acc | **0.5476** | 0.5238 | -0.024 |
| Stage 1 acc | **0.7381** | 0.6905 | **-0.048** |
| Pipeline acc | **0.5714** | 0.5238 | **-0.048** |

**모든 지표에서 v09가 v10보다 우수하다.**

<br><br>

### v10이 오히려 나빠진 원인 분석

| 원인 | 설명 |
|------|------|
| Feature noise 역효과 | 표준화된 피처(std=1)에 noise std=0.03은 3% 상대 노이즈. 206개 소규모 데이터에서 신호 자체가 약한데 noise가 덮어버림 |
| Label smoothing 역효과 | Summer는 이미 recall=0.89로 잘 분리됨. 잘 분리된 클래스의 confidence를 억제해 오히려 성능 낮춤 |
| 더 큰 모델이 과적합 | 128-dim 3-layer ≈ 22,000 params vs 학습 데이터 164개. 정규화 추가에도 일반화 나빠짐 |
| 새 피처 5개가 noise | ITA 등 이론적으로 유효하지만, 기존 18개 피처와 상관관계 높아 중복 정보로 작용 |

<br><br>

### 결론

**소규모 데이터(206개)에서는 단순한 모델 + 적절한 정규화가 복잡한 모델보다 낫다.**

→ 서비스 및 배포에는 **v09 모델 사용 권장**

<br><br><br>  

--- 

<br>

## 전체 버전 성능 요약

### 이미지 기반 CNN (v01~v05)

| 버전 | 백본 | Params | Test Acc | 특이사항 |
|------|------|--------|----------|---------|
| v01 | EfficientNetB0 | 4.75M | 53.29% | 베이스라인 |
| v02 | EfficientNetV2-S | 21M | 55.37%† | test=validation 누출 |
| v03 | EfficientNet-B3 | 12M | 54.17% | train/val/test 올바른 분리 |
| v04 | EfficientNet-B3 | 12M | 51.97% | Focal Loss 역효과 |

†v02의 55.37%는 test set을 validation으로 사용한 수치로 공정한 비교 불가.

<br><br>

### 색상 수치 기반 MLP (v07~v10)

| 버전 | CV mean acc | Pipeline acc | 특이사항 |
|------|-------------|--------------|---------|
| v07 | 65.07% | 측정 불가 | 독립 split 사용 |
| v08 | 65.07% (동일) | 측정 불가 | CSV 자동 탐색 추가 |
| **v09** | **65.08%** | **57.14%** | **공유 split + end-to-end 평가** |
| v10 | 62.15% | 52.38% | 피처·모델 강화 시도, 오히려 악화 |

<br>

**최종 : v09** — 파이프라인 acc 57.14%, CV mean acc 65.08%
