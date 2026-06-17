# Personal Color Classifier

사람 얼굴 이미지를 입력받아 **봄(Spring) / 여름(Summer) / 가을(Autumn) / 겨울(Winter)** 퍼스널 컬러 분류

<br>

---

<br>

## 프로젝트 개요

### 이미지 CNN이 아닌 색상 수치 기반 MLP

v01~v05는 얼굴 이미지를 EfficientNet CNN으로 직접 분류하는 방식을 사용했다. 

하지만 퍼스널 컬러 분류에는 구조적 한계가 있었다.

이미지 기반 접근의 핵심 문제는 CNN이 분류에 사용하는 신호 자체가 불안정하다는 점이다. 

조명 조건, 카메라 화이트밸런스, 피부 보정 처리, 메이크업, 의상·배경 색상이 모두 CNN의 입력 픽셀 분포를 바꾼다. 

퍼스널 컬러의 실제 기준은 피부 고유의 색온도(warm/cool)와 밝기·채도 조합인데 이미지에서는 이 신호가 노이즈에 묻힌다.

v07부터 방향을 전환했다. 

<br>

얼굴 이미지에서 대표 피부색(RGB)을 먼저 추출한 뒤 그 색상값을 

CIE L\*a\*b\*, HSV, Chroma 등 18개 피처로 변환해 MLP(Multi-Layer Perceptron)로 분류한다. 

이 방식은 이미지의 조명·배경·메이크업 노이즈를 색상 추출 단계에서 분리해낼 수 있고 

모델 자체는 "이 색상이 어느 시즌에 속하는가"라는 순수한 색채 문제에 집중한다.

<br>

---

<br>

## 시스템 아키텍처

### 계층형 2단계 파이프라인 (v07~v09)

4계절을 한 번에 분류하는 대신 퍼스널 컬러 이론에 맞춰 문제를 두 단계로 나눈다.

```
입력 색상 (RGB)
      │
      ▼
[색상 피처 변환] → 18개 수치 피처 (LAB, HSV, Chroma 등)
      │
      ▼
Stage 1: Warm / Cool 분류
  봄·가을 → Warm   /   여름·겨울 → Cool
      │
      ├── Warm → Stage 2a: Spring vs Autumn
      │
      └── Cool → Stage 2b: Summer vs Winter
```

**Stage 1**은 색온도(warm/cool) 이진 분류. 

퍼스널 컬러에서 warm/cool 구분은 LAB의 `b*`(노랑–파랑 축), `a*`(빨강–초록 축), Hue 각도로 잘 포착된다.

<br>

**Stage 2**는 밝기·채도 기반 이진 분류. 

Spring vs Autumn은 같은 웜톤이지만 봄은 밝고 선명하고 가을은 깊고 차분. 

Summer vs Winter는 같은 쿨톤이지만 여름은 부드럽고 겨울은 선명하고 대비가 강하다.

<br>

### 18개 색상 피처

| 카테고리 | 피처 | 의미 |
|----------|------|------|
| CIE L\*a\*b\* | L, a, b | 명도, 빨강–초록, 노랑–파랑 |
| 채도·밝기 | C (Chroma), S, V | 색의 선명도, HSV 채도·밝기 |
| Hue 인코딩 | H_sin, H_cos | 색상각 순환성 보정 |
| RGB | R, G, B (정규화) | 원본 채널 |
| 파생 피처 | C_div_L, S_div_V, a_div_b | 비율 기반 상대 지표 |
| 도메인 피처 | warm_yellow_score, red_yellow_sum, clarity_score, darkness_score | 퍼스널 컬러 이론 직결 |

<br>

### MLP 구조

```python
Linear(input_dim → 64) → BatchNorm1d → ReLU → Dropout(0.25)
Linear(64 → 64)        → BatchNorm1d → ReLU → Dropout(0.25)
Linear(64 → num_classes)
```

- 4-class 모델: `input_dim=18, num_classes=4`
- Stage 1: `input_dim=18, num_classes=2` (Warm/Cool)
- Stage 2a: `input_dim=18, num_classes=2` (Spring/Autumn)
- Stage 2b: `input_dim=18, num_classes=2` (Summer/Winter)

<br>


---

<br>

## 학습 데이터

`personal_color_palette_full.csv` — 전문가가 라벨링한 퍼스널 컬러 팔레트 색상표

<br>

| Season | Count |
|--------|-------|
| Spring | 60 |
| Winter | 59 |
| Autumn | 45 |
| Summer | 42 |
| **합계** | **206** |

- Train: 164개 / Test: 42개 (Stratified split, 모든 모델이 동일한 split 공유)

- 5-Fold Cross Validation으로 단일 split 분산 측정

<br>

---

<br>

## 결과 (v09 — 최종 모델)

### 5-Fold Cross Validation (Four-Class 기준)

| Fold | Accuracy | Macro F1 |
|------|----------|----------|
| 1 | 0.5952 | 0.5967 |
| 2 | 0.6341 | 0.6411 |
| 3 | 0.6585 | 0.6655 |
| 4 | 0.6585 | 0.6531 |
| 5 | 0.7073 | 0.7104 |
| **평균** | **0.6508** | **0.6533** |

<br>

### 계층형 파이프라인 End-to-End 결과

v09의 핵심은 **파이프라인 전체를 실제로 통과시켜 end-to-end 정확도를 측정**한 것이다. 

v08까지는 각 Stage를 독립적으로 평가했기 때문에 Stage 1 오분류가 Stage 2로 전파되는 효과를 반영한 실제 성능을 알 수 없었다.

| 모델 | Accuracy | Macro F1 | 비고 |
|------|----------|----------|------|
| Four-Class 직접 분류 | 0.5476 | 0.5432 | 4계절 한 번에 |
| **Hierarchical Pipeline** | **0.5714** | **0.5745** | **end-to-end** |
| Stage 1 Warm/Cool | 0.7381 | 0.7368 | 개별 stage |
| Stage 2a Spring/Autumn | 0.8095 | 0.7981 | 개별 stage |
| Stage 2b Summer/Winter | 0.8095 | 0.8091 | 개별 stage |

계층형 파이프라인(57.14%)이 직접 4계절 분류(54.76%)보다 **+2.4%p** 높다.

각 stage 개별 정확도(74~81%)는 높지만 Stage 1 오분류가 전파되어 파이프라인 전체는 57%입니다. 

단순 곱 계산(`0.74 × 0.81 ≈ 0.60`)과 달리 실제 end-to-end는 57%로 오류 전파 효과를 직접 확인할 수 있었다.

<br>

### 파이프라인 Confusion Matrix

|        | Spring | Summer | Autumn | Winter |
|--------|--------|--------|--------|--------|
| Spring | **5**  | 3      | 1      | 3      |
| Summer | 1      | **8**  | 0      | 0      |
| Autumn | 3      | 1      | **5**  | 0      |
| Winter | 2      | 3      | 1      | **6**  |

**핵심 패턴:** Spring 오류의 대부분(6/7)은 Stage 1이 Warm을 Cool로 오라우팅한 결과

<br>

---

<br>

## 사용법

### 1단계: 모델 학습

```bash
cd model
python train_personal_color.py --epochs 800 --cv
```

학습 완료 후 `outputs/` 폴더에 6개 파일이 생성:

```
outputs/
├── temperature_spring_summer_autumn_winter.pt       ← Stage 1 (Warm/Cool)
├── temperature_spring_summer_autumn_winter_scaler.joblib
├── season_spring_autumn.pt                          ← Stage 2a (Spring/Autumn)
├── season_spring_autumn_scaler.joblib
├── season_summer_winter.pt                          ← Stage 2b (Summer/Winter)
└── season_summer_winter_scaler.joblib
```

<br>

### 2단계: 퍼스널 컬러 예측

```bash
# 이미지로 예측 (대표 색 자동 추출)
python infer.py --image photo.jpg

# HEX 코드로 예측
python infer.py --hex d3af8c

# RGB 값으로 예측
python infer.py --rgb 210 175 140

# 모델 폴더 위치 지정
python infer.py --image photo.jpg --model-dir outputs
```

<br>

### 출력 예시

```
[Image] photo.jpg
  Extracted color: RGB(210, 175, 140)  #d2af8c

==================================================
  퍼스널 컬러 결과: Spring (봄)
==================================================
  밝고 따뜻한 톤 — 노란 기운의 선명한 색상이 어울립니다.

  입력 색상   : RGB(210, 175, 140)  #d2af8c
               L=73.2  a=8.4  b*=18.6

  Stage 1  Warm/Cool → Warm
    Cool   0.123  ██
    Warm   0.877  █████████████████

  Stage 2  계절 분류 → Spring
    Autumn 0.214  ████
    Spring 0.786  ███████████████
==================================================
```

<br>

---

<br>

## 파일 구조

```
.
├── README.md                          ← 현재 파일
├── version.md                         ← v01~v10 상세 버전 히스토리
├── personal_color_palette_full.csv    ← 퍼스널 컬러 팔레트 데이터
└── model/
    ├── train_personal_color.py        ← 학습 스크립트 (v09)
    ├── infer.py                       ← 예측 스크립트
    ├── photo.jpg                      ← 샘플 이미지
    └── outputs/                       ← 학습 후 자동 생성
        ├── temperature_spring_summer_autumn_winter.pt
        ├── temperature_spring_summer_autumn_winter_scaler.joblib
        ├── season_spring_autumn.pt
        ├── season_spring_autumn_scaler.joblib
        ├── season_summer_winter.pt
        └── season_summer_winter_scaler.joblib
```

<br>

---

<br>

## history 요약

### Phase 1 — 이미지 기반 CNN (v01~v05)

EfficientNet 백본에 피부색 피처를 결합하는 방식으로 시작

v01의 EfficientNetB0(4.75M)에서 시작해 v02의 EfficientNetV2-S(21M), v03의 EfficientNet-B3(12M)으로 test

ColorJitter 제거, 배경을 검정에서 중립 회색으로 변경, train/val/test 분리, early stopping 등을 도입했지만 

test 정확도는 51~55% 수준에서 정체되어있었다. 

과적합과 이미지 노이즈(조명, 배경, 메이크업)가 근본 원인이었다.

<br>

### Phase 2 — 색상 수치 기반 MLP로 전환 (v07~v10)

v07에서 접근 방식을 근본적으로 바꿨다. 

이미지 전체를 CNN으로 처리하는 대신 색상값(hex, LAB, HSV)을 MLP로 분류하는 방식이다. 

이 전환으로 이미지 노이즈가 분리되고 데이터도 훨씬 효율적으로 활용할 수 있게 되었다. 

v08에서 실행 편의성을 개선했고 v09에서 end-to-end 파이프라인 평가를 도입해 실제 성능을 정확히 측정했다. 

v10은 피처 추가와 모델 강화를 시도했지만 206개 소규모 데이터에서는 단순한 v09가 오히려 더 좋은 결과를 냈다.

<br>

**최종 모델: v09** — 소규모 데이터에서 단순 모델 + 적절한 정규화가 최적.

자세한 버전별 개선 내용은 [version.md](version.md)를 참조.

<br>

---

<br>

## 의존성

```
torch
numpy
scikit-learn
joblib
Pillow  (이미지 입력 시)
```

```bash
pip install torch numpy scikit-learn joblib Pillow
```
