# 얼굴형 분류 모델 🧑‍💻

사진 한 장을 넣으면 얼굴형이 뭔지 맞춰주는 AI 모델.

Heart(하트형) / Oblong(긴형) / Oval(달걀형) / Round(둥근형) / Square(각진형) 총 5가지를 분류.

첫 정확도 36% → 최종 정확도 **90.45%** (여성) / **76.92%** (남성)

<br>

---

<br>

## 📁 파일 구조

<br>

```
face/
├── female/                   ← 여성 학습 코드 & 모델 (v22)
│   ├── train.py              ← 여성 학습 스크립트
│   ├── swa_model.pth         ← 추론 및 남성 전이학습 출발점 (필수)
│   ├── geo_scaler.pkl        ← 기하학 특징 스케일러 — 남성 학습이 이 파일을 직접 참조 (필수)
│   ├── class_geo_medians.pkl ← 클래스별 기하학 중앙값 — 추론 설명문 생성용 (선택)
│   └── training_curve.png    ← 학습 곡선
│
├── male/                     ← 남성 전이학습 코드 & 모델 (v22)
│   ├── train.py              ← 남성 전이학습 스크립트 (HeadOnly + exclude_list)
│   ├── best_model.pth        ← Best 모델 (76.92%)
│   ├── confusion_matrix.png  ← 혼동행렬
│   ├── training_curve.png    ← 학습 곡선
│   ├── train_summary.json    ← 학습 요약 (하이퍼파라미터, 정확도)
│   ├── geo_scaler.pkl        ← 기하학 특징 스케일러 (필수)
│   ├── class_geo_medians.pkl ← 클래스별 기하학 중앙값 — 추론 설명문 생성용 (선택)
│   └── exclude_list.txt      ← 학습에서 제외한 이미지 목록 (23장)
│
├── common.py                 ← 공통 모듈 (모델 구조, 전처리, 학습 함수)
├── infer.py                  ← 추론 스크립트 (--gender female/male)
├── image/                    ← 추론 테스트용 이미지 폴더
│   └── photo.jpg
│
├── pyproject.toml            ← 의존성 (uv, CUDA 12.1)
├── requirements.txt          ← 패키지 목록 (pip 대안용)
├── version.md                ← 버전별 train 기록 (상세)
└── README.md
```

<br>

---

<br>

## 📂 데이터셋 구조

<br>

```
dataset/
├── training_set/         ← 여성 학습 데이터
│   ├── Heart/
│   ├── Oblong/
│   ├── Oval/
│   ├── Round/
│   └── Square/
├── testing_set/          ← 여성 테스트 데이터
└── men/
    ├── training_set/     ← 남성 학습 데이터
    │   ├── ovale/          → Oval 매핑
    │   ├── rectangular/    → Oblong 매핑
    │   ├── round/
    │   └── square/
    └── testing_set/      ← 남성 테스트 데이터
```

> Heart 클래스는 남성 데이터 없음 → 여성 모델 가중치 유지

<br>

---

<br>

## VERSION 정리 

<br>

### 1단계 — 픽셀 활용 (v0.1, 36%)

처음엔 이미지 픽셀을 넣어봤다.

HOG라는 방식으로 이미지에서 1764개 숫자를 뽑아서 RandomForest로 학습했는데 결과가 **36%** 나왔다.

5개 중에 랜덤으로 찍어도 20%가 나오니까 사실상 별로 못 맞추는 결과다.

픽셀은 조명이 밝은지 어두운지, 배경이 뭔지 같은 것도 다 학습해버려서 얼굴 *모양* 자체를 제대로 못 배웠던 것이다.

<br>

### 2단계 — 얼굴 점 478개로 비율 계산으로 변경(v0.2 ~ v0.3, 44 ~ 46%)

MediaPipe로 얼굴에 점 478개를 찍고, 그 점들 사이 거리로 "얼굴 길이 / 광대 너비" 같은 비율을 계산했더니 **44%** 로 올랐다.

여기서 추가로 얼굴 부분만 잘라내서(크롭) 다시 점을 찍는 **2-pass** 방식을 적용했더니 **46%** 까지 올랐다.

배경이 끼어들지 않으니까 더 정확하게 점을 찍게 된 것이다. 

<br>

### 3단계 — 턱 각도 추가 후 10% 증가 (v0.4, 56%)

비율만으로는 한계가 있었다. heart형·oval형·round형·oblong형은 광대 돌출도 수치가 거의 똑같아서 구분이 안 됐다.

그래서 **턱이 얼마나 각졌는지(jaw_angle)** 를 추가로 계산했다.

각진형은 턱 각도가 약 143°, 나머지 얼굴형은 148~151° 정도인데, 이 7° 차이로 훨씬 잘 구분하게 됐다.

결과: **46% → 56%** (+10%p). 단일 개선 중 가장 큰 수치였다.

<br>

### 4단계 — 데이터 증가 (v0.9, 65%)

그동안 클래스당 100장(총 500장)으로 학습하고 있었는데, 클래스당 800장(총 4000장)으로 늘렸다.

코드를 거의 안 바꾸지 않고 데이터만 늘렸는데 **65%** 까지 올랐다.

AI 학습은 결국 데이터 양이 제일 중요하다는 걸 수치로 확인했다.

<br>

### 5단계 — 비율 수치 + 픽셀 이미지 활용 후 정확도 증가 (v1.2, 69%) 

비율 숫자 15개 + 크롭한 얼굴 이미지를 PCA로 압축한 픽셀 50개를 합쳐서 **65차원** 복합 특징으로 학습했다.

숫자만으로 못 잡는 얼굴 윤곽 패턴을 이미지가 보완해줬다.

결과: **69.1%** — ML 기반 모델 중 최고 정확도였다.

<br>

### 6단계 — EfficientNet으로 변경 (v1.3 ~ v1.7, 71 ~ 82%)

지금까지는 숫자(비율)로 학습하는 ML이었는데 이미지 자체를 딥러닝으로 학습하는 방식으로 전환했다.

ImageNet으로 이미 학습된 **EfficientNet** 모델을 가져와서 우리 데이터로 추가 학습(파인튜닝)했다.

EfficientNet-B4로 파인튜닝하니까 **82.7%** 까지 올랐다. ML 최고 기록(69%)에서 단번에 13%p 올랐다. 

<br>

### 7단계 — 해상도 높이고 2단계 학습 (v1.8, 87.4%)

EfficientNet-B4의 원래 입력 크기는 380×380인데, 그동안 224×224로 줄여서 넣고 있었다.

원래 크기로 맞추고, 학습을 두 단계로 나눴다.

- **Phase 1**: 뒷부분(head)만 학습 → 기존 특징 보존
- **Phase 2**: 전체 학습 → 얼굴형에 맞게 세밀하게 조정

여기에 Mixup 증강(두 이미지를 섞어서 새 학습 데이터 만들기)도 추가했더니 **87.4%** 까지 올랐다.

<br>

### 8단계 — 백본 바꾸고 가중치 평균 내고 5번 예측 평균 : 정확도 90%  (v1.9, 90.9%)

세 가지를 한꺼번에 바꿨다.

**① EfficientNetV2-S로 교체** — B4보다 최신 모델이고 ImageNet 정확도도 더 높다. 입력 크기 384×384가 딱 맞는다.

**② CutMix + Mixup 혼합 증강** — Mixup은 두 이미지를 반투명하게 겹치는 방식

CutMix는 한 이미지 일부를 잘라서 다른 이미지에 붙이는 방식이다. 둘을 50:50으로 랜덤하게 사용했다.

**③ SWA (가중치 평균)** — 학습이 수렴하면 그 이후 20번의 모델 가중치를 전부 평균 냈다. 

한 번 잘한 모델보다 평균 모델이 더 안정적으로 잘 맞춘다.

**④ TTA (5번 예측 평균)** — 이미지 한 장을 5가지 방식으로 변형해서 각각 예측하고 평균 냈다.

<br>

결과:

```
Best 모델 단독     : 89.8%
Best 모델 + TTA    : 90.3%
SWA 모델 + TTA     : 90.9%
```

<br>

### 9단계 — v22 하이브리드 아키텍처: 기하학 + CNN 융합 (90.45%)

v1.9까지는 EfficientNetV2-S CNN만 단독으로 사용했다.

v22에서는 MediaPipe로 추출한 기하학 특징 17개를 CNN과 하나의 네트워크 안에서 융합하는 하이브리드 구조로 바꿨다.

**① GeoEncoder** — R1~R4(얼굴 비율), 삼정비율, 하악각, 턱끝각도 등 17개 기하학 수치를 64차원 임베딩으로 변환한 뒤 CNN 임베딩(1280차원)과 이어붙인다. CNN이 못 잡는 "왜 이 얼굴형인지"를 수치로 설명할 수 있게 된다.

**② HeartGate** — Heart형(이마 넓고 턱 좁은 역삼각형)은 머리카락이 이마를 가리면 CNN이 Heart와 Oval을 잘 구분하지 못한다. 기하학 특징(`R2`, `chin_angle`, `face_taper_r`)으로 Heart 신호 강도를 0~1로 추정해서 분류기 입력에 추가한다. 모델이 스스로 "게이트 값이 높으면 기하학 신호를 더 믿는다"는 패턴을 학습한다.

**③ 3단계 앞머리 분류** — 기존에는 `upper_r < 0.15`이면 무조건 제외했다. v22에서는 완전 가림(`upper_r < 0.08`)만 제외하고, 부분 가림은 표본 가중치 0.5로 포함한다. 학습 데이터를 덜 버리면서 신뢰도를 반영한다.

<br>

결과: **90.45%** (여성 Best 모델 기준)

<br>

### 10단계 — 남성 HeadOnly 전이학습 + exclude_list (76.92%)

남성 얼굴형 분류 모델은 v22 여성 모델 가중치를 전이학습하는 방식으로 진행했다.

**기존 방식과의 차이:**

| 항목 | 기존 (features[5-7] 파인튜닝) | v22 HeadOnly |
|---|---|---|
| 학습 대상 | `features[5~7]` + `classifier` | `classifier` + `geo_enc` + `heart_gate` |
| backbone | `features[0~4]` 동결 | 전체 동결 |
| 결과 | 과적합 경향 | 안정적 수렴 |

HeadOnly 방식이 더 안정적인 이유는 남성 데이터 수가 적어서 backbone까지 파인튜닝하면 과적합이 빠르게 일어나기 때문이다.

<br>

**오분류 분석 + exclude_list:**

HeadOnly 전이학습으로 약 74% 수준에 도달한 뒤, 오분류 이미지를 분석했다.

`Oval↔Oblong`, `Round↔Square` 사이의 혼동이 많았는데, 이는 남성 데이터에서 클래스 경계가 모호한 이미지가 섞여 있기 때문이다.

confidence가 높은 오분류 샘플 중 라벨이 애매한 이미지 23장을 `exclude_list.txt`에 등록해서 제외한 뒤 재학습하니 **76.92%** 까지 올랐다.

<br>

최종 조건:

```
전이학습 방식 : 여성 v22 모델 기반 HeadOnly
학습 대상     : classifier + geo_enc + heart_gate (backbone 전체 동결)
mixup         : 0.1
learning rate : 0.0005
exclude 이미지: 23장
최고 정확도   : 76.92% (Epoch 11)

실행:
uv run python female/train.py      # 1단계 — 여성 모델 학습
uv run python male/train.py --no-cache --mixup 0.1 --lr 0.0005
```

현재 남성 데이터 기준에서는 약 76~77% 수준이 안정적으로 도달 가능한 성능이다. 추가 향상을 위해서는 코드 튜닝보다 `Oval↔Oblong`, `Round↔Square` 경계에 있는 라벨 재정의 등 데이터 품질 개선이 더 효과적이다.

**클래스 매핑:**

| 남성 폴더명 | 매핑 클래스 |
|---|---|
| rectangular | Oblong |
| ovale | Oval |
| round | Round |
| square | Square |

<br>

---

<br>

## 📊 전체 정확도 흐름
```
v0.1  픽셀(HOG)만 사용               →  36%
v0.2  얼굴 랜드마크 비율로 전환      →  44%
v0.3  얼굴 크롭 2-pass 적용          →  46%
v0.4  턱 각도 특징 추가              →  56%   ← +10%p 단일 최대
v0.9  데이터 8배 확장 (500→4000장)  →  65%   ← 데이터가 제일 중요
v1.2  비율 + 픽셀 PCA 복합           →  69%   ← ML 기반 최고
v1.3  EfficientNet 임베딩 도입       →  72%
v1.7  EfficientNet-B4 파인튜닝       →  83%
v1.8  380px + 2-Phase + Mixup        →  87%
v1.9  V2-S + CutMix + SWA + TTA     →  91%
v22   기하학+CNN 하이브리드(HeartGate) →  90% (여성 best_model 기준)
v22m  남성 HeadOnly + exclude_list   →  77% (남성 4클래스)
```

<br>

---

<br>

## 🚀 실행 방법

### 설치

```bash
# uv 사용 (권장 — CUDA 12.1 저장소 자동 적용)
cd face
uv sync
```

CPU 환경이라면 `pyproject.toml`의 `[tool.uv.sources]` 블록을 제거한 뒤 실행한다.

<br>

### 추론 (여성 / 남성)

```bash
# 여성 (기본값)
uv run python infer.py image/photo.jpg

# 남성
uv run python infer.py image/photo.jpg --gender male

# JSON 출력 (다른 프로그램 연동 시)
uv run python infer.py image/photo.jpg --gender male --json
```

<img src="https://github.com/user-attachments/assets/2b6d6692-84d1-45f0-88ad-f596f5572b57" width="45%" />

<br>

### 여성 모델 학습 재현

```bash
cd face
uv run python female/train.py
```

| 하이퍼파라미터 | 값 |
|---|---|
| 백본 | EfficientNetV2-S (384×384) |
| 배치 크기 | 8 (Gradient Accum × 4 = 실효 32) |
| Phase 1 | 10 epoch (head + geo_enc + heart_gate만 학습) |
| Phase 2 | EarlyStopping patience=15 (backbone 일부 unfreeze) |
| Phase 3 | SWA 20 epoch |
| 증강 | CutMix + Mixup |
| 기하학 특징 | MediaPipe 17개 → GeoEncoder(64차원) |

<br>

### 남성 전이학습

`female/swa_model.pth`를 초기 가중치로, `classifier + geo_enc + heart_gate`만 학습합니다(backbone 전체 동결).

```bash
# 1단계 — 여성 모델 학습 (여성 swa_model.pth 필요)
uv run python female/train.py

# 2단계 — 남성 HeadOnly 전이학습 (최적 조건)
uv run python male/train.py --no-cache --mixup 0.1 --lr 0.0005

# exclude_list 없이 전체 데이터 사용
uv run python male/train.py --no-cache --no-exclude

# 이어서 학습 (checkpoint_latest.pth 필요)
uv run python male/train.py --resume
```

오분류 이미지를 분석해서 `exclude_list.txt`에 추가하면 정확도를 더 높일 수 있다.

```bash
# 오분류 이미지 추출 → misclassified/ 폴더 생성
uv run python male/export_misclassified.py --clear

# 이미지 확인 후 exclude_list.txt에 파일명 작성 → 재학습
uv run python male/train.py --no-cache --mixup 0.1 --lr 0.0005
```

<br>

---

<br>

## 🛠 사용 기술

| 항목 | 내용 |
|------|------|
| 언어 | Python 3.11+ |
| 딥러닝 | PyTorch, EfficientNetV2-S (384×384) |
| 랜드마크 | MediaPipe FaceLandmarker (17개 기하학 특징) |
| 하이브리드 구조 | GeoEncoder (기하학 → 64차원) + HeartGate (Heart 신호 게이트) |
| 머신러닝 | scikit-learn (RandomForest, SVM, StackingClassifier) |
| 증강 | Mixup, CutMix |
| 최적화 | SWA, CosineAnnealingLR, AdamW |
| 배경 제거 | rembg (AI 세그멘테이션, GrabCut 폴백) |
| 남성 전이학습 | 여성 v22 가중치 → HeadOnly (backbone 동결, head만 학습) |
| exclude_list | 오분류 분석 후 라벨 모호 샘플 제외 재학습 |

<br>

---

<br>

## 📌 클래스별 최종 성능

### 여성 (v22 Best 모델, 90.45%)

| 얼굴형 | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| Heart (하트형) | 0.9412 | 0.9026 | 0.9215 | 195 |
| Oblong (긴형) | 0.9529 | 0.9479 | 0.9504 | 192 |
| Oval (달걀형) | 0.9066 | 0.8462 | 0.8753 | 195 |
| Round (둥근형) | 0.8186 | 0.9072 | 0.8606 | 194 |
| Square (각진형) | 0.9096 | 0.9144 | 0.9120 | 187 |
| **전체 평균** | **0.9058** | **0.9034** | **0.9038** | 963 |

<br>

> Oval(달걀형)이 다른 클래스보다 조금 낮다. 달걀형이 heart·oblong·round의 중간쯤 생겨서 경계가 애매하기 때문이다.

<br>

### 남성 (v22 Best 모델, 76.92%)

| 얼굴형 | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| Oblong (긴형) | 0.6726 | 0.7600 | 0.7136 | 100 |
| Oval (달걀형) | 0.8265 | 0.7864 | 0.8060 | 103 |
| Round (둥근형) | 0.8000 | 0.7273 | 0.7619 | 55 |
| Square (각진형) | 0.8000 | 0.7742 | 0.7869 | 93 |
| **전체 평균 (weighted)** | **0.7715** | **0.7664** | **0.7677** | 351 |

<br>

> Oblong 정밀도가 낮다. 남성 데이터에서 Oval↔Oblong 경계가 모호한 샘플이 많기 때문이며, 추가 향상을 위해서는 데이터 라벨 재정의가 필요하다.

<br>

## 📝 상세 train 기록


버전별 상세 train 과정, 코드 구조, 수치 비교는 **[version.md](./version.md)** 에 정리돼 있다.
