# 얼굴형 분류 모델 🧑‍💻

사진 한 장을 넣으면 얼굴형이 뭔지 맞춰주는 AI 모델.

Heart(하트형) / Oblong(긴형) / Oval(달걀형) / Round(둥근형) / Square(각진형) 총 5가지를 분류.

첫 정확도 36% → 최종 정확도 **90.9%**

<br>

---

<br>

## 📁 파일 구조

<br>

```
face-shape-classifier/
├── train_v19.py        ← 최종 학습 코드
├── infer.py            ← 이미지 넣으면 얼굴형 답해주는 추론 코드
├── requirements.txt    ← 필요한 패키지 목록
├── version.md           ← 버전별 train 기록 (상세)
└── v19/
    ├── best_model.pth  ← 학습 중 가장 정확도 높았던 모델
    └── swa_model.pth   ← 최종 사용 모델 (정확도 90.9%)
```

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
SWA 모델 + TTA     : 90.9%  ← 최종
```

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
v1.9  V2-S + CutMix + SWA + TTA     →  91%   ← 최종 (90% 돌파)
```

<br>

---

<br>

## 🚀 실행 방법

### 설치

```bash
# uv 설치 (처음 한 번만)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 패키지 설치 (CUDA 12.1)
uv add numpy matplotlib pillow scikit-learn torch torchvision \
    --index https://download.pytorch.org/whl/cu121

# 패키지 설치 (CPU 환경)
uv add numpy matplotlib pillow scikit-learn torch torchvision \
    --index https://download.pytorch.org/whl/cpu
```

<br>

### 이미지 추론

```bash
# 기본 (swa_model 사용, 권장)
uv run infer.py --image photo.jpg

# TTA까지 적용하면 정확도 최대
uv run infer.py --image photo.jpg --tta

# best_model 사용
uv run infer.py --image photo.jpg --model best
```

<br>

### 학습 재현

```bash
cd v19
uv run train_v19.py
```

<br>

---

<br>

## 🛠 사용 기술

| 항목 | 내용 |
|------|------|
| 언어 | Python 3.11+ |
| 패키지 관리 | uv |
| 딥러닝 | PyTorch, EfficientNetV2-S |
| 랜드마크 | MediaPipe FaceLandmarker |
| 머신러닝 | scikit-learn (RandomForest, SVM, StackingClassifier) |
| 증강 | Mixup, CutMix, TTA, RandomPerspective |
| 최적화 | SWA, CosineAnnealingWarmRestarts, AdamW |


<br>

---

<br>

## 📌 클래스별 최종 성능 (v1.9 SWA + TTA)

| 얼굴형 | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Heart (하트형) | 0.933 | 0.905 | 0.919 |
| Oblong (긴형) | 0.935 | 0.935 | 0.935 |
| Oval (달걀형) | 0.907 | 0.825 | 0.864 |
| Round (둥근형) | 0.866 | 0.935 | 0.899 |
| Square (각진형) | 0.909 | 0.945 | 0.927 |
| **전체 평균** | **0.910** | **0.909** | **0.909** |

<br>

> Oval(달걀형)이 다른 클래스보다 조금 낮은데, 달걀형이 다른 얼굴형들의 중간쯤 생겨서 경계가 애매하기 때문이다.

<br>

## 📝 상세 train 기록


버전별 상세 train 과정, 코드 구조, 수치 비교는 **[version.md](./version.md)** 에 정리돼 있다.
