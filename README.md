# 재활용 품목 분류를 위한 Object Detection

## 1. 문제 정의 및 목표 설정

### 1.1 문제 정의

재활용 품목 분류는 수작업으로 이루어질 경우 시간이 많이 소요되고, 사람의 실수로 인해 분류 정확도가 떨어질 수 있다. 본 프로젝트는 객체 탐지 기술을 활용하여 재활용 품목(General trash, Paper, Plastic 등)을 자동으로 탐지하고 분류함으로써, 효율적이고 정확한 분리배출 시스템을 구축하는 것을 목표로 한다.

### 1.2 목표
- COCO 형식의 재활용 품목 데이터셋에서 10개 클래스(General trash, Paper 등)를 정확히 탐지하는 객체 탐지 모델 개발.
- 성능 지표로 mAP@0.5:0.95 ≥ 0.5 달성 및 IoU ≥ 0.7 목표.
- 초기에는 CO-DETR(Transformer 기반 모델)을 고려했으나, 학습 및 추론 시간이 오래 걸려 테스트가 어려운 점을 감안해, 경량화와 실시간 처리에 강점이 있는 YOLOv11n으로 진행 방향을 전환

### 1.2 데이터셋 선정

- **데이터셋**
    - COCO (Common Objects in Context)
- **클래스 개수**
• 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- **이미지 크기**
    - height: 1024
    - width: 1024
- **어노테이션 형식**
    - COCO JSON

## 2. [데이터 탐색(EDA)](https://github.com/Batwan01/Upstage_recycling_Object_Detection/blob/main/eda.ipynb)

### 2.1 데이터 구조 확인

- 이미지 및 라벨 데이터 구조 분석
- 클래스별 객체 수 시각화
- 바운딩 박스 크기 및 위치 분석

### 2.2 클래스 불균형 확인

- 클래스별 데이터 수 시각화 (히스토그램)

### 2.3 데이터 품질 검사

- 중복 이미지 및 잘못된 라벨링 확인
- 너무 작은 객체 및 잘린 객체 분석
- 바운딩 박스 위치 분포

### 2.4 데이터 정제

- 2.1 ~ 2.3를 참고하여 데이터 정제
  - [makesense.ai활용](https://www.makesense.ai)

    ![aa](https://github.com/user-attachments/assets/08129a09-e515-4d61-a426-25ee95ea7ca8)

- 담배꽁초는 라벨링이 되어 있지 않아 제외
  - 0001, 0038, 0051, 0069, 0071, 0120, 0147, 0160, 0164, 0186, 0189, 0208, 0752, 1557, 1670, 1941, 1954, 2381, 2431, 2434, 2491, 2555, 2561, 2610, 2622, 2656, 2675, 2965, 2978, 3024, 3044, 3078, 3088, 3100, 3126, 3131, 3161, 3170, 3476, 3492, 3511, 3531, 3786, 3821, 3874, 4097, 4139, 4233, 4242, 4246, 4280, 4358
  
    ![3476](https://github.com/user-attachments/assets/ab56b26e-cf75-4462-bf59-00f931bac627)

- 라벨 제거 및 추가
  - 0067, 0134, 1624, 0230, 0906, 0975, 2629, 2961, 2971, 3106, 3501, 3539, 3560, 3797, 3904, 4098, 4100, 4165, 4191, 4209, 4260, 4357, 4408

    <img src="https://github.com/user-attachments/assets/17049fdf-cc61-4425-a75a-a3712399fd54" width="45%" /> <img src="https://github.com/user-attachments/assets/b46c029b-e073-45b7-b289-5702267885df" width="45%" />

  
- 잘못된 라벨 정제
  - 0210, 1569, 2361, 2422, 2424, 2505, 3127, 3163, 4153, 4186, 4225, 4270
    
    <img src="https://github.com/user-attachments/assets/79656282-d7a5-4e70-857b-0b40ce347732" width="45%" /> <img src="https://github.com/user-attachments/assets/069f0f18-a262-4be3-8f31-8c3b6ec35fc0" width="45%" />
    
    <img src="https://github.com/user-attachments/assets/1b4ae7ac-0d01-4041-8f68-45e6128dace7" width="45%" /> <img src="https://github.com/user-attachments/assets/0969969a-c039-425f-aa4b-fd90aaca1a51" width="45%" />
  
## 3. 데이터 전처리 및 증강

### 3.1 데이터셋 분할

- **Train/Validation/check 비율**: 70/15/15 (check: class wise 용도)
- **Stratified Split 적용** (클래스별 샘플 개수 균형 고려)
- 
### 3.2 앵커 박스 튜닝 (Anchor Box Tuning)

- **목적**: 데이터셋의 객체 크기와 종횡비를 기반으로 YOLO 모델의 앵커 박스를 최적화.
- **방법**: 데이터셋의 바운딩 박스 분포를 분석하여 사전 정의된 앵커 박스를 튜닝.

#### 앵커 박스
```text
anchors:
  [75.08939902, 69.50222697, 162.38635929, 166.20860914, 346.43648184, 189.85621415] # P3/8
  [230.93618444, 347.09457937, 281.05258752, 579.09482496, 478.85510678, 372.36021356] # P4/16
  [749.27767768, 418.93633634, 516.2599182, 691.93343558, 874.50543478, 810.12217391] # P5/32
  [925, 792, 739, 380, 436, 615] # P6/64
```

### 3.3 데이터 증강(Augmentation)

- **기본 변환**: 크롭, 좌우 반전, 밝기 조절
- **객체 보존 증강**: CutMix, MixUp, Mosaic 적용
- **라이브러리 활용**: Albumentations
- **커리큘럼 러닝**: 점차 어려운 증강 적용 easy - 기본 학습률, Mosaic, 밝기 대비 조정
   학습 단계 요약

  - **Easy**: 기본 객체 인식 학습  
    - Epochs: 10, 이미지 크기: 640, 배치: 8  
    - 학습률: 0.002, Mosaic: 0.2, Flipud: 0.5, HSV 조정: 0.2  

  - **Medium**: 중간 복잡도 데이터로 일반화 강화  
    - Epochs: 10, 이미지 크기: 832, 배치: 8  
    - 학습률: 0.002, Mosaic: 0.3, Mixup: 0.2, Rotate: ±15°, 기타 증강 추가  

  - **Hard**: 복잡한 상황 대응 학습  
    - Epochs: 10, 이미지 크기: 1024, 배치: 8  
    - 학습률: 0.001, Mosaic: 0.5, Mixup: 0.3, Rotate: ±30°, 강한 증강 적용

## 4 모델 선정

### 4.1 모델 선정
<img src="https://github.com/user-attachments/assets/5ef51ff4-f8e5-4b36-a8ca-137b51bf184a" width="70%"/>

- **CO-DETR**: Swin-Large 백본으로 높은 정확도 제공, 다만 연산 비용이 높아 실시간 활용에는 제약
- **YOLOv11n**: 경량화된 구조로 실시간 처리(FPS 30 이상) 가능, GPU(T4)에서 학습 가능


### 비교 모델 실험
- YOLO11n을 기본 모델로 설정하여 성능 비교 실험 진행.

### 4.2 실험 결과 (mAP)

| Model                        | mAP    |
|------------------------------|--------|
| **CO-DETR O365 Cleansing**             | 0.7373 |
| **YOLO11n**                  | 0.3814 |
| **YOLO11n Cleansing + Curriculum Learning** | 0.4783 |
| **YOLO11n_TTA**              | 0.3944 |
| **YOLO11n_Cleansing + Curriculum Learning + TTA**              | 0.5271 |

### 4.3 실험 요약

- **CO-DETR O365 cleansing**: 데이터 클렌징과 Objects365 데이터셋으로 사전 학습 활용해 mAP 0.7373으로 최고 성능을 기록 
- **YOLO11n**: YOLO11n 모델은 소규모 데이터셋으로 학습되어 Validation mAP가 0.3814로 낮은 성능을 보임
- **YOLO11n Cleansing + Curriculum Learning**: 데이터 클렌징과 커리큘럼 러닝 기법을 적용하여 mAP가 0.4783으로 개선되어 일반화 성능이 향상
- **YOLO11n_TTA**: TTA(Test-Time Augmentation) 적용하여 mAP 0.3944으로 개선

### 4.4 결론 및 분석
- CO-DETR O365:  대규모 사전 학습(OpenImages 365)을 사용해 mAP 0.7373으로 최고 성능을 기록.
- YOLO11n은 소규모 데이터셋에서 과적합 경향(Validation mAP 0.3814) 보임. Cleansing과 Curriculum Learning으로 일반화 개선(0.4783).
- TTA는 소폭 향상(0.3944)했으나, 계산 비용 증가로 실시간성 저하.

## 5. 모델 최적화 및 개선

### 5.1 후처리 최적화
- **앙상블**: 서로 다른 구조나 학습 조건을 가진 모델들을 조합하여 단일 모델의 한계를 보완하고, 보다 robust한 결과를 도출하는 것을 목표
- **class wise**: 데이터를 Train/Validation/Check로 나눈 후, Check 데이터에서 클래스별로 높은 점수를 기록한 모델을 선별해 앙상블하려 함
  - 모델 학습 시간이 길어 시간 부족으로 인해 구현하지 못함
### 5.2 앙상블 결과

| 모델 조합                | 비율  | mAP    |
|--------------------------|-------|--------|
| WBF (O365 + Codino)     | 1:1   | 0.7297 |
| WBF (O365 + YOLO11n)    | 2:1   | 0.7180 |
| WBF (O365 + YOLO11n_TTA)    | 2:1   | 0.6490 |
## 6. 실험 환경
- **하드웨어**: Google Colab Pro, NVIDIA L4 GPU (24GB VRAM), 32GB RAM.
- **소프트웨어**: PyTorch 2.0, Albumentations 1.3, Python 3.9.

## 7. 한계점 및 미래 작업

### 한계점 (YOLO11n)
- **탐지율 저하**: `General trash`와 `Metal` 클래스의 탐지율이 낮음(mAP 0.3 이하). 주요 원인은 작은 객체(예: 50x50 픽셀 미만)가 많아 모델이 이를 인식하는 데 어려움이 있음.
- **소수 클래스 문제**: `Battery`와 같은 소수 클래스의 낮은 탐지율이 여전히 해결되지 않은 과제로 남아 있음.
- **실시간 성능**: 실시간 처리 속도(FPS)가 개선이 필요하며, 특히 `YOLOv11n`에서 복잡한 배경을 처리할 때 지연이 발생함.

### 개선점
- **클래스별 가중치 조정**: `General trash`와 `Metal` 클래스의 작은 객체에 대해 손실 함수 가중치를 높여 탐지율을 개선할 계획.
- **모델 경량화**: `Pruning`과 `Quantization` 기법을 적용하여 모델 크기와 연산 비용을 줄이고, 실시간 성능을 향상.
- **데이터셋 강화**: 작은 객체를 포함한 추가 데이터를 수집하고, 다양한 증강 기법을 통해 모델의 일반화 능력을 강화.

