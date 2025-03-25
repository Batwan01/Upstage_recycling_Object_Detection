# 재활용 품목 분류를 위한 Object Detection
---

## 1. 문제 정의 및 목표 설정

### 1.1 목표

객체 탐지 모델을 설계하고 특정 데이터셋에서 객체를 정확히 탐지하는 것을 목표로 한다. 이를 위해 ~~ 모델을 기반으로 학습 및 평가를 진행하며, 성능 평가 지표로 mAP(mean Average Precision)와 IoU(Intersection over Union)를 사용한다.

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

- 라벨링 제거 및 추가
  - 0067, 0134, 1624, 0230, 0906, 0975, 2629, 2961, 2971, 3106, 3501, 3539, 3560, 3797, 3904, 4098, 4100, 4165, 4191, 4209, 4260, 4357, 4408

    <img src="https://github.com/user-attachments/assets/17049fdf-cc61-4425-a75a-a3712399fd54" width="45%" /> <img src="https://github.com/user-attachments/assets/b46c029b-e073-45b7-b289-5702267885df" width="45%" />

  
- 잘못된 라벨링 정제
  - 0210, 1569, 2361, 2422, 2424, 2505, 3127, 3163, 4153, 4186, 4225, 4270
    
    <img src="https://github.com/user-attachments/assets/79656282-d7a5-4e70-857b-0b40ce347732" width="45%" /> <img src="https://github.com/user-attachments/assets/069f0f18-a262-4be3-8f31-8c3b6ec35fc0" width="45%" />

## 3. 데이터 전처리 및 증강

### 3.1 데이터셋 분할

- **Train/Validation/check 비율**: 70/15/15
- **Stratified Split 적용** (클래스별 샘플 개수 균형 고려)

### 3.2 데이터 증강(Augmentation)

- **기본 변환**: 크롭, 좌우 반전, 밝기 조절
- **객체 보존 증강**: CutMix, MixUp, Mosaic 적용
- **라이브러리 활용**: Albumentations
- **## 3. 데이터 전처리 및 증강

### 3.1 데이터셋 분할

- **Train/Validation/check 비율**: 70/15/15
- **Stratified Split 적용** (클래스별 샘플 개수 균형 고려)

### 3.2 데이터 증강(Augmentation)

- **기본 변환**: 크롭, 좌우 반전, 밝기 조절
- **객체 보존 증강**: CutMix, MixUp, Mosaic 적용
- **라이브러리 활용**: Albumentations
- 커큘럼 러닝**: 점차 어려운 증강 적용

## 4. 모델 선택 및 학습 설정

### 4.1 모델 선정

- **YOLOv11n**: 실시간 성능과 정확도의 균형을 고려하여 선택 (Yolo11x 사양문제로 Yolo11n으로 실험)
- **CO-DETR**: 5-scale Swin-Large 백본, co_dino_5scale_swin_large_16e_o365tococo 가중치 사용
- **비교 모델 실험**: YOLO11n
