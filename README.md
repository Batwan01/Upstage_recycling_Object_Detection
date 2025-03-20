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
