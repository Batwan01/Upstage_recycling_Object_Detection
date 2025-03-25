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

