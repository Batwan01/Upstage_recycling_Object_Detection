from ultralytics import YOLO

# 모델 초기화 (사전 훈련된 모델을 불러옴)
model = YOLO("ultralytics/cfg/models/v10/yolov10l.yaml").load("ultralytics/yolov10l.pt")

# 데이터 경로 설정 (data.yaml)
data_path = './data.yaml'

# 단계 1: 쉬운 데이터 (Easy)
# 목표: 모델이 기본적인 객체 인식을 익히게 함.
model.train(
    data=data_path,
    epochs=10,  # 쉬운 데이터이므로 학습 횟수 적게
    imgsz=640,  # 해상도 640으로 설정
    batch=8,    # 배치 사이즈 8
    lr0=0.002,  # 학습률 설정
    mosaic=0.2,  # 기본적으로 mosaic을 추가
    mixup=0.0,  # 쉬운 데이터에서는 mixup을 최소화
    flipud=0.5,  # 수평 뒤집기 (50%)
    hsv_h=0.2,  # 밝기/대비 조정
    hsv_s=0.2,
    hsv_v=0.2,
    crop_fraction=0.8,  # 랜덤 크롭
)

# 단계 2: 중간 데이터 (Medium)
# 목표: 복잡도가 약간 높은 데이터를 학습하며 일반화 능력 키움.
model.train(
    data=data_path,
    epochs=10,  # 중간 데이터에서는 10 에폭
    imgsz=832,  # 해상도 832로 설정
    batch=8,    # 배치 사이즈 8
    lr0=0.002,  # 학습률 설정
    mosaic=0.3, # mosaic 증강 비율 증가
    mixup=0.2,  # 약간의 mixup 사용
    rotate=15,  # ±15도 회전
    shift=0.1,  # 이동
    scale=0.1,  # 크기 조정
    shear=0.1,  # 왜곡
    blur=0.1,   # 흐림 효과
)

# 단계 3: 어려운 데이터 (Hard)
# 목표: 복잡한 상황에서 잘 동작하도록 모델을 학습.
model.train(
    data=data_path,
    epochs=10,  # 어려운 데이터에서는 10 에폭
    imgsz=1024,  # 해상도 최대로 설정
    batch=8,    # 배치 사이즈 8
    lr0=0.001,  # 학습률 설정
    mosaic=0.5, # mosaic 증강 비율 강하게
    mixup=0.3,  # mixup 비율 증가
    rotate=30,  # ±30도 회전
    shift=0.2,  # 이동
    scale=0.2,  # 크기 조정
    shear=0.2,  # 왜곡
    blur=0.2,   # 흐림 효과
    erasing=0.4,  # 무작위 가리기 (Random Erasing)
    noise=0.2,    # 노이즈 추가 (Gaussian)
    cutout=0.3,   # Cutout 사용
)
