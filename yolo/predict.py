import csv
from ultralytics import YOLO
import os

# Function to write predictions to a CSV file
def write_to_csv(predictions, csv_path):
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["PredictionString", "image_id"])
        if f.tell() == 0:  # If the file is empty, write the header
            writer.writeheader()
        for data in predictions:  # 여러 결과를 한 번에 기록
            writer.writerow(data)

# Load the custom YOLO model
model = YOLO("/content/drive/MyDrive/Colab Notebooks/data/upstage_test/runs/yolo11n/best.pt")

# 테스트 데이터셋 이미지 경로 리스트 생성
test_dir = "/content/drive/MyDrive/Colab Notebooks/data/upstage_test/datasets/test"
# 숨김 파일(._로 시작)을 제외하고 유효한 이미지 확장자만 포함
image_paths = [
    os.path.join(test_dir, fname) 
    for fname in os.listdir(test_dir) 
    if fname.endswith(('.jpg', '.png')) and not fname.startswith('._')
]

# 파일 이름을 오름차순으로 정렬
image_paths = sorted(image_paths, key=lambda x: os.path.basename(x))

# 주기적 저장 간격 설정
save_interval = 100  # 100장마다 저장

# 저장 경로 설정
csv_path = "/content/predictions_1024.csv"

# 중간 저장을 위한 리스트
predictions_buffer = []
processed_count = 0

# Process images one by one in sorted order
for image_path in image_paths:
    try:
        # Predict on a single image
        results = model(image_path, save=True)
        result = results[0]  # 단일 이미지 결과는 리스트의 첫 번째 요소

        pred_str = ""
        for box in result.boxes:
            cls = int(box.cls.item())
            conf = box.conf.item()
            xyxy = [coord.item() for coord in box.xyxy[0]]
            pred_str += f"{cls} {conf} {xyxy[0]} {xyxy[1]} {xyxy[2]} {xyxy[3]} "

        # 예측 결과를 버퍼에 추가
        predictions_buffer.append({
            "PredictionString": pred_str.strip(),
            "image_id": f'test/{os.path.basename(result.path)}'
        })

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        # 오류 발생 시 공백 PredictionString 추가
        predictions_buffer.append({
            "PredictionString": "",  # 공백 문자열
            "image_id": f'test/{os.path.basename(image_path)}'
        })

    processed_count += 1

    # 주기적으로 저장
    if processed_count % save_interval == 0:
        write_to_csv(predictions_buffer, csv_path)
        predictions_buffer = []  # 버퍼 초기화
        print(f"Saved {processed_count} predictions to {csv_path}")

# 남은 결과 저장
if predictions_buffer:
    write_to_csv(predictions_buffer, csv_path)
    print(f"Saved remaining {len(predictions_buffer)} predictions to {csv_path}")

print(f"All predictions saved to {csv_path}")