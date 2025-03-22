import json
import numpy as np
from sklearn.model_selection import train_test_split

# JSON 파일 경로
data_root = "/content/drive/MyDrive/Colab_Notebooks/data/Upstage_recycling_Object_Detection/recycling"
json_path = f"{data_root}/train.json"

# JSON 파일 로드
with open(json_path, "r") as f:
    ann_data = json.load(f)

# 이미지 및 어노테이션 정보 추출
images = ann_data["images"]
annotations = ann_data["annotations"]
categories = ann_data["categories"]

# 총 이미지 개수 확인
total_images = len(images)
print(f"📊 총 이미지 개수: {total_images}장")

# 이미지 ID별 어노테이션 매핑
image_to_ann = {img["id"]: [] for img in images}
for ann in annotations:
    image_to_ann[ann["image_id"]].append(ann)

# 각 이미지의 대표 클래스 추출 (첫 번째 어노테이션 기준, 없으면 -1)
image_labels = [image_to_ann[img["id"]][0]["category_id"] if image_to_ann[img["id"]] else -1 for img in images]

# Stratified Split (train 70%, val 15%, check 15%)
train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    images, image_labels, test_size=0.3, stratify=image_labels, random_state=42
)

val_imgs, check_imgs, val_labels, check_labels = train_test_split(
    temp_imgs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# 해당 이미지 ID에 맞는 어노테이션만 필터링
def filter_annotations(imgs):
    img_ids = {img["id"] for img in imgs}
    return [ann for ann in annotations if ann["image_id"] in img_ids]

train_anns = filter_annotations(train_imgs)
val_anns = filter_annotations(val_imgs)
check_anns = filter_annotations(check_imgs)

# 새로운 JSON 데이터 생성
def save_json(file_name, imgs, anns):
    split_data = {
        "images": imgs,
        "annotations": anns,
        "categories": categories
    }
    with open(f"{data_root}/{file_name}", "w") as f:
        json.dump(split_data, f, indent=4)

# JSON 저장
save_json("train_split.json", train_imgs, train_anns)
save_json("val.json", val_imgs, val_anns)
save_json("check.json", check_imgs, check_anns)

# 결과 출력
print(f"📂 데이터 분할 완료!")
print(f"Train: {len(train_imgs)}개")
print(f"Val: {len(val_imgs)}개")
print(f"Check: {len(check_imgs)}개")
