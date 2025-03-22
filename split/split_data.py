import json
import random
from sklearn.model_selection import train_test_split

# 원본 train.json 로드
train_json_path = '/content/drive/MyDrive/Colab_Notebooks/data/Upstage_recycling_Object_Detection/recycling/train.json'
with open(train_json_path, 'r') as f:
    data = json.load(f)

# 이미지 리스트와 어노테이션 리스트 추출
images = data['images']
annotations = data['annotations']
categories = data['categories']

# 이미지 ID로 그룹화된 어노테이션 매핑
image_ids = [img['id'] for img in images]
ann_by_image = {img_id: [] for img_id in image_ids}
for ann in annotations:
    ann_by_image[ann['image_id']].append(ann)

# train/val 이미지 분리 (예: 80% train, 20% val)
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

# 분리된 이미지에 맞는 어노테이션 분리
train_image_ids = set(img['id'] for img in train_images)
val_image_ids = set(img['id'] for img in val_images)

train_annotations = [ann for img_id in train_image_ids for ann in ann_by_image[img_id]]
val_annotations = [ann for img_id in val_image_ids for ann in ann_by_image[img_id]]

# 새로운 train.json과 val.json 생성
train_data = {
    'images': train_images,
    'annotations': train_annotations,
    'categories': categories
}
val_data = {
    'images': val_images,
    'annotations': val_annotations,
    'categories': categories
}

# 저장
new_train_json_path = '/content/drive/MyDrive/Colab_Notebooks/data/Upstage_recycling_Object_Detection/recycling/train_split.json'
val_json_path = '/content/drive/MyDrive/Colab_Notebooks/data/Upstage_recycling_Object_Detection/recycling/val.json'
with open(new_train_json_path, 'w') as f:
    json.dump(train_data, f)
with open(val_json_path, 'w') as f:
    json.dump(val_data, f)

print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")
print(f"Train annotations: {len(train_annotations)}, Val annotations: {len(val_annotations)}")