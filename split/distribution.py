import json
import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 파일 경로
data_root = "/content/drive/MyDrive/Colab_Notebooks/data/Upstage_recycling_Object_Detection/recycling/"
datasets = ["train.json", "val.json", "check.json"]

# 클래스 정의 (순서 유지)
classes = [
    "General trash", "Paper", "Paper pack", "Metal", "Glass",
    "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"
]

# 클래스별 개수 저장용
class_counts = {dataset: {cls: 0 for cls in classes} for dataset in datasets}

# 데이터셋별 클래스 개수 계산
for dataset in datasets:
    with open(data_root + dataset, "r") as f:
        data = json.load(f)
    
    for ann in data["annotations"]:
        class_id = ann["category_id"]
        class_name = classes[class_id]  # COCO format이라면 class_id가 0부터 시작할 것
        class_counts[dataset][class_name] += 1

# 결과 출력 및 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, dataset in enumerate(datasets):
    counts = list(class_counts[dataset].values())
    labels = list(class_counts[dataset].keys())
    
    axes[i].barh(labels, counts, color="skyblue")
    axes[i].set_title(f"{dataset} Class Distribution")
    axes[i].set_xlabel("Count")
    axes[i].set_ylabel("Class")

plt.tight_layout()
plt.show()

# 개수 출력
for dataset in datasets:
    print(f"\n{dataset} 클래스 분포:")
    for cls, count in class_counts[dataset].items():
        print(f"{cls}: {count}")
