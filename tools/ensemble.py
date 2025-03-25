import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from ensemble_boxes import *
from multiprocessing import Pool

def set_parser():
    parser = argparse.ArgumentParser(prog="Ensemble", description="Ensemble csv files")
    p = Path.cwd()
    parser.add_argument('-n', '--name', type=str, default='weighted_boxes_fusion', help="앙상블 방법")
    parser.add_argument('-i', '--iou_thr', type=float, default=0.5, help="IOU 임계값")
    parser.add_argument('-sbt', '--skip_box_thr', type=float, default=0.0001, help="박스 스킵 임계값")
    parser.add_argument('-sig', '--sigma', type=float, default=0.1, help="Soft NMS의 시그마 값")
    parser.add_argument('-t', '--target_directory', type=str, default=p.joinpath('target'), help="앙상블 대상 CSV 디렉토리")
    parser.add_argument('-o', '--output_directory', type=str, default=p.joinpath('ensemble'), help="앙상블 결과 저장 디렉토리")
    parser.add_argument('-l', '--log_file', type=str, default=p.joinpath('meta_data.md'), help="로그 파일 경로")
    parser.add_argument('-w', '--width', type=int, default=1024, help="이미지 너비")
    parser.add_argument('-hi', '--height', type=int, default=1024, help="이미지 높이")
    return parser

def load_all_csv_data(target_dir):
    csv_datas = {}
    for output in sorted(os.listdir(target_dir)):
        if not output.endswith('.csv'):
            continue
        csv_path = os.path.join(target_dir, output)
        try:
            csv_data = pd.read_csv(csv_path).set_index('image_id', drop=False)
            csv_datas[output] = csv_data
            print(f"Loaded: {output}")
        except Exception as e:
            print(f"Warning: Failed to load {csv_path} - {e}")
    return csv_datas

def return_image_ids(csv_datas):
    image_ids = set()
    for csv_data in csv_datas.values():
        image_ids.update(csv_data['image_id'].unique())
    return sorted(list(image_ids))

def save_target_data(target_dir, log_file_name='meta_data.md'):
    try:
        os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
        with open(log_file_name, 'a') as f:
            f.write("Ensemble target:\n")
            for target_file in os.listdir(target_dir):
                if target_file.endswith('.csv'):
                    f.write(f'- {target_file}\n')
        print("Success: Saved target names to log")
    except Exception as e:
        print(f"Error saving target data: {e}")

def save_output_data(submission, submission_file, log_file='meta_data.md', error_msg=None):
    p = Path(submission_file)
    try:
        os.makedirs(p.parent, exist_ok=True)
        submission.to_csv(submission_file, index=False)
        condition = 'Success' if not error_msg else 'Error'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(f"{condition}: Submission file saved to {submission_file}\n")
            if error_msg:
                f.write(f'Error details: {error_msg}\n')
        print(f"{condition}: Saved submission to {submission_file}")
    except Exception as e:
        print(f"Error saving submission: {e}")
    return submission_file

def make_ensemble_format_per_image(image_id, csv_datas, image_width, image_height):
    boxes_list, scores_list, labels_list = [], [], []
    for csv_data in csv_datas:
        predict_row = csv_data[csv_data['image_id'] == image_id]['PredictionString']
        if predict_row.empty:
            continue
        predict_string = predict_row.iloc[0]
        if pd.isna(predict_string) or len(str(predict_string).strip()) < 2:
            continue

        predict_list = np.array(str(predict_string).split()).reshape(-1, 6)
        boxes = predict_list[:, 2:6].astype(float) / [image_width, image_height, image_width, image_height]
        scores = predict_list[:, 1].astype(float)
        labels = predict_list[:, 0].astype(int)

        if (boxes > 1).any() or (boxes < 0).any():
            print(f"Warning: Boxes out of range for image_id {image_id}")

        boxes_list.append(boxes.tolist())
        scores_list.append(scores.tolist())
        labels_list.append(labels.tolist())
    return boxes_list, scores_list, labels_list

def prediction_format_per_image(boxes, scores, labels, image_width, image_height):
    if len(boxes) == 0 or boxes is None:  # 수정된 조건
        return ''
    output = ''
    for box, score, label in zip(boxes, scores, labels):
        box = [min(max(b, 0), 1) for b in box]
        x1, y1, x2, y2 = [b * s for b, s in zip(box, [image_width, image_height, image_width, image_height])]
        output += f'{int(label)} {score} {x1} {y1} {x2} {y2} '
    return output.strip()

def process_image(args_tuple):
    image_id, csv_datas, ensemble_name, iou_thr, skip_box_thr, sigma, width, height = args_tuple
    boxes, scores, labels = make_ensemble_format_per_image(image_id, csv_datas, width, height)
    if not boxes:
        return image_id, ''

    weights = [2, 1]
    if len(csv_datas) != 2:
        print(f"Warning: Expected 2 models, but got {len(csv_datas)}. Using equal weights.")
        weights = [1] * len(boxes)

    if ensemble_name == 'nms':
        results = nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr)
    elif ensemble_name == 'soft_nms':
        results = soft_nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    elif ensemble_name == 'non_maximum_weighted':
        results = non_maximum_weighted(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    elif ensemble_name == 'weighted_boxes_fusion':
        results = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_name}")
    
    return image_id, prediction_format_per_image(*results, width, height)

def main():
    parser = set_parser()
    args = parser.parse_args()

    ensemble_name = args.name
    iou_thr = args.iou_thr
    skip_box_thr = args.skip_box_thr
    sigma = args.sigma
    target_dir = args.target_directory
    output_dir = args.output_directory
    log_file = args.log_file
    image_width = args.width
    image_height = args.height

    csv_datas = load_all_csv_data(target_dir)
    if not csv_datas:
        print("Error: No valid CSV files found in target directory")
        return

    image_ids = return_image_ids(csv_datas)
    save_target_data(target_dir, log_file)

    args_list = [(image_id, list(csv_datas.values()), ensemble_name, iou_thr, skip_box_thr, sigma, image_width, image_height) 
                 for image_id in image_ids]
    
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_image, args_list), total=len(image_ids)))

    image_ids, prediction_strings = zip(*results)
    submission = pd.DataFrame({'image_id': image_ids, 'PredictionString': prediction_strings})

    output_file = Path(output_dir) / f'{ensemble_name}_result.csv'
    save_output_data(submission, output_file, log_file)

if __name__ == '__main__':
    main()

# %cd /content/drive/MyDrive/Colab Notebooks/data/ensemble
# !python ensemble.py -n weighted_boxes_fusion -t '/content/drive/MyDrive/Colab Notebooks/data/ensemble/csv' -o ./ensemble_csv -w 1024 -hi 1024 -i 0.5 -sbt 0.0001