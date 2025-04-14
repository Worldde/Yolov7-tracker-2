'''
一次測試多個追蹤器的 MOTA 與 IDF1
'''

import motmetrics as mm
import pandas as pd
import os

# 設定資料夾與 tracker 檔名
# dataset_name = "multi_drone_241016_001"
dataset_name = "A2A_002"
data_dir = f"/kaggle/working/Yolov7-tracker-2/track_demo_results/{dataset_name}"
gt_path = f'/kaggle/working/Yolov7-tracker-2/ground_truth/{dataset_name}_gt.txt'

# 要評估的 tracker 預測檔名
tracker_files = [
    "deepsort_pred.txt",
    "sort_pred.txt",
    "botsort_pred.txt",
    "bytetrack_pred.txt",
    "c_bioutrack_pred.txt"
]

# 命名欄位
columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']

# 載入 GT
gt = pd.read_csv(gt_path, header=None)
gt.columns = columns

# 評估結果收集器
acc_dict = {}

# 建立距離計算器（用 IoU）
def iou_distance_matrix(gt_df, pred_df):
    from motmetrics.utils import iou_matrix
    gt_boxes = gt_df[['x', 'y', 'w', 'h']].values
    pred_boxes = pred_df[['x', 'y', 'w', 'h']].values
    return 1 - iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)

# 對每個 tracker 做評估
for pred_file in tracker_files:
    tracker_name = pred_file.replace('_pred.txt', '')

    pred_path = os.path.join(data_dir, pred_file)
    if not os.path.exists(pred_path):
        print(f"找不到預測檔案：{pred_path}")
        continue

    pred = pd.read_csv(pred_path, header=None)
    pred.columns = columns

    acc = mm.MOTAccumulator(auto_id=True)

    for frame in sorted(gt['frame'].unique()):
        gt_frame = gt[gt['frame'] == frame]
        pred_frame = pred[pred['frame'] == frame]

        gt_ids = gt_frame['id'].values
        pred_ids = pred_frame['id'].values

        if len(gt_frame) == 0 and len(pred_frame) == 0:
            acc.update([], [], [])
            continue

        dist = iou_distance_matrix(gt_frame, pred_frame)
        acc.update(gt_ids, pred_ids, dist)

    acc_dict[tracker_name] = acc

# 統一輸出評估表格
mh = mm.metrics.create()
summary = mh.compute_many(
    list(acc_dict.values()),   # Accumulator 列表
    metrics=mm.metrics.motchallenge_metrics,
    names=list(acc_dict.keys())  # 把 dict_keys 強制轉為 list
)


# 印出漂亮的比較表格
print(mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
))

