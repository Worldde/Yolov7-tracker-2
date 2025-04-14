import sys, os
import numpy as np
import torch
import cv2 
from PIL import Image
from tqdm import tqdm
import yaml 
import time
import json
from loguru import logger 
import argparse

# 確保 yolov7 模型能正確載入
sys.path.append(os.getcwd())  # 加入目前工作目錄

from tracking_utils.envs import select_device
from tracking_utils.tools import *
from tracking_utils.visualization import plot_img, save_video

from tracker_dataloader import TestDataset, DemoDataset

# trackers 
from trackers.byte_tracker import ByteTracker
from trackers.sort_tracker import SortTracker
from trackers.botsort_tracker import BotTracker
from trackers.c_biou_tracker import C_BIoUTracker
from trackers.ocsort_tracker import OCSortTracker
from trackers.deepsort_tracker import DeepSortTracker

TRACKER_DICT = {
    'sort': SortTracker, 
    'bytetrack': ByteTracker, 
    'botsort': BotTracker, 
    'c_bioutrack': C_BIoUTracker,
    'deepsort': DeepSortTracker
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str, default='yolov7', help='yolov7, yolox, yolov8')
    parser.add_argument('--dataset', type=str, default='multi_drone_241016_001', help='dataset name')
    parser.add_argument('--reid_model', type=str, default='osnet_x0_25')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--conf_thresh', type=float, default=0.5)
    parser.add_argument('--nms_thresh', type=float, default=0.7)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--yolox_exp_file', type=str, default='./tracker/yolox_utils/yolox_m.py')
    parser.add_argument('--detector_model_path', type=str, default='./weights/best_AGDS_v001.pt')
    parser.add_argument('--trace', type=bool, default=True)
    parser.add_argument('--reid_model_path', type=str, default='./weights/osnet_x0_25.pth')
    parser.add_argument('--dhn_path', type=str, default='./weights/DHN.pth')
    parser.add_argument('--discard_reid', action='store_true')
    parser.add_argument('--track_buffer', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--min_area', type=float, default=200)
    parser.add_argument('--save_dir', type=str, default='track_demo_results')
    parser.add_argument('--save_images', default=False)
    parser.add_argument('--save_videos', default=False)
    parser.add_argument('--track_eval', type=bool, default=True)
    parser.add_argument('--kalman_format', type=str, default='sort', help='use what kind of Kalman: sort, byte, bot, etc.')
    args = parser.parse_args()
    args.obj = f'video/{args.dataset}_output_video.mp4'
    return args

def main(args, tracker_name):
    args.tracker = tracker_name

    # 自動設定 kalman_format
    if tracker_name == 'sort':
        args.kalman_format = 'sort'
    elif tracker_name in ['deepsort', 'bytetrack']:
        args.kalman_format = 'byte'
    elif tracker_name in ['botsort', 'c_bioutrack']:
        args.kalman_format = 'bot'
    else:
        args.kalman_format = 'sort'

    testdata = args.dataset
    device = select_device(args.device)

    if args.detector == 'yolov7':
        from models.experimental import attempt_load
        from utils.torch_utils import TracedModel
        from utils.general import non_max_suppression, scale_coords, check_img_size
        from yolov7_utils.postprocess import postprocess as postprocess_yolov7

        model = attempt_load(args.detector_model_path, map_location=device)
        stride = int(model.stride.max())
        model_img_size = check_img_size(args.img_size, s=stride)
        model = TracedModel(model, device=device, img_size=args.img_size)
        logger.info("Yolov7 detector loaded")

    else:
        logger.error("Only yolov7 currently supported in this script")
        return

    dataset = DemoDataset(file_name=args.obj, img_size=model_img_size, model=args.detector, stride=stride)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    tracker = TRACKER_DICT[args.tracker](args)

    results = []
    results_json = {tracker_name: []}
    fps_dict = {tracker_name: 0}
    frame_count = 0
    start_time = time.time()

    for frame_idx, (ori_img, img) in tqdm(enumerate(data_loader), total=len(data_loader)):
        img = img.to(device).float()
        ori_img = ori_img.squeeze(0)
        with torch.no_grad():
            output = model(img)
        output = postprocess_yolov7(output, args.conf_thresh, args.nms_thresh, img.shape[2:], ori_img.shape)

        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        output[:, 2] -= output[:, 0]
        output[:, 3] -= output[:, 1]

        current_tracks = tracker.update(output, img, ori_img.cpu().numpy())

        cur_tlwh, cur_id, cur_cls, cur_score = [], [], [], []
        for trk in current_tracks:
            bbox = trk.tlwh
            id = trk.track_id
            cls = trk.category
            score = trk.score
            if bbox[2] * bbox[3] > args.min_area:
                cur_tlwh.append(bbox)
                cur_id.append(id)
                cur_cls.append(cls)
                cur_score.append(score)

        results.append((frame_idx + 1, cur_id, cur_tlwh, cur_cls, cur_score))
        frame_count += 1

        if args.save_images:
            plot_img(img=ori_img, frame_id=frame_idx, results=[cur_tlwh, cur_id, cur_cls], 
                     save_dir=os.path.join(args.save_dir, 'vis_results', tracker_name))

    org_h, org_w = 720, 1280
    re_h, re_w = ori_img.shape[:2]
    scale_x, scale_y = org_h / re_h, org_w / re_w

    pred_path = os.path.join(args.save_dir, testdata, f"{tracker_name}_pred.txt")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)

    with open(pred_path, 'w') as f:
        for frame_idx, ids, boxes, classes, scores in results:
            for i in range(len(ids)):
                x, y, w, h = boxes[i]
                x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
                f.write(f"{frame_idx},{ids[i]},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{scores[i]:.2f},{classes[i]},1\n")

    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    fps_dict[tracker_name] = fps
    logger.info(f"{tracker_name}: FPS={fps:.2f}")

    if args.save_videos:
        save_video(images_path=os.path.join(args.save_dir, 'vis_results', tracker_name))

    tracker_results = {
        'results': results_json[tracker_name],
        'fps_dict': fps_dict[tracker_name]
    }
    with open(os.path.join(args.save_dir, testdata, f'tracker_results_{tracker_name}.json'), 'w') as f:
        json.dump(tracker_results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    args = get_args()
    for tracker_name in TRACKER_DICT.keys():
        print(f"\n===== Testing tracker: {tracker_name} =====")
        main(args, tracker_name)
