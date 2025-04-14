import cv2
import datetime

# 參數設置
txt_file = "D:/Project/Yolov7-tracker-2/track_results/track_demo_results/txt_results/demo.txt"  # 追蹤輸出TXT檔案

input_video = "D:/Project/Yolov7-tracker-2/video/A2A_002.mp4"  # 輸入影片

current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
output_video = f"output/tracking_result_{current_time}.mp4"  # 輸出的影片檔案名稱


# 開啟影片
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 獲取FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化影片寫入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# 讀取追蹤資料，存為字典 {frame_id: [(obj_id, x, y, w, h, conf), ...]}
tracking_data = {}
with open(txt_file, 'r') as f:
    for line in f:
        data = line.strip().split(',')
        frame_id = int(data[0])
        obj_id = int(data[1])
        x, y, w, h = map(float, data[2:6])
        conf = float(data[6])

        if frame_id not in tracking_data:
            tracking_data[frame_id] = []
        tracking_data[frame_id].append((obj_id, x, y, w, h, conf))

# 繪製Bounding Box於影片影格上
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 影片讀取完畢

    # 如果有對應的追蹤資料，繪製Bounding Box
    if frame_id in tracking_data:
        for obj_id, x, y, w, h, conf in tracking_data[frame_id]:
            # 繪製Bounding Box
            color = (0, 255, 0)  # 綠色框
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

            # 繪製標籤
            label = f"ID:{obj_id} Conf:{conf:.2f}"
            cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 寫入影格至輸出影片
    out.write(frame)
    frame_id += 1

# 釋放資源
cap.release()
out.release()
print("影片已儲存至", output_video)
