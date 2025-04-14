import cv2
import os
import datetime

def images_to_video(image_folder, output_video, fps=30):
    # 取得圖片檔案清單，並按名稱排序
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))  # 根據檔名數字排序

    # 讀取第一張圖片以取得影片尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # 定義影片的格式與輸出
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 編碼格式
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 將每張圖片寫入影片
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)

    # 釋放資源
    out.release()
    print(f"影片已儲存至：{output_video}")

# 使用範例
image_folder = "D:/Project/Yolov7-tracker-2/track_demo_results/vis_results"  # 修改為圖片資料夾路徑
# 取得當前時間並格式化為字串
current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
output_video = f"output/output_video_{current_time}.mp4"  # 修改為輸出影片名稱
images_to_video(image_folder, output_video)
