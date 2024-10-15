import cv2
import numpy as np
import pyautogui

# 加载 YOLO 模型
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')  # 确保 yolov3.weights 和 yolov3.cfg 文件存在
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 加载类别名称（COCO 数据集）
classes = []
with open("coco.names", "r") as f:  # 确保 coco.names 文件存在
    classes = [line.strip() for line in f.readlines()]

# 获取屏幕尺寸（用于鼠标控制）
screen_width, screen_height = pyautogui.size()

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # 读取视频帧
    if not ret:
        break

    height, width, channels = frame.shape

    # YOLO 模型预处理
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 处理输出数据
    for out in outs:
        for detection in out:
            scores = detection[5:]  # 检测类别的得分
            class_id = np.argmax(scores)  # 获取类别ID
            confidence = scores[class_id]  # 获取置信度

            if confidence > 0.5 and classes[class_id] == "person":  # 只处理置信度大于 50% 且为 "person" 的目标
                # 获取边界框的中心点和宽高
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 计算边界框的左上角坐标
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # 绘制矩形框（可视化）
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 模拟鼠标瞄准敌人头部
                head_x = center_x
                head_y = int(center_y - h / 2)  # 假设头部在检测框的顶部位置

                # 将检测框的坐标映射到屏幕坐标
                screen_x = int((head_x / width) * screen_width)
                screen_y = int((head_y / height) * screen_height)

                # 移动鼠标到敌人头部
                pyautogui.moveTo(screen_x, screen_y)
                pyautogui.click()  # 模拟鼠标点击（开火）

    # 显示处理后的帧
    cv2.imshow("Game Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

cap.release()
cv2.destroyAllWindows()
