import os
import cv2
import numpy as np


def gray_world_white_balance(img):
    # 分离 BGR 三通道
    B, G, R = cv2.split(img)
    
    # 计算每个通道的均值
    mean_B, mean_G, mean_R = np.mean(B), np.mean(G), np.mean(R)
    
    # 计算整体灰度均值
    mean_gray = (mean_B + mean_G + mean_R) / 3
    
    # 计算缩放比例
    scale_B, scale_G, scale_R = mean_gray / mean_B, mean_gray / mean_G, mean_gray / mean_R
    
    # 调整通道
    B = np.clip(B * scale_B, 0, 255).astype(np.uint8)
    G = np.clip(G * scale_G, 0, 255).astype(np.uint8)
    R = np.clip(R * scale_R, 0, 255).astype(np.uint8)

    # 合并回 BGR
    balanced_img = cv2.merge([B, G, R])
    return balanced_img


def sort_corners(points):
    """按照左上、右上、右下、左下顺序排列四个点嘻嘻"""
    points = sorted( points, key=lambda x: x[1])  # 先按 y 排序
    top_two = sorted( points[:2], key=lambda x: x[0])  # 取最上面的两个点，按 x 排序
    bottom_two = sorted( points[2:], key=lambda x: x[0])  # 取最下面的两个点，按 x 排序
    return np.array([top_two[0], top_two[1], bottom_two[1], bottom_two[0]])  # 返回左上、右上、右下、左下


def transform(points, img):
    """将数据变换成正面图像"""
    pts1 = np.float32([points[0], points[1], points[2], points[3]])
    pts2 = np.float32([[0, 0], [300, 0], [300, 150], [0, 150]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (300, 150))
    canvas = np.zeros((150, 300, 3), dtype=np.uint8)
    canvas[:dst.shape[0], :dst.shape[1]] = dst
    return dst

input_folder = 'input_folder_path'  # 输入文件夹路径
output_folder = 'output_folder_path'  # 输出文件夹路径


# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Unable to load image {filename}")
            continue
    # 读取图像并转换为 HSV
    wb_img = gray_world_white_balance(image)
    

    # 显示矫正后图像
    # cv2.imshow("White Balanced", wb_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    hsv = cv2.cvtColor(wb_img, cv2.COLOR_BGR2HSV)

    # 设定蓝色车牌的 HSV 颜色范围
    lower_blue = np.array([70, 60, 95])   # 调整范围适配不同光照
    upper_blue = np.array([140, 255, 255])

    #  颜色过滤
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #msck图像
    # cv2.imshow('Mask', mask)

    # 形态学操作去噪
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((15, 15), np.uint8)
    # 使用形态学闭运算填补缝隙
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有轮廓并绘制最小外接矩形
    approxlist = []
    a=0
    for cnt in contours:
    
        if cv2.contourArea(cnt) > 500 and cv2.contourArea(cnt) < 150000 :  # 过滤小轮廓
            peri = cv2.arcLength(cnt, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True) 
            
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            #遍历轮廓图像
            # cv2.imshow('Bounding Box', image)

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if len(approx) == 4: 
                # print(approx)
                a = a+1
                print(a)
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
                point = sort_corners(approx.reshape(4, 2))
                dst = transform(point,image)
                approxlist.append(dst)
                #去掉break即可识别多张车牌在同一个照片里
                break
            else:
                break
    # 保存拼接后的图像
    if len(approxlist) !=0:
        output_path = os.path.join(output_folder, f"processed_{filename}")
        cv2.imwrite(output_path, dst)
        print(f"Processed image saved to {output_path}")
    approxlist = []


        

        