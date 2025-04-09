import cv2
from matplotlib import pyplot as plt
import numpy as np


def gray_world_white_balance(img):
    """修正原图白平衡"""
    B, G, R = cv2.split(img)
    mean_B, mean_G, mean_R = np.mean(B), np.mean(G), np.mean(R)
    mean_gray = (mean_B + mean_G + mean_R) / 3
    scale_B, scale_G, scale_R = mean_gray / mean_B, mean_gray / mean_G, mean_gray / mean_R
    B = np.clip(B * scale_B, 0, 255).astype(np.uint8)
    G = np.clip(G * scale_G, 0, 255).astype(np.uint8)
    R = np.clip(R * scale_R, 0, 255).astype(np.uint8)
    balanced_img = cv2.merge([B, G, R])
    return balanced_img


def sort_corners(points):
    """按照左上、右上、右下、左下顺序排列四个点"""
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
def averageBlueLight(wb_img):
    """计算蓝色平均亮度"""
    blue = wb_img[:,:,0]
    average_blue_light = np.mean(blue)
    average_blue_light = int(average_blue_light)
    # print(f"Average Brightness: {average_blue_light}")
    return average_blue_light, average_blue_light


# 读取图像-----------------------------------------------------------------------------------------------------------------------------
image = cv2.imread("image_path")
wb_img = gray_world_white_balance(image)

# 设定蓝色车牌的 HSV 颜色范围-------------------------------------------------------------------------------------------------------------
bluelight, avlight = averageBlueLight(wb_img)
hsv = cv2.cvtColor(wb_img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([70, 80, avlight-30])   # 调整范围适配不同光照
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 施工ing-------------------------------------------------------------------------------------------------------------------------------
s_values = hsv[:, :, 1][mask > 0]
s_values=int(np.mean(s_values))
# 形态学操作去噪-------------------------------------------------------------------------------------------------------------------------
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 使用形态学闭运算填补缝隙-----------------------------------------------------------------------------------------------------------------
kernel = np.ones((15, 15), np.uint8)
closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 开始寻找轮廓---------------------------------------------------------------------------------------------------------------------------
approxlist = []
a=0
for cnt in contours:
    # print(cv2.contourArea(cnt))#查看每个轮廓面积
    if cv2.contourArea(cnt) > 500 and cv2.contourArea(cnt) < 150000 :  # 过滤小轮廓
        peri = cv2.arcLength(cnt, True)  # 计算轮廓周长
        approx = cv2.approxPolyDP(cnt, 0.06 * peri, True) 
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        # #遍历轮廓图像
        # for i in range(len(approx)):
        #     cv2.imshow(f'Bounding Box{i}', image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        if len(approx) == 4: 
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            point = sort_corners(approx.reshape(4, 2))
            dst = transform(point,image)
            approxlist.append(dst)
            #去掉break即可识别多张车牌在同一个照片里
            break
print(f"图中已识别出{len(approxlist)}个车牌")


plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('car_plate')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.title("plate")
plt.axis('off')

plt.show()
