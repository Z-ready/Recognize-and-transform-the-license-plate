import cv2
import numpy as np

def sort_corners(points):
    """按照左上、右上、右下、左下顺序排列四个点"""
    points = sorted(points, key=lambda x: (x[0] + x[1]))  # 先按 x+y 排序
    top_two = sorted(points[:2], key=lambda x: x[0])  # 取最上面的两个点，按 x 排序
    bottom_two = sorted(points[2:], key=lambda x: x[0])  # 取最下面的两个点，按 x 排序
    return np.array([top_two[0], top_two[1], bottom_two[1], bottom_two[0]])  # 返回左上、右上、右下、左下



image = cv2.imread("code\data\Computer_Vision_data\pics\plate2.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 先进行高斯模糊，减少噪声
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

#

#
cv2.imshow('Corrected Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

draw_img = image.copy()
res = cv2.drawContours(draw_img, contours, -1, (0, 255, 0), 4) 
#遍历每一个轮廓
for c in contours:
        peri = cv2.arcLength(c, True)      
        area = cv2.contourArea(c)
        print(area)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)  
        if len(approx) ==4 and area >5000:  
            print(area)
            print(approx.reshape(4,2))
            box =sort_corners( approx.reshape(4,2))
            break
res2 = cv2.polylines(image, [box], isClosed=True, color=(0, 0, 255), thickness=2)

cv2.imshow('Corrected Image', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(box)

pts1 = np.float32([box[0], box[1], box[2],box[3]])
pts2 = np.float32([[0, 0], [0, 150], [310, 150],[310, 0] ])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(image, M, (310, 150))

canvas = np.zeros((150, 310, 3), dtype=np.uint8)

canvas[:dst.shape[0], :dst.shape[1]] = dst


cv2.imshow('Corrected Image', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()