import cv2
import numpy as np

image = cv2.imread('p2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)


edges = cv2.Canny(blurred, 50, 100, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=5, maxLineGap=8)


# 绘制直线
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 保存结果
cv2.imwrite('result.png', image)
