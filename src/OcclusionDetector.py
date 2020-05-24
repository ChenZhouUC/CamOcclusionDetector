import cv2
import numpy as np


def OcclusionDetector(frame, size=(150, 100)):
    # resize
    frame = cv2.resize(frame, size)
    # edge detection
    sobelX = cv2.Sobel(frame, cv2.CV_16S, 1, 0, ksize=3)
    sobelY = cv2.Sobel(frame, cv2.CV_16S, 0, 1, ksize=3)
    absX = cv2.convertScaleAbs(sobelX)   # 转回uint8
    absY = cv2.convertScaleAbs(sobelY)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # thresholding
    ret, thresh1 = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)

    cv2.imshow("winname", thresh1)
    cv2.waitKey(0)
    return np.random.random()


if __name__ == "__main__":

    frame = cv2.imread('../data/Occluded/309.jpg', 0)
    score = OcclusionDetector(frame)
    print(score)
