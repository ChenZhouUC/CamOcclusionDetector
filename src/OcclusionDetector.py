import cv2
import numpy as np


def OcclusionDetector(frame, size=(150, 100), super_pixel_set=(3, 2)):
    # resize
    frame = cv2.resize(frame, size)
    # cv2.imshow("winname", frame)
    # cv2.waitKey(0)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # edge detection
    sobelX = cv2.Sobel(frame, cv2.CV_16S, 1, 0, ksize=3)
    sobelY = cv2.Sobel(frame, cv2.CV_16S, 0, 1, ksize=3)
    absX = cv2.convertScaleAbs(sobelX)   # 转回uint8
    absY = cv2.convertScaleAbs(sobelY)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # thresholding
    ret, thresh = cv2.threshold(dst, 50, 1, cv2.THRESH_BINARY)
    # super pixel setting
    w_num_pixel = int(size[0]/super_pixel_set[0])
    h_num_pixel = int(size[1]/super_pixel_set[1])
    score_mat = np.zeros(super_pixel_set[::-1])
    for w in range(super_pixel_set[0]):
        for h in range(super_pixel_set[1]):
            score = np.mean(thresh[h_num_pixel*h:h_num_pixel *
                                   (h+1), w_num_pixel*w:w_num_pixel*(w+1)])
            score_mat[h, w] = score
    divider = np.mean(score_mat)
    score = np.std(score_mat) / divider / \
        2 if divider > 0 else 1
    # print(score)
    return min(1, score)


if __name__ == "__main__":

    frame = cv2.imread('../data/Occluded/309.jpg', 0)
    score = OcclusionDetector(frame)
    print(score)
