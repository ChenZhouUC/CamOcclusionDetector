import cv2
import numpy as np


def OcclusionDetector(frame, size=(150, 100)):
    frame = cv2.resize(frame, size)
    return np.random.random()


if __name__ == "__main__":

    frame = cv2.imread('../data/Occluded/309.jpg')
    score = OcclusionDetector(frame)
    print(score)
