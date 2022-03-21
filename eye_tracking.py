import cv2
import numpy as np
import matplotlib.pyplot as plt


class EyeTracker:
    def __init__(self):
        self.leye_idx = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        self.reye_idx = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

        self.l_eye_vert = [159, 145]
        self.l_eye_hor = [39, 13]
        self.r_eye_vert = [384, 374]
        self.r_eye_hor = [263, 362]

        self.fig, self.axes = plt.subplots(constrained_layout=True, nrows=2, ncols=1)
        self.ax1 = self.axes[0]
        self.ax2 = self.axes[1]

        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 1500
        detector_params.minArea= 100

        self.detector = cv2.SimpleBlobDetector_create(detector_params)

        self.threshold = 90


    def track_eyes(self, image, masks, eyes, threshold):

        self.threshold = threshold


        l, r = eyes
        l_x_min, l_y_min, l_x_max, l_y_max = min(l[:, 0]), min(l[:, 1]), max(l[:, 0]), max(l[:, 1])
        r_x_min, r_y_min, r_x_max, r_y_max = min(r[:, 0]), min(r[:, 1]), max(r[:, 0]), max(r[:, 1])

        l_mask, r_mask = masks

        # Get bounding box of the left eye in the image
        l_sub = image[l_y_min-15:l_y_max+15,l_x_min-15:l_x_max+15]
        l_image_box = cv2.cvtColor(l_sub, cv2.COLOR_BGR2GRAY)



        _, l_image = cv2.threshold(l_image_box, self.threshold, 255, cv2.THRESH_BINARY)
        l_image = cv2.erode(l_image, None, iterations=2)  # 1
        l_image = cv2.dilate(l_image, None, iterations=4)  # 2
        l_image = cv2.medianBlur(l_image, 9)  # 3Q

        keypoints = self.detector.detect(l_image)
        cv2.imshow('r_eye', l_image)
        l_image = cv2.drawKeypoints(l_image_box, keypoints, l_image_box, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Get bounding box of right eye in the image
        r_sub = image[r_y_min-5:r_y_max+5, r_x_min:r_x_max]
        r_image = cv2.cvtColor(r_sub, cv2.COLOR_BGR2GRAY)

        lx, ly, c = l_image.shape
        resized_image = cv2.resize(l_image, (ly*6,lx*6))
        #cv2.imshow('eye_tracking', image)



        eye_frame = cv2.bitwise_and(image, image, mask=(l_mask + r_mask))
        return resized_image
        # cv2.imshow('eye', gray_frame)
