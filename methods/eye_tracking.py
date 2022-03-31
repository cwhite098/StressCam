import cv2
import numpy as np
import pandas as pd


class Eye:
    def __init__(self, name):
        self.tracker = 0
        self.circles = None
        self.threshold = 25
        self.saved_prev = None
        self.coords = []
        self.name = name
        self.history = []

    def get_boxes(self):
        """
        Find the bounding box given an eye mask
        :return: the coordinates of the boxes
        """
        x_min, y_min, x_max, y_max = min(self.coords[:, 0]), min(self.coords[:, 1]), \
                                     max(self.coords[:, 0]), max(self.coords[:, 1])
        return x_min, y_min, x_max, y_max


class EyeTracker:
    def __init__(self):

        # PARAMETERS FOR OLDER MODEL

        # detector_params = cv2.SimpleBlobDetector_Params()
        # detector_params.filterByArea = True
        # detector_params.maxArea = 600
        # detector_params.minArea = 100
        #
        # # Filter by Circularity
        # detector_params.filterByCircularity = True
        # detector_params.minCircularity = 0.4
        #
        # # Filter by Convexity
        # detector_params.filterByConvexity = True
        # detector_params.minConvexity = 0.5
        # # Filter by Inertia
        # detector_params.filterByInertia = True
        # detector_params.minInertiaRatio = 0.1
        #
        # self.detector = cv2.SimpleBlobDetector_create(detector_params)

        self.l_eye = Eye('left eye')
        self.r_eye = Eye('right eye')
        self.threshold = 20
        self.saved_prev = None
        self.timer = 0
        self.data = []

    def track_eyes(self, image, eyes):
        """
        Calculates the position of the eyes using simple image processing
        TODO: MAKE THE PARAMS PROPORTIONAL TO IMAGE SIZE
        :param image: Frame
        :param eyes: Eye mask coordinates
        :return: Circle coordinates relative to the eye mask and radius for the frame.
        """

        self.l_eye.coords, self.r_eye.coords = eyes
        outs = [None, None]
        self.eye_list = [self.l_eye, self.r_eye]

        for idx, eye in enumerate(self.eye_list):
            x_min, y_min, x_max, y_max = eye.get_boxes()
            # Get bounding box around the eye
            # hard coded added extra bits (improves performance)

            eye_box = image[y_min - 15:y_max + 15, x_min - 15:x_max + 15]
            # eye_box = image[y_min:y_max, x_min:x_max]

            eye_width = x_max - x_min

            # Make image grayscale
            try:
                grey_eye_box = cv2.cvtColor(eye_box, cv2.COLOR_BGR2GRAY)

                # Find the circles, using Hough Transform. May have to adapt these params
                circ = cv2.HoughCircles(grey_eye_box, cv2.HOUGH_GRADIENT, 1, 20,
                                        param1=np.mean(grey_eye_box), param2=eye.threshold,
                                        minRadius=int(eye_width / 5),
                                        maxRadius=int(eye_width / 4))
            except:
                circ = None

            # Assigning value to the circle variable
            # Retrieve previous value of circle if there is no circle found

            if circ is None:
                if eye.tracker < 20:
                    eye.circles = eye.saved_prev
                # Slowly reduce the tolerance/threshold if none are found
                elif eye.tracker % 5 == 0:
                    eye.threshold -= 1
                eye.tracker += 1

            # Discard result if multiple are found, increase threshold
            elif len(circ[0]) != 1:
                print(circ)
                if eye.tracker < 20:
                    eye.circles = eye.saved_prev
                eye.tracker += 1
                eye.threshold += 1

            # Single circle found, reset timer and threshold
            else:
                eye.circles = circ
                eye.threshold = 20
                eye.saved_prev = circ
                eye.tracker = 0

            # Save value in eye history list and list for outputs

            if eye.circles is not None:

                # Find midpoints of eye box
                x_mid, y_mid = (x_max - x_min) / 2, (y_max - y_min) / 2
                # 15 is added to box for better circle detection, so remove this when scaling
                x = eye.circles[0][:][:][0][0] - 15
                y = eye.circles[0][:][:][0][1] - 15
                # Scale position between -1,1, (0,0) is centre.
                x, y = 2 * (x - x_mid) / (x_max - x_min), 2 * (y_mid - y) / (y_max - y_min)

                eye_coords = np.array([x, y])
                eye.history.append(eye_coords)
                outs[idx] = eye_coords
            else:
                eye.history.append([np.nan, np.nan])

        return outs
        # ================= Find the keys using blob detection points (OLD, DIDN'T REALLY WORK) ==================
        #
        # blur = cv2.medianBlur(grey_eye_box, 3)
        # _, p_image = cv2.threshold(blur, self.threshold, 255, cv2.THRESH_BINARY)
        # ret3, p_image = cv2.threshold(grey_eye_box, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #
        # p_image = cv2.erode(p_image, None, iterations=1)  # 1
        #
        # p_image = cv2.dilate(p_image, None, iterations=2)  # 2
        #
        # p_image = cv2.medianBlur(p_image, 5)  # 3
        # grey_eye_box = cv2.cvtColor(grey_eye_box, cv2.COLOR_GRAY2RGB)
        #
        # p_image = cv2.drawKeypoints(grey_eye_box, keypoints, grey_eye_box, (0, 0, 255),
        #                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # keypoints = self.detector.detect(p_image)
        #
        # ========================================================================================================

    def draw_circles(self, image):
        """
        Draws the circles found from the eye tracking method on the input fram
        :param image: Input frame
        :return: A frame with circles drawn on the eyes.
        """
        for eye in self.eye_list:
            if eye.circles is not None:
                circle = np.uint16(np.around(eye.circles))
                x_min, y_min, x_max, y_max = eye.get_boxes()
                for i in circle[0, :]:
                    # draw the outer circle
                    cv2.circle(image, (i[0] + x_min - 15, i[1] + y_min - 15), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv2.circle(image, (i[0] + x_min - 15, i[1] + y_min - 15), 2, (0, 0, 255), 3)

        return image

    def get_history(self):
        """
        Assigns the circle data to the data attribute.
        :return: Data attribute
        """
        self.data = [self.l_eye.history, self.r_eye.history]
        return self.data
