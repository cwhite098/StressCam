import cv2
import numpy as np


class Eye:
    def __init__(self):
        self.tracker = 0
        self.circles = None
        self.saved_prev = None
        self.coords = []

    def get_boxes(self):
        """
        Find the bounding box given an eye mask
        :return: the coordinates of the boxes
        """
        x_min, y_min, x_max, y_max = min(self.coords[:, 0]), min(self.coords[:, 1]), max(self.coords[:, 0]), max(
            self.coords[:, 1])
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

        self.l_eye = Eye()
        self.r_eye = Eye()

        self.saved_prev = None
        self.timer = 0

    def track_eyes(self, image, eyes):
        """
        Calculates the position of the eyes using simple image processing
        TODO: MAKE THE PARAMS PROPORTIONAL TO IMAGE SIZE
        :param image: Frame
        :param eyes: Eye mask coordinates
        :return: Frame with the circles plotted (will change this to the circles of the image)
        """

        self.l_eye.coords, self.r_eye.coords = eyes

        self.eye_list = [self.l_eye, self.r_eye]

        for eye in self.eye_list:
            x_min, y_min, x_max, y_max = eye.get_boxes()
            # Get bounding box around the eye
            # hard coded added extra bits (improves performance)

            eye_box = image[y_min - 15:y_max + 15, x_min - 15:x_max + 15]
            # eye_box = image[y_min:y_max, x_min:x_max]

            eye_width = x_max - x_min

            # Make image grayscale
            grey_eye_box = cv2.cvtColor(eye_box, cv2.COLOR_BGR2GRAY)

            # Find the circles, using Hough Transform. May have to adapt these params
            circ = cv2.HoughCircles(grey_eye_box, cv2.HOUGH_GRADIENT, 1, 20,
                                    param1=160, param2=25, minRadius=1, maxRadius=int(eye_width / 3))

            if circ is None:
                if eye.tracker < 20:
                    eye.circles = eye.saved_prev
                eye.tracker += 1
            else:
                eye.saved_prev = circ
                eye.tracker = 0

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
        for eye in self.eye_list:
            x_min, y_min, x_max, y_max = eye.get_boxes()
            try:
                circle = np.uint16(np.around(eye.circles))
                for i in circle[0, :]:
                    # draw the outer circle
                    cv2.circle(image, (i[0] + x_min - 15, i[1] + y_min - 15), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv2.circle(image, (i[0] + x_min - 15, i[1] + y_min - 15), 2, (0, 0, 255), 3)
            except:
                pass

        return image
