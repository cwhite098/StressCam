from methods.blink_mouth_detection import Eyes_Mouth_Detector
from methods.heart_rate import Heart_Rate_Monitor
from methods.head_tracker import Head_Tracker
from methods.eye_tracking import EyeTracker
from methods.respiratory_tracking import Resp_Rate
import mediapipe as mp
import cv2
import numpy as np
from stroop.stroop import *
import time
from methods.utils import get_hull, nothing

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

fps = 15
cap = cv2.VideoCapture()
# The device number might be 0 or 1 depending on the device and the webcam
cap.open(0, cv2.CAP_ANY)    # CAP_ANY instead of CAP_DSHOW to work on mac
cap.set(cv2.CAP_PROP_FPS, fps)


realWidth = 1280
realHeight = 720

# cap.set(3, realWidth)
# cap.set(4, realHeight)

boxWidth = 50
boxHeight = 60

cap.set(3, realWidth)
cap.set(4, realHeight)

# Record frame times for fps counter
prev_frame_time = 0
new_frame_time = 0

# Points
leye_idx = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
reye_idx = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
mouth_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
face_top_idx = [243, 244, 245, 122, 6, 351, 465, 464, 463, 112, 26, 22, 23, 24, 110, 25, 226, 35, 143, 34,
                127, 341, 256, 252, 253, 254, 339, 255, 446, 265, 372, 264, 356, 389, 251, 284, 332, 297, 338, 10, 109,
                67, 103, 54, 21, 162]

HRM = Heart_Rate_Monitor(fps, realWidth, realHeight, show_plots=True, process_signal=True)
BD = Eyes_Mouth_Detector(show_plots=True)
HT = Head_Tracker(realWidth, realHeight, show_plots=True)
ET = EyeTracker()
RR = Resp_Rate(cap=cap, fps=fps)

cv2.namedWindow("Display_Image", cv2.WINDOW_NORMAL)
cv2.namedWindow('HRM frame', cv2.WINDOW_NORMAL)
# st = stroop_test()


# init model
RR.live_feed_init()
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                           refine_landmarks=True, min_detection_confidence=0.5) as face_detection:
    while True:

        # Read the output from the webcam
        ret, image = cap.read()
        face_mask = np.zeros((realHeight, realWidth), dtype=np.uint8)
        leye_mask = np.zeros((realHeight, realWidth), dtype=np.uint8)
        reye_mask = np.zeros((realHeight, realWidth), dtype=np.uint8)
        mouth_mask = np.zeros((realHeight, realWidth), dtype=np.uint8)
        face_top_mask = np.zeros((realHeight, realWidth), dtype=np.uint8)

        # set to false before processing - apparently increases performance
        image.flags.writeable = False
        # Detect the face
        results = face_detection.process(image)
        image.flags.writeable = True

        if results.multi_face_landmarks and ret:
            for detection in results.multi_face_landmarks:

                # Draws the mesh over the face
                '''
				mp_drawing.draw_landmarks(image=image, landmark_list=detection,
										connections=mp_face_mesh.FACEMESH_TESSELATION,
										landmark_drawing_spec=None,
										connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
				'''
                # Extract the landmarks from the mesh and get the coords
                total_landmarks = []
                total_landmarks_3d = []
                for lm in detection.landmark:
                    point = [lm.x * realWidth, lm.y * realHeight]
                    total_landmarks.append(point)
                    total_landmarks_3d.append([lm.x * realWidth, lm.y * realHeight, lm.z * 3000])
                total_landmarks = np.array(total_landmarks, dtype=int)
                total_landmarks_3d = np.array(total_landmarks_3d, dtype=np.float64)

                total_face = np.array(get_hull(total_landmarks), dtype=int)

                left_eye = np.array(get_hull(total_landmarks[leye_idx]), dtype=int)
                right_eye = np.array(get_hull(total_landmarks[reye_idx]), dtype=int)

                mouth = np.array(get_hull(total_landmarks[mouth_idx]), dtype=int)
                top_face = np.array(get_hull(total_landmarks[face_top_idx]), dtype=int)
                cv2.drawContours(face_mask, [total_face], -1, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.drawContours(leye_mask, [left_eye], -1, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.drawContours(reye_mask, [right_eye], -1, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.drawContours(mouth_mask, [mouth], -1, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.drawContours(face_top_mask, [top_face], -1, (255, 255, 255), -1, cv2.LINE_AA)

                ROI_mask = face_mask - face_top_mask - mouth_mask

                # Use the mask to get the coloured region of the original frame for HRM
                ROI_colour = cv2.bitwise_and(image, image, mask=ROI_mask)

                # Add the eyes to the display
                # display_mask = ROI_mask + leye_mask + reye_mask
                display_mask = face_mask
                display_frame = cv2.bitwise_and(image, image, mask=display_mask)
        else:
            print('Frame failed, quitting...')
            break

        frame = HRM.get_bpm(ROI_colour)
        BD.get_ratio(total_landmarks)
        l_eye, r_eye = ET.track_eyes(display_frame, [left_eye, right_eye])
        pointer_frame = HT.get_angular_position(detection.landmark, display_frame)
        display_frame = ET.draw_circles(display_frame)
        resp_frame = RR.live_feed_per_frame_loop(image, fps)

        # Finished processing, record frame time
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        #print(int(fps))

        #cv2.imshow('eye_tracking', eye_frame)
        cv2.imshow('HRM frame', frame)
        cv2.imshow('Display_Image', pointer_frame)
        cv2.imshow('Respiration frame', resp_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print('Detected Blinks: ', BD.blink_counter)
ET.get_history()
cap.release()
cv2.destroyAllWindows()
HRM.save_data('HRM_data.csv')
