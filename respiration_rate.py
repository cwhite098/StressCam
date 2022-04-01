from turtle import shape
import cv2 as cv
import numpy as np
from OpticalFlow import optical_flow
import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils
# mp_holistic = mp.solutions.holistic
  
cap = cv.VideoCapture()
# The device number might be 0 or 1 depending on the device and the webcam
cap.open(0, cv.CAP_DSHOW)

mp_holistic = mp.solutions.holistic

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (100, 3))

_, old_frame = cap.read()
height, width, _ = old_frame.shape
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

hsv = np.zeros_like(old_frame)
hsv[..., 1] = 255

Coor = []
for i in range(old_frame.shape[0]):
        for j in range(old_frame.shape[1]):
            Coor.append([i, j])
Coor = np.array(Coor)

# r = cv.selectROI(old_frame)
# imCrop = old_frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


while True:
    # Capture frame-by-frame
    # ret, frame = cap.read()
    _ , cur_frame = cap.read()
    frame_gray = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)
    # width = cap.get(cv.CAP_PROP_FRAME_WIDTH )
    # height = cap.get(cv.CAP_PROP_FRAME_HEIGHT )

    # Optical flow vectors
    # V = optical_flow(gray, gray2, width)

    # calculate optical flow
    with mp_holistic.Holistic(
            static_image_mode=False, model_complexity=0, enable_segmentation=False,
            refine_face_landmarks=False) as holistic:

        
        image = cv.cvtColor(cur_frame, cv.COLOR_BGR2RGB)
        hight, width, _ = image.shape
        image.flags.writeable = False
        image_recolored = holistic.process(image)

        # mp_drawing.draw_landmarks(image, image_recolored.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        p0 = [[image_recolored.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x*width,
        image_recolored.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y*height],
        [image_recolored.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x*width,
        image_recolored.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y*height]]

        print(
          f'Nose coordinates: ('
          f'{p0[0]}, '
          f'{p0[1]})'
          )
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, np.array(p0), None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(cur_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    # #Dense optical flow
    # flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # arrow_frame = cur_frame.copy()
    # for i in range(flow.shape[0]):
    #     for j in range(flow.shape[1]):
    #         pt = (i, j)
    #         arrow_frame = cv.arrowedLine(arrow_frame, (i, int(i+flow[i, j, 0])), (j, int(j+flow[i, j, 1])), (0, 0, 255))
    # cv.imshow('Image', arrow_frame)
    # old_gray = frame_gray.copy()


    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # # Display the resulting frame
    cv.imshow('frame', img)
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
