import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

image = cv2.imread('face.jpg')
cv2.imshow('image',image)
cv2.waitKey(0)

# init model
with mp_holistic.Holistic(
    static_image_mode=True, model_complexity=1, enable_segmentation=True, refine_face_landmarks=True) as holistic: 
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_recolored = holistic.process(image)

    mp_drawing.draw_landmarks(image, image_recolored.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, image_recolored.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, image_recolored.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, image_recolored.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow('image2',image)
    cv2.waitKey(0)


with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:

    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_hight, image_width, _ = image.shape

    print(
      f'Nose coordinates: ('
      f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
      f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight}, '
      f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z})'
    )

    print(
      f'Shoulder coordinates: ('
      f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width}, '
      f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_hight}, '
      f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z})'
    )