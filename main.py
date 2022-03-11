import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

image = cv2.imread('workout.jpg')
cv2.imshow('image',image)
cv2.waitKey(0)

# init model
with mp_holistic.Holistic(
    static_image_mode=True, model_complexity=2, enable_segmentation=True, refine_face_landmarks=True) as holistic: 
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_recolored = holistic.process(image)

    mp_drawing.draw_landmarks(image, image_recolored.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, image_recolored.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, image_recolored.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, image_recolored.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


    cv2.imshow('image2',image)
    cv2.waitKey(0)