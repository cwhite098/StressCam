import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture()
# The device number might be 0 or 1 depending on the device and the webcam
cap.open(0, cv2.CAP_DSHOW)

while(True):
    ret, frame = cap.read()

    with mp_holistic.Holistic(
            static_image_mode=False, model_complexity=1, enable_segmentation=False,
            refine_face_landmarks=False) as holistic:

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_recolored = holistic.process(image)

        mp_drawing.draw_landmarks(image, image_recolored.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, image_recolored.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, image_recolored.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, image_recolored.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()