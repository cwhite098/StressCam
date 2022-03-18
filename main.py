
from heart_rate import Heart_Rate_Monitor
from stroop import stroop_test
import mediapipe as mp
import cv2
import numpy as np
from stroop import *
import time


mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection

fps=20
cap = cv2.VideoCapture()
# The device number might be 0 or 1 depending on the device and the webcam
cap.open(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, fps)

realWidth = 640
realHeight = 480

boxWidth = 50
boxHeight = 60

cap.set(3, realWidth);
cap.set(4, realHeight);

# Record frame times for fps counter
prev_frame_time = 0
new_frame_time = 0

HRM = Heart_Rate_Monitor(fps, boxWidth, boxHeight)
# st = stroop_test()
while True:

	# Read the output from the webcam
	ret, image = cap.read()

	# init model
	with mp_face.FaceDetection(
		model_selection=0, min_detection_confidence=0.5) as face_detection: 

		# Change the colours (not sure why)
		#image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image.flags.writeable = True

		# Detect the face
		results = face_detection.process(image)

		if results.detections:
			for detection in results.detections:
				# detection contains the coordinates
				# draw the detected face points
				mp_drawing.draw_detection(image, detection)
				# Get the height of the face, relative to the frame
				rel_height = detection.location_data.relative_bounding_box.height
				rel_width = detection.location_data.relative_bounding_box.width
				# Fraction in from the left (or the right when its flipped as it is here)
				rel_x_min = detection.location_data.relative_bounding_box.xmin
				# Position down the screen as a fraction of total height
				rel_y_min = detection.location_data.relative_bounding_box.ymin

				nose_coords = mp_face.get_key_point(detection, mp_face.FaceKeyPoint.NOSE_TIP)
				leye_coords = mp_face.get_key_point(detection, mp_face.FaceKeyPoint.LEFT_EYE)
				lear_coords = mp_face.get_key_point(detection, mp_face.FaceKeyPoint.LEFT_EAR_TRAGION)


				start_tuple = (int((leye_coords.y*realHeight+lear_coords.y*realHeight)/2) 
								,int(leye_coords.x*realWidth - (0.5*boxWidth)))


		frame = HRM.get_bpm(image, start_tuple, 
										(int(rel_height*realHeight), int(rel_width*realWidth)))
		

		# Finished processing, record frame time
		new_frame_time = time.time()
		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		print(int(fps))

		cv2.imshow('frame', image)

	if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()
HRM.save_data('HRM_data.csv')
