from heart_rate import Heart_Rate_Monitor
import mediapipe as mp
import cv2
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection


cap = cv2.VideoCapture()
# The device number might be 0 or 1 depending on the device and the webcam
cap.open(0, cv2.CAP_DSHOW)

realWidth = 640
realHeight = 480

cap.set(3, realWidth);
cap.set(4, realHeight);

HRM = Heart_Rate_Monitor()

while True:
	# Read the output from the webcam
	ret, image = cap.read()
	# init model
	with mp_face.FaceDetection(
		model_selection=0, min_detection_confidence=0.5) as face_detection: 

		# Change the colours (not sure why)
		#image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image_hight, image_width, _ = image.shape
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
				print(rel_x_min)


		frame, bpm = HRM.get_bpm(image, (int(np.ceil(rel_y_min*realWidth)), int(np.ceil(rel_x_min*realHeight))), 
										(int(rel_height*realHeight), int(rel_width*realWidth)))


		cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()