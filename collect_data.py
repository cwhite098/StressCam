import cv2
from stroop.stroop import *
import multiprocessing
from threading import Timer
from os import path

global stop
stop = False

def start_stroop():
    st = StroopTest(sound=False)
    st.start_screen()

def break_loop(stroop_thread):
    global stop
    stop = True
    stroop_thread.terminate()


def main(name, stressed, timeout=60):
    fps = 15
    realWidth = 1280    # I think these need to be the same size as the webcam output
    realHeight = 720


    cap = cv2.VideoCapture()
    cap.open(0, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FPS, fps)

    cap.set(3, realWidth)
    cap.set(4, realHeight)

    # setting filetype to .avi with XVID codec works best
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    """Creating new filenames for videos in format '(name)_(video no.).avi'  """
    if stressed:
        stress = 1
    else:
        stress = 0

    i = 0
    while path.exists(f"data/videos/{name}_{stress}_{i}.avi"):
        i += 1
    filepath = f'data/videos/{name}_{stress}_{i}.avi'
    output = cv2.VideoWriter(filepath, fourcc, fps, (realWidth, realHeight))

    stroop_thread = multiprocessing.Process(target=start_stroop)
    stroop_thread.start()
    timer = Timer(timeout, break_loop, [stroop_thread])
    timer.start()

    while True:
        if stop:
            break
        ret, frame = cap.read()
        output.write(frame)
        cv2.imshow('Video feed', frame)

        # hold q to exit
        if cv2.waitKey(1) &0XFF == ord('q'):
            break

    cap.release()
    output.release()
    stroop_thread.terminate()

if __name__ == '__main__':
    """
        timeout is length of video to record
        changed stressed to True if playing high stress mode, false if not
        videos are saved as 'name_(0/1)_number.avi' with 0 for not stressed, 1 for stressed
    """

    main(name='finn', stressed=True, timeout=60) #