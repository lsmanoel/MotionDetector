# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import threading
import cv2

class MotionSense:
    def __init__(self,
                 video=None,
                 min_area=500):

        self.video = video

        if self.video is None:
            self.vs = VideoStream(src=0).start()
            time.sleep(2.0)

        # otherwise, we are reading from a video file
        else:
            self.vs = cv2.VideoCapture(self.video)

        self.min_area = min_area

        self._main_thread_list = []

    def run(self):
        main_thread = threading.Thread(target=self._main_loop)
        main_thread.start()
        self._main_thread_list.append(main_thread)

    def _main_loop(self):
        # initialize the first frame in the video stream
        firstFrame = None

        # loop over the frames of the video
        while True:
            # grab the current frame and initialize the occupied/unoccupied
            # text
            frame = self.vs.read()
            frame = frame if self.video is None else frame[1]
            text = "Unoccupied"

            # if the frame could not be grabbed, then we have reached the end
            # of the video
            if frame is None:
                break

            # resize the frame, convert it to grayscale, and blur it
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = gray
                continue

            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < self.min_area:
                    continue

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Occupied"
            
            # draw the text and timestamp on the frame
            cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # show the frame and record if the user presses a key
            
            win_dict ={
                "Security_Feed" : cv2.namedWindow("Security_Feed"),
                "Gray" : cv2.namedWindow("Gray"),
                "Thresh" : cv2.namedWindow("Thresh"),
                "Frame_Delta" : cv2.namedWindow("Frame_Delta")
            }

            cv2.moveWindow("Security_Feed", 900, 30)
            cv2.imshow("Security_Feed", frame)
            cv2.moveWindow("Gray",          1400, 30)
            cv2.imshow("Gray", gray)
            cv2.moveWindow("Thresh",        900, 500)
            cv2.imshow("Thresh", thresh)
            cv2.moveWindow("Frame_Delta",   1400, 500)
            cv2.imshow("Frame_Delta", frameDelta)

            key = cv2.waitKey(1) & 0xFF

            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break

        # cleanup the camera and close any open windows
        self.vs.stop() if self.video is None else self.vs.release()
        cv2.destroyAllWindows()


# test = MotionSense()
test = MotionSense(video="./video_dataset/walk_1.mp4")
test.run()