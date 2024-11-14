import cv2

from facial_recognition import FacialRecognition
from logo_recognition import LogoRecognition
from text_recognition import TextRecognition

class CameraFeed:

    def __init__(self):
        self.facial_recognition = FacialRecognition()
        self.logo_recognition = LogoRecognition()
        self.logo_recognition.load_model()
        self.text_recognition = TextRecognition()

        self.camera = cv2.VideoCapture(0)
    
    def getCameraFeed(self):
        ''' Show the camera and have it display in a window. '''

        if self.camera.isOpened():
            while True:
                ret, frame = self.camera.read()
                if ret:
                    cv2.imshow('Christian Bloemhof Camera Feed', frame)
                    
                    # Check for words on the screen
                    self.text_recognition.check_for_text(frame)

                    # Check to see if Christian is in frame
                    self.facial_recognition.check_for_face(frame)

                    # Check for a trained logo
                    self.logo_recognition.detect_logo(frame)

                    # Press the space bar to exit
                    if cv2.waitKey(1) & 0xFF == ord(' '):
                        return
                else:
                    raise Exception('Could not read frame.')
        else:
            raise Exception('Camera not opened')
    
    def close(self):
        ''' Clean up and close down all camera operations '''

        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows()
