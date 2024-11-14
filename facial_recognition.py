import face_recognition
import os

import cv2

FRAME_COUNT = 50

class FacialRecognition:
    def __init__(self):
        self.encodings_of_christian = []
        self.see_christian = False
        self.frame_count = 0
        
        self.load_images()
    
    def load_images(self):
        image_files = os.listdir('images_of_christian')

        for file in image_files:
            image = face_recognition.load_image_file(f'images_of_christian/{file}')
            encoding = face_recognition.face_encodings(image)
            if encoding:
                self.encodings_of_christian.append(encoding[0])
            else:
                print(f'Cannot get encoding for file: {file}')

    def test(self):
        camera = cv2.VideoCapture(0)
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            self.check_for_face(frame)
            
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

            self.frame_count = (self.frame_count + 1) % FRAME_COUNT
    

    def check_for_face(self, frame):
        # Only check every few frames to reduce camera lag
        if self.frame_count == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            if face_encodings:
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.encodings_of_christian, face_encoding)
                    if True in matches: # If we can see Christian
                        if not self.see_christian:
                            print('is Christian Bloemhof!')
                            self.see_christian = True
                    else: # If we see someone other than Christian
                        if self.see_christian:
                            print("I don't think that's Christian...")
                            self.see_christian = False
            else: # If we see nobody in frame
                if self.see_christian:
                    print("Can no longer see anyone")
                    self.see_christian = False

        self.frame_count = (self.frame_count + 1) % FRAME_COUNT