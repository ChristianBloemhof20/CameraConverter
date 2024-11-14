import cv2, pytesseract
import nltk
from nltk.corpus import words

FRAME_COUNT = 8

class TextRecognition:
    def __init__(self):
        nltk.download('words')
        self.english_words = set(words.words())
        self.english_words.update(['christian', 'bloemhof'])

        self.previous_word = ''
        self.frame_count = 0

        # Knowing what words I'm looking for, don't include any others
        self.noise_reducer = ['hi', 'my', 'name', 'i', 'like', 'hire?', 'hire', 'not', 'thank', 'you']

    def check_for_text(self, frame):
        if self.frame_count == 0:
            text = self.process_frame(frame)
            if text and text != self.previous_word:
                print(text)
                self.previous_word = text
        
        self.frame_count = (self.frame_count + 1) % FRAME_COUNT

    def process_frame(self, frame):
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding (inverse to make text white on black background)
        adaptive_thresh = cv2.adaptiveThreshold(
            grey_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )

        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_image = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

        final_image = cv2.medianBlur(cleaned_image, 5)

        # OCR with specific configuration
        custom_config = r'--oem 3 --psm 7'
        text = pytesseract.image_to_string(final_image, config=custom_config)

        # cv2.imshow("Grey frame", grey_frame)
        # cv2.imshow("Thresholded Image", adaptive_thresh)
        # cv2.imshow("Cleaned Image", cleaned_image)
        cv2.imshow("Final Image", final_image)
        
        valid_words = [word for word in text.split() if word.lower() in self.noise_reducer] 
        return ' '.join(valid_words)