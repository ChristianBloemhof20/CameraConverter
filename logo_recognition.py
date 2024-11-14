import tensorflow as tf
import cv2
import numpy as np

FRAME_COUNT = 50

class LogoRecognition:
    def __init__(self):
        self.frame_count = 0
        self.spotted_logo = 3
    
    def train_model(self):
        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        self.model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )

        self.model.fit(train_generator, epochs=5)
        self.model.save('logo_detector.keras')
    

    def detect_logo(self, frame):
        if self.frame_count == 0:
            img = cv2.resize(frame, (224, 224))
            img = np.expand_dims(img, axis=0) / 255.0

            prediction = self.model.predict(img, verbose=0)
            classification_val = np.argmax(prediction)
            confidence = np.max(prediction)
            if confidence >= 0.9:
                if classification_val == 0 and self.spotted_logo != 0: # Bad logo
                    print('Anime!')
                    self.spotted_logo = 0
                elif classification_val == 1 and self.spotted_logo != 1: # Clippers logo
                    print('Eww... not this one')
                    self.spotted_logo = 1
                elif classification_val == 2 and self.spotted_logo != 2: # Rams logo
                    print('The Los Angeles Clippers!')
                    self.spotted_logo = 2
                elif classification_val == 3:
                    print('My Nintendo Switch!')
                    self.spotted_logo = 3
                elif classification_val == 4:
                    print('Pokemon!')
                    self.spotted_logo = 4
                elif classification_val == 5:
                    print('The Log Angeles Rams!')
                    self.spotted_logo = 5
        
        self.frame_count = (self.frame_count + 1) % FRAME_COUNT
    
    def load_model(self):
        self.model = tf.keras.models.load_model('logo_detector.keras')
    
    def test_detection(self):
        self.frame_count = 0
        test_image = cv2.imread('train/rams_logo/la-rams-logo-1.png')
        logo_rec.detect_logo(test_image)

        self.frame_count = 0
        test_image = cv2.imread('train/clippers_logo/la-clips-logo-2.png')
        logo_rec.detect_logo(test_image)

        self.frame_count = 0
        test_image = cv2.imread('train/bad_logo/bad-logo-1.jpg')
        logo_rec.detect_logo(test_image)

        self.frame_count = 0
        test_image = cv2.imread('train/nintendo_switch/IMG_6107.jpeg')
        logo_rec.detect_logo(test_image)

        self.frame_count = 0
        test_image = cv2.imread('train/pokemon/IMG_6110.jpeg')
        logo_rec.detect_logo(test_image)

        self.frame_count = 0
        test_image = cv2.imread('train/anime/IMG_6108.jpeg')
        logo_rec.detect_logo(test_image)



if __name__ == '__main__':
    logo_rec = LogoRecognition()
    # logo_rec.train_model()
    logo_rec.load_model()
    logo_rec.test_detection()
    