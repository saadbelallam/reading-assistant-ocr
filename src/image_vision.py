
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

import cv2

from PIL import Image

import pytesseract # https://github.com/madmaze/pytesseract?tab=readme-ov-file
from gtts import gTTS
import pygame
import pyttsx3

# If you don't have tesseract executable in your PATH, include the following:
#pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Simple image to string


# In order to bypass the image conversions of pytesseract, just use relative or absolute image path
# NOTE: In this case you should provide tesseract supported images or tesseract will return error
#print(pytesseract.image_to_string('test.png'))




model = ResNet50(weights='imagenet')

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

cap = cv2.VideoCapture(0)

def go():
        while True:
                while True:
                        # Read from webcam
                        ret, frame = cap.read()
                        #print(pytesseract.image_to_string(frame))
                        
                        # Crop
                        frame = frame[:min(frame.shape[0:2]),:min(frame.shape[0:2])]
                        
                        # format image
                        img = cv2.resize(frame,(224,224))
                        resnetimg = image.img_to_array(img)
                        resnetimg = np.expand_dims(resnetimg, axis=0)
                        resnetimg = preprocess_input(resnetimg)
                        
                        # Write text on image
                        #cv2.putText(img,decode_predictions(preds, top=1)[0][0][1],(10,30), font, 
                        #fontScale,
                        #fontColor,
                        #lineType)
                        
                        # Show the image
                        cv2.imshow("frame",img)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                                # NN prediction
                                preds = model.predict(resnetimg)
                                text = pytesseract.image_to_string(frame,lang="fra")
                                print(text)
                                myobj = gTTS(text=text.replace("\n"," "), lang="fr", slow=False)
                                myobj.save("welcome.mp3")
                                pygame.mixer.init()
                                pygame.mixer.music.load("welcome.mp3")
                                pygame.mixer.music.play()
                                break


go()

