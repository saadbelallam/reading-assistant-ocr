"""
Test automatrisé (a revoir !!!!!)
"""

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
# from textblob import TextBlob
import numpy as np
import os
import cv2
import torch


from PIL import Image
import pytesseract  # https://github.com/madmaze/pytesseract
from gtts import gTTS
import pygame
from correction import main

torch.cuda.empty_cache()

# # Path configuration
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


correction = input("Voulez-vous corriger les images? (oui/non) : ")

while True :
    if correction == "oui":
        main()
        raw_dir = "../test/processed"
        break
    elif correction == "non":
        raw_dir = "../test/images"  # Images de test
        break
    else :
        correction = input("Voulez-vous corriger les images? (oui/non) : ")



# Directories for raw images and results
results_dir = "../test/results/" # Resultats .txt et .mp3 des tests

# Ensure results directory exists
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load the ResNet50 model
model = ResNet50(weights='imagenet')

def process_images():
    """
    Process all images in the raw directory, perform OCR, and save results in the results directory.
    """
    for file_name in os.listdir(raw_dir):
        file_path = os.path.join(raw_dir, file_name)

        # Check if it's an image file
        if not (file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg')):
            print(f"Skipping non-image file: {file_name}")
            continue

        print(f"Processing: {file_name}")

        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error reading image: {file_name}")
            continue

        # Resize for ResNet50 input
        img_resized = cv2.resize(image, (224, 224))
        resnet_img = img_to_array(img_resized)
        resnet_img = np.expand_dims(resnet_img, axis=0)
        resnet_img = preprocess_input(resnet_img)

        # OCR to extract text
        extracted_text = pytesseract.image_to_string(image, lang="fra").strip()
        # extracted_text  = str( TextBlob(extracted_text).correct())
        # NN prediction (optional)
        # preds = model.predict(resnet_img)

        # Save extracted text to results directory
        result_file_path = os.path.join(results_dir, f"{os.path.splitext(file_name)[0]}_result.txt")
        with open(result_file_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        print(f"Result saved to: {result_file_path}")

        # si le text est = null, on ne genere pas d audio 
        if extracted_text:
            generate_audio(extracted_text, file_name)

def generate_audio(text, file_name):
    """
    Generate an audio file for the extracted text.
    """
    audio_path = os.path.join(results_dir, f"{os.path.splitext(file_name)[0]}.mp3")
    myobj = gTTS(text=text, lang="fr", slow=False)
    myobj.save(audio_path)
    print(f"Audio saved to: {audio_path}")

# Main entry point
if __name__ == "__main__":
    process_images()