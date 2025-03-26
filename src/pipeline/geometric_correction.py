################
#    Step 1    #
################

import cv2
import numpy as np
import os
from imutils.perspective import four_point_transform

def scan_document(input_image_path, output_dir):
    print("Running geometric correction ...")
    os.makedirs(output_dir, exist_ok=True)
    
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Image not found: {input_image_path}")
    
    def process_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray)
        #print("its intensity : ", avg_intensity)
        # Determine thresholding method based on average intensity
        if avg_intensity > 215:  # Pre-scanned documents (bright background)
            _, thresh = cv2.threshold(gray, avg_intensity, 255, cv2.THRESH_BINARY)
        else:  # Non-scanned documents
            #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            #                              cv2.THRESH_BINARY, 11, 2)
            _, thresh = cv2.threshold(gray, min(avg_intensity, 100), 255, cv2.THRESH_BINARY)
        
        return thresh

    def detect_document(img):
        (h, w) = img.shape[:2]
        doc_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use Otsu's thresholding for better contour detection
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
                if area > max_area and len(approx) == 4:
                    doc_contour = approx
                    max_area = area
        return doc_contour

    doc_contour = detect_document(image)
    warped = four_point_transform(image, doc_contour.reshape(4, 2))
    
    processed = process_image(warped)
    processed = processed[10:processed.shape[0]-10, 10:processed.shape[1]-10]
    
    output_path = os.path.join(output_dir, "1_scanned.jpg")
    cv2.imwrite(output_path, processed)
    return output_path