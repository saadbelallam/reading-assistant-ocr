#!/usr/bin/env python3
import cv2
import os
import sys
from datetime import datetime

def capture_and_process(output_dir):
    # Create webcam directory if it doesn't exist
    webcam_dir = os.path.join(output_dir, "webcam")
    os.makedirs(webcam_dir, exist_ok=True)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("Webcam started. Press 'x' to capture an image, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the frame
        cv2.imshow("Webcam - Press 'x' to Capture", frame)

        # Wait for keypress
        key = cv2.waitKey(1) & 0xFF

        if key == ord('x'):  # Capture image on 'x'
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(webcam_dir, f"{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
            return image_path  # Return path for Makefile processing

        if key == ord('q'):  # Quit on 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: webcam_capture.py <output_directory>")
        sys.exit(1)

    output_dir = sys.argv[1]
    captured_image = capture_and_process(output_dir)

    if captured_image:
        print(f"Captured image ready for processing: {captured_image}")
    else:
        print("No image captured.")