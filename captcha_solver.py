import cv2
import pytesseract
import numpy as np
from PIL import Image
import sys
import os

# Path to tesseract.exe (change this if installed in a different location)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """Preprocess the CAPTCHA image for better OCR accuracy"""
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding to remove noise
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Remove small noise using morphological operations
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Invert back (black text on white)
    result = 255 - clean

    # Save processed image instead of showing
    processed_path = "processed_captcha.png"
    cv2.imwrite(processed_path, result)
    print(f"Processed image saved as: {processed_path}")

    return result

def solve_captcha(image_path):
    """Read text from CAPTCHA image"""
    processed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(processed_image, config='--psm 8')
    return text.strip()

if __name__ == "__main__":
    # Default image path if none is provided
    if len(sys.argv) < 2:
        default_path = r"C:\Users\ADITYA\PycharmProjects\CAPTCHA\test1.jpg"
        print(f"No image path provided. Using default: {default_path}")
        img_path = default_path
    else:
        img_path = sys.argv[1]

    if not os.path.exists(img_path):
        print("Image file not found!")
        sys.exit(1)

    solved_text = solve_captcha(img_path)
    print("Predicted CAPTCHA:", solved_text)

