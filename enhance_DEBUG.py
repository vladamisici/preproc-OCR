import cv2
import numpy as np
from PIL import Image

img_path = "output.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_contrast = clahe.apply(img)

img_binary = cv2.adaptiveThreshold(
    img_contrast, 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 
    15, 
    8
)

img_denoised = cv2.medianBlur(img_binary, 3)

kernel = np.ones((2, 2), np.uint8)
img_dilated = cv2.dilate(img_denoised, kernel, iterations=1)



cv2.imwrite("ocr_ready_binary.jpg", img_binary)
cv2.imwrite("ocr_ready_denoised.jpg", img_denoised)
cv2.imwrite("ocr_ready_dilated.jpg", img_dilated)

