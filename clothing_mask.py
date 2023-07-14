import cv2
from PIL import Image
import numpy as np

image = Image.open('10000.jpg')

new_image = image.resize((768,1024))
new_image.save('10001.jpg')

# Load and preprocess the image
image = cv2.imread('5.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow('gray', gray)
# cv2.waitKey(0)


# Perform edge detection
edges = cv2.Canny(gray, threshold1=5, threshold2=13)
# Thresholding
_, binary_mask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)

cv2.imshow('binary_mask', binary_mask)
cv2.imshow('edges', edges)
cv2.waitKey(0)


# Morphological operations (Optional)
kernel = np.ones((5, 5), np.uint8)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

# Find the largest connected component
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

# Create the clothing mask
clothing_mask = np.zeros_like(binary_mask)
cv2.drawContours(clothing_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

kernel = cv2.getStructuringElement(cv2.MORPH_, (5, 5))  # Define the structuring element
smoothed_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel)
# Display the clothing mask
cv2.imshow('Clothing Mask', clothing_mask)
cv2.imshow("Smoothed_mask", smoothed_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

