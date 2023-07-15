import cv2
import numpy as np
img = cv2.imread('images/9.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res = np.zeros(gray.shape).astype('uint8')
gray = cv2.GaussianBlur(gray, (5,5), 0)
kernel = np.ones((5,9),np.uint8)
th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
contours = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
cnt = sorted(contours, key=lambda cnt:cv2.contourArea(cnt))[-2]
cv2.drawContours(res, [cnt], -1, 255, -1)
cv2.drawContours(img, [cnt], -1, (0,255,0), 2)
# cv2.imshow('th', th)
cv2.imshow('d', res)
cv2.imshow('src', img)
cv2.waitKey(0)
# import cv2
# from PIL import Image
# import numpy as np

# image = Image.open('10000.jpg')

# new_image = image.resize((768,1024))
# new_image.save('10001.jpg')

# # Load and preprocess the image
# image = cv2.imread('5.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # cv2.imshow('gray', gray)
# # cv2.waitKey(0)


# # Perform edge detection
# edges = cv2.Canny(gray, threshold1=5, threshold2=13)
# # Thresholding
# _, binary_mask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)

# cv2.imshow('binary_mask', binary_mask)
# cv2.imshow('edges', edges)
# cv2.waitKey(0)


# # Morphological operations (Optional)
# kernel = np.ones((5, 5), np.uint8)
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

# # Find the largest connected component
# contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# largest_contour = max(contours, key=cv2.contourArea)

# # Create the clothing mask
# clothing_mask = np.zeros_like(binary_mask)
# cv2.drawContours(clothing_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

# kernel = cv2.getStructuringElement(cv2.MORPH_, (5, 5))  # Define the structuring element
# smoothed_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel)
# # Display the clothing mask
# cv2.imshow('Clothing Mask', clothing_mask)
# cv2.imshow("Smoothed_mask", smoothed_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

