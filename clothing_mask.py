import cv2
import numpy as np
import json
from os import path as osp

import torch
from torch.utils import data


class image_mask(data.Dataset):
    def __init__(self, opt):
        self.width = opt.load_width
        self.height = opt.load_height
        self.dim = (self.width, self.height)

        # load clothing list        
        c_names = []
        with open(osp.join(opt.dataset_dir, opt.clothing_list), 'r') as f:
            for line in f.readlines():
                c_name = line 
                c_names.append(c_name)                

        self.c_names = c_names   
             

    def mask(self, opt):
        cloth = []
        for key in self.c_names:
            # img.append(cv2.imread(key))
            img = cv2.imread(osp.join(opt.clothing_dir, 'cloth', key))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, self.dim, interpolation=cv2.INTER_AREA)     

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            res = np.zeros(gray.shape).astype('uint8')
            gray = cv2.GaussianBlur(gray, (15,15), 0)
            kernel = np.ones((9,9),np.uint8)
            th = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            contours = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            cnt = sorted(contours, key=lambda cnt:cv2.contourArea(cnt))[-1]
            cloth.append(cv2.drawContours(res, [cnt], -1, 255, -1))
            cv2.drawContours(img, [cnt], -1, (0,255,0), 2)    

        self.clothes = cloth
       
    
    def save_mask(self, opt):
        folder_path = osp.join(opt.clothing_dir, 'cloth-mask')
        # if not osp.exists(folder_path):
        #     os.makedirs(folder_path)
        for i, image in enumerate(self.clothes):
            image_path = osp.join(folder_path, f'image_{i+1}.jpg')  # Construct the image path
            cv2.imwrite(image_path, image) 

        



# img = cv2.imread('0.jpg')

# width = 768
# height = 1024
# dim = (width, height)
# img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


# # # Filename
# # filename = '10001_00.jpg'
  
# # # Using cv2.imwrite() method
# # # Saving the image
# # cv2.imwrite(filename, img)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# res = np.zeros(gray.shape).astype('uint8')
# gray = cv2.GaussianBlur(gray, (15,15), 0)
# kernel = np.ones((9,9),np.uint8)
# th = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
# th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
# th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
# contours = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
# cnt = sorted(contours, key=lambda cnt:cv2.contourArea(cnt))[-1]
# cv2.drawContours(res, [cnt], -1, 255, -1)
# cv2.drawContours(img, [cnt], -1, (0,255,0), 2)
# # cv2.imshow('th', th)

# # Filename
# filename = 'savedImage.jpg'
  
# # Using cv2.imwrite() method
# # Saving the image
# cv2.imwrite(filename, res)

# cv2.imshow('d', res)
# cv2.imshow('src', img)
# cv2.waitKey(0)
# # import cv2
# # from PIL import Image
# # import numpy as np

# # image = Image.open('10000.jpg')

# # new_image = image.resize((768,1024))
# # new_image.save('10001.jpg')

# # # Load and preprocess the image
# # image = cv2.imread('5.jpg')
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # # cv2.imshow('gray', gray)
# # # cv2.waitKey(0)


# # # Perform edge detection
# # edges = cv2.Canny(gray, threshold1=5, threshold2=13)
# # # Thresholding
# # _, binary_mask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)

# # cv2.imshow('binary_mask', binary_mask)
# # cv2.imshow('edges', edges)
# # cv2.waitKey(0)


# # # Morphological operations (Optional)
# # kernel = np.ones((5, 5), np.uint8)
# # binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

# # # Find the largest connected component
# # contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # largest_contour = max(contours, key=cv2.contourArea)

# # # Create the clothing mask
# # clothing_mask = np.zeros_like(binary_mask)
# # cv2.drawContours(clothing_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

# # kernel = cv2.getStructuringElement(cv2.MORPH_, (5, 5))  # Define the structuring element
# # smoothed_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel)
# # # Display the clothing mask
# # cv2.imshow('Clothing Mask', clothing_mask)
# # cv2.imshow("Smoothed_mask", smoothed_mask)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

