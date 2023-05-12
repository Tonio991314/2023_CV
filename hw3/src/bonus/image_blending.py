import numpy as np
import cv2
from image_feature import *


def image_blending(img1, img2, shift):
    dy, dx = shift
    
    # Swap images and negate shift if dx < 0
    if dx < 0:
        img1, img2 = img2, img1
        dx = -dx
        dy = -dy

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Calculate occlusion region and create blank image
    occlusion = w2 - dx
    combined_h = max(h1, h2) + abs(dy)
    combined_w = w1 + dx
    output_image = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    
    # Combine images
    if dy > 0:
        output_image[:h1, :w1-occlusion] = img1[:h1, :w1-occlusion] # paste img1 on the left
        output_image[dy:h2+dy, w1:] = img2[:h2, w2-dx:] # paste img2 on the right
        for i in range(occlusion):
            alpha1 = (occlusion - i) / occlusion
            alpha2 = 1 - alpha1
            output_image[:h1, w1-occlusion+i] += (img1[:h1, w1-occlusion+i] * alpha1).astype(np.uint8)
            output_image[dy:h2+dy, w1-occlusion+i] += (img2[:h2, i] * alpha2).astype(np.uint8)
    else:
        dy = -dy
        output_image[dy:h1+dy, :w1-occlusion] = img1[:h1, :w1-occlusion] # paste img1 on the left
        output_image[:h2, w1:] = img2[:h2, w2-dx:] # paste img2 on the right
        for i in range(occlusion):
            alpha1 = (occlusion - i) / occlusion
            alpha2 = 1 - alpha1
            
            output_image[dy:h1+dy, w1-occlusion+i] += (img1[:h1, w1-occlusion+i] * alpha1).astype(np.uint8)
            output_image[:h2, w1-occlusion+i] += (img2[:h2, i] * alpha2).astype(np.uint8)
    
    return output_image



