import numpy as np
import math
import cv2
import os

def get_images_and_focals(folder, reverse=True, focal=False):
    
    image_folder = os.path.join(folder, "images")
    
    images = []
    focals = []
    focal_default = 1800

    if focal:
        focal_file = os.path.join(folder, f"{folder.split('/')[1]}.txt")
        with open(focal_file, 'r') as f:
            focal_lines = f.readlines()
            focal_lines = [line.strip() for line in focal_lines]    
    
    for filename in sorted(os.listdir(image_folder), reverse=reverse):
        # print(filename)
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(image_folder, filename))
            images.append(img)
            if focal:
                focals.append(np.float64(focal_lines[focal_lines.index(filename)+11]))
            else:
                focals.append(focal_default)
    return images, focals

def get_proj_images_and_focals(folder, reverse=True, focal=False):

    images, focals = get_images_and_focals(folder, reverse=reverse, focal=focal)
    
    h, w, c = images[0].shape
    proj_images = [np.empty((h, w, c), dtype=np.uint8) for _ in range(len(focals))]

    x_offset = int(w/2)
    y_offset = h-1

    for i in range(h):
        y = y_offset - i
        for j in range(w):
            x = j - x_offset
            for num, focal in enumerate(focals):
                x1 = int(focal*np.arctan(x/focal)) + x_offset
                y1 = y_offset - int(focal*y/math.sqrt(focal**2 + x**2))
                # print(f"testttt: {images[num][i, j]}")
                proj_images[num][y1, x1] = images[num][i, j]

    return proj_images, focals

def alignment_image(image, shift):
    shift_y = shift[0]
    width = image.shape[1]
    dy = np.linspace(0, shift_y, width, dtype=int)

    alignment_image = np.zeros_like(image)
    for w in range(width):
        alignment_image[:, w] = np.roll(image[:, w], -dy[w], axis=0)

    return alignment_image

def ransac(matches, iterCount=10000, threshold=2):
    maxInliner = -1
    for iter in range(iterCount):
        idx = np.random.randint(0, len(matches))
        matches = np.array(matches)
        shift_ = np.subtract(matches[idx][0], matches[idx][1])
        shifts = matches[:, 1] - matches[:, 0] + shift_
        inliner = 0
        for shift in shifts:
            y, x = shift
            if np.sqrt(x**2+y**2) < threshold:
                inliner += 1
        if inliner > maxInliner:
            maxInliner = inliner
            best_shift = tuple(shift_)
    return best_shift
 
def crop_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_new = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)

    h, w = img_new.shape
    lower = np.argmax(np.sum(img_new, axis=1) > 0.9*w)
    upper = h - np.argmax(np.flip(np.sum(img_new, axis=1), axis=0) > 0.9*w)

    return image[lower:upper, :]
