import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian
import os

def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=3.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--save_path', default='./image', help='path to save image')

    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)

    # ### TODO ###
    DoG = Difference_of_Gaussian(threshold=args.threshold)
    keypoints = DoG.get_keypoints(image=img)
    
    image_name = f"{args.image_path.split('/')[1]}{args.image_path.split('/')[2][:-4]}"
    save_path = f"{args.save_path}/{image_name}_threshold_{args.threshold}.png"
    plot_keypoints(img, keypoints, save_path)

if __name__ == '__main__':
    main()
