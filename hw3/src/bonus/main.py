import os
import cv2
import numpy as np
import argparse

from utils import *
from utils import get_proj_images_and_focals
from image_blending import *
from image_feature import *

parser = argparse.ArgumentParser(description='Add parameter')
parser.add_argument('-f', '--folder', default="data/")
parser.add_argument('-r', '--reverse', default=True)  # left: true, right: false
parser.add_argument('-t', '--type', default="harris")
parser.add_argument('-fo', '--focal', default=False)

args = parser.parse_args()

if __name__ == "__main__":

    print("[STEP 1] Read images and focals ...")
    proj_images, focals = get_proj_images_and_focals(args.folder, args.reverse, args.focal)

    result = proj_images[0]
    for idx in range(len(proj_images)-1):
        
        img1 = proj_images[idx]   
        img2 = proj_images[idx+1]
        
        print(f"[STEP 2_{idx}] image matching with {args.type} ...")
        matches = image_match_with_ransac(img1, img2, threshold=0.8, Type=args.type)
        # print(f"   matches: {matches}")

        print(f"[STEP 2_{idx}] combine image ...")
        result = image_blending(result, img2, matches)

        cv2.imwrite(f'{args.folder}/blend_{args.type}.png', result)

    print(f"[STEP 3] align and crop image ...")
    matches_fl = image_match_with_ransac(proj_images[0], proj_images[-1], threshold=0.8, Type=args.type)
    result = alignment_image(result, matches_fl)
    cv2.imwrite(f'{args.folder}/blend_{args.type}_align.png', result)
    result = crop_image(result)
    cv2.imwrite(f'{args.folder}/result_{args.type}.png', result)

    print("[DONE] All finish!!")
