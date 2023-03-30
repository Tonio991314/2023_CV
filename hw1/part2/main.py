import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    parser.add_argument('--save_path', default='./image', help='path to save image')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    setting=np.loadtxt(args.setting_path, dtype=np.str_)
    sigma_s = int(setting[-1].split(",")[1])
    sigma_r = float(setting[-1].split(",")[3])

    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_gt = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    
    img_gray_list=[]
    img_gray_list.append(img_gray)

    ## make 5 gray image from different R G B
    for idx in range(1, 6):
        img_gray_ = np.zeros_like(img_gray)
        R = float(setting[idx].split(",")[0])
        G = float(setting[idx].split(",")[1])
        B = float(setting[idx].split(",")[2])
        # print(f"R:{R} G:{G} B:{B}")

        for x in range(img_gray.shape[0]):
            for y in range(img_gray.shape[1]):
                # print(img_rgb[x][y][0] * R + img_rgb[x][y][1] * G + img_rgb[x][y][2] * B)
                img_gray_[x][y] = img_rgb[x][y][0] * R + img_rgb[x][y][1] * G + img_rgb[x][y][2] * B
        img_gray_list.append(img_gray_)


    ## find max and min cost
    min_jbf = np.zeros_like(img_rgb)
    max_jbf = np.zeros_like(img_rgb)

    min_cost = 100000000
    min_idx = 0
    max_cost = 0
    max_idx = 0
    for idx in range(len(img_gray_list)): ## (0,1,2,3,4,5)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray_list[idx]).astype(np.uint8)
        cost = np.sum(np.abs(jbf_out.astype('int32')-bf_gt.astype('int32')))
        print(f"gray image {idx} cost: {cost}")
        if cost < min_cost:
            min_cost = cost
            min_idx = idx
            min_jbf = jbf_out
            if idx == 0:
                RGB = "cv2 BGR to GRAY"
            else:
                RGB = setting[idx]
            print(f"current min cost is index {idx}, and its (R, G, B) is ({RGB})")
        if cost > max_cost:
            max_cost = cost
            max_idx = idx
            max_jbf = jbf_out
    
    ## plot min and max cost
    # image_num = args.image_path.split("/")[-1][:-4]
    # cv2.imwrite(f"{args.save_path}/filterRGB_img{image_num}_max_cost.png", cv2.cvtColor(max_jbf, cv2.COLOR_RGB2BGR)) ### filterRGB max cost 
    # cv2.imwrite(f"{args.save_path}/filterRGB_img{image_num}_min_cost.png", cv2.cvtColor(min_jbf, cv2.COLOR_RGB2BGR)) ### filterRGB min cost
    # cv2.imwrite(f"{args.save_path}/grayscale_img{image_num}_max_cost.png", img_gray_list[max_idx]) ### grayscale max cost
    # cv2.imwrite(f"{args.save_path}/grayscale_img{image_num}_min_cost.png", img_gray_list[min_idx]) ### grayscale min cost

if __name__ == '__main__':
    main()