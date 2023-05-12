import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping


random.seed(999)


def image_with_alpha(im1, im2, H):
    im2_copy = im2.copy()
    H_inv = np.linalg.inv(H)
    w_src = im1.shape[1]
    h_src = im1.shape[0]
    xmin, xmax, ymin, ymax = 0, im2.shape[1], 0, im2.shape[0]
    x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    im2_homo = np.vstack([x.flatten(), y.flatten(), np.ones(x.size)]).astype(np.int32)
    src_pixels = np.dot(H_inv, im2_homo)
    src_pixels /= src_pixels[2]
    src_pixels = np.round(src_pixels[:2].T).astype(np.int32)
    mask = (src_pixels[:, 0] >= 0) & (src_pixels[:, 1] >= 0) & (src_pixels[:, 0] < w_src) & (src_pixels[:, 1] < h_src)
    src_pixels_masked = src_pixels[mask]
    im2_homo_masked = im2_homo[:, mask]

    min_val = src_pixels_masked[:,0].min()
    max_val = src_pixels_masked[:,0].max()

    alpha = (src_pixels_masked[:,0] - min_val) / (max_val - min_val)
    alpha = alpha.reshape((-1, 1))

    dst_y, dst_x = im2_homo_masked[1, :], im2_homo_masked[0, :]
    src_y, src_x = src_pixels_masked[:, 1], src_pixels_masked[:, 0]

    im2_copy[dst_y, dst_x] = (1-alpha) * im2[dst_y, dst_x] + alpha * im1[src_y, src_x]
    
    return im2_copy
    

def panorama(imgs, alpha):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = dst.copy()
    
    w_start = 0
    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        w_start += im1.shape[1]  

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        us = []
        vs = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                us.append(kp1[m.queryIdx].pt)
                vs.append(kp2[m.trainIdx].pt)
        us = np.float32(us)
        vs = np.float32(vs)


        # TODO: 2. apply RANSAC to choose best H
        ## RANSAC parameters
        num_iterations = 10000
        threshold = 3
        inlineNmax = 20
        best_H = np.eye(3)

        ## iterate
        for _ in range(0, num_iterations):
            random_indices = np.random.choice(len(us), size=4, replace=False)
            u_rand = us[random_indices]
            v_rand = vs[random_indices]
            H = solve_homography(v_rand, u_rand)

            # compute inliers using threshold
            M = np.concatenate([np.transpose(vs), np.ones((1, len(vs)))], axis=0)
            W = np.concatenate([np.transpose(us), np.ones((1, len(us)))], axis=0)
            HM = np.dot(H, M)
            HM = HM / HM[-1]
            error = np.linalg.norm(HM[:-1] - W[:-1], axis=0)
            inliers = error < threshold
            num_inliers = np.count_nonzero(inliers)

            # update best H if more inliers are found
            if num_inliers > inlineNmax:
                inlineNmax = num_inliers
                best_H = H

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_H)
        
        # TODO: 4. apply warping and alpha
        ## create alpha im2
        if alpha:
            im2_copy = image_with_alpha(im1, im2, best_H)
        else:
            im2_copy = im2.copy()
        out = warping(im2_copy, out, last_best_H, 0, im2_copy.shape[0], w_start, w_start+im2_copy.shape[1], direction='b') 
        
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs, alpha=False)
    cv2.imwrite('output4.png', output4)
