import numpy as np
import cv2

from scipy.spatial import KDTree
from scipy.ndimage import maximum_filter
from skimage.transform import FundamentalMatrixTransform
from skimage.measure import ransac
from scipy.spatial.distance import cdist
from utils import *
"""
Note:

This py file is for feature extraction with 3 parts:
1. Harris corner detection or SIFT (default: harris)--> get feature points
2. get feature descriptors
3. match feature points between two images

"""

def get_sift_keypoints(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    num_octaves = 2
    num_DoG_images_per_octave = 4
    num_guassian_images_per_octave = num_DoG_images_per_octave + 1
    sigma = 2**(1/4)
    threshold = 3

    gaussian_images = []
    for i in range(num_octaves): #(0, 1)
        octave=[]
        resize = True
        if i == 1 and resize == True: ### resize
            image = cv2.resize(gaussian_images[0][num_DoG_images_per_octave], (image.shape[1]//2, image.shape[0]//2),  interpolation=cv2.INTER_NEAREST)
            octave.append(image)
            resize = False
        elif i ==0 :
            octave.append(image)

        for j in range(num_guassian_images_per_octave-1): ###(0,1,2,3)
            img = cv2.GaussianBlur(image, (0,0), sigmaX=sigma**(j+1), sigmaY=sigma**(j+1))
            octave.append(img)
        
        gaussian_images.append(octave)

    dog_images = []
    for i in range(num_octaves): ### (0,1)
        octave=[]
        for j in range(num_DoG_images_per_octave): ### (0,1,2,3)
            dog = cv2.subtract(gaussian_images[i][j+1], gaussian_images[i][j])
            octave.append(dog)
        dog_images.append(octave)

    keypoints = []
    for i in range(num_octaves):  # (0,1)
        for j in range(1, num_DoG_images_per_octave-1): ## (1,2)
            prev_img, curr_img, next_img = dog_images[i][j-1], dog_images[i][j], dog_images[i][j+1]
            h, w = curr_img.shape[:2]
            for x in range(1, h-1):
                for y in range(1, w-1):
                    patch = np.stack([
                        prev_img[x-1:x+2, y-1:y+2],
                        curr_img[x-1:x+2, y-1:y+2],
                        next_img[x-1:x+2, y-1:y+2]
                    ])
                    if np.isnan(patch[1, 1, 1]).all()==False and np.count_nonzero(patch == patch[1, 1, 1])==1:
                        center_value = patch[1, 1, 1]
                        if center_value == np.nanmax(patch) or center_value == np.nanmin(patch):
                            if abs(center_value) >= threshold:
                                if i == 0:
                                    keypoints.append(((x, y)))
                                else:
                                    keypoints.append((2*x, 2*y))
    
    keypoints = np.unique(keypoints, axis=0)
    # print(f"keypoints: {keypoints}")
    keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
    # print(keypoints)

    return keypoints

def get_harris_keypoints(img, k=0.04, threshold_ratio=0.01, border=10):
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = img

    # Compute image gradients
    Iy, Ix = np.gradient(cv2.GaussianBlur(img_gray, (5,5), 0))

    # Compute products of gradients at each pixel
    Sx2 = cv2.GaussianBlur(Ix**2, (5,5), 0)
    Sy2 = cv2.GaussianBlur(Iy**2, (5,5), 0)
    Sxy = cv2.GaussianBlur(Ix*Iy, (5,5), 0)

    # Compute the Harris corner response function
    det_M = Sx2*Sy2 - Sxy*Sxy
    trace_M = Sx2 + Sy2
    R = det_M - k*trace_M**2

    # Threshold the corner response
    threshold = threshold_ratio*np.max(R)
    R[R < threshold] = 0

    # Find local maxima of the corner response
    local_max_R = maximum_filter(R, size=3, mode='constant')
    R[R < local_max_R] = 0

    # Get coordinates of remaining corner points
    keypoints = np.column_stack(np.where(R > 0))

    img_shape = img.shape[:2] # (512, 384)
    new_keypoints = [kp for kp in keypoints if border < kp[0] <
                     img_shape[0]-border and border < kp[1] < img_shape[1]-border]
    # print(new_keypoints)
    return new_keypoints

def rotate_img(img, theta, center=None):
    if center is None:
        center = tuple(np.array(img.shape[1::-1]) / 2)
    # print(f"center: {center}")
    rot_mat = cv2.getRotationMatrix2D(center, theta, 1.0)
    rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return rotated_img

def get_histogram(img, bins=36, sigma=0):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute image gradients
    Iy, Ix = np.gradient(cv2.GaussianBlur(img_gray, (5,5), 0))

    # magnitude and orientation
    mag, theta = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)
    theta = (theta + 360) % 360

    bin_size = 360 // bins
    bucket = np.round(theta / bin_size).astype(int)
    histogram = np.zeros((bins,) + mag.shape)
    for b in range(bins):
        histogram[b] = (bucket == b) * mag
        histogram[b] = cv2.GaussianBlur(histogram[b], (5, 5), sigmaX=sigma, sigmaY=sigma)
    
    return histogram

def get_descriptors(img, keypoints, patch_size=16):
    histogram = get_histogram(img, bins=36)
    orientations = np.argmax(histogram, axis=0) * 10 + 5
    descriptors = []
    h, w = img.shape[:2]

    for y, x in keypoints:
        rotated_img = rotate_img(img, theta=orientations[y, x], center=(int(x), int(y)))
        histogram = get_histogram(rotated_img, bins=8, sigma=8)

        descriptor = []
        for y_ in range(y-patch_size//2, y+patch_size//2, patch_size//2):
            for x_ in range(x-patch_size//2, x+patch_size//2, patch_size//2):
                histogram_ = [np.sum(histogram[bin][y_:y_+patch_size//2, x_:x_+patch_size//2]) for bin in range(8)]
                descriptor += histogram_

        descriptor = np.array(descriptor)
        descriptor_norm = np.linalg.norm(descriptor)
        if descriptor_norm != 0:
            descriptor /= descriptor_norm

        descriptor[descriptor > 0.2] = 0.2
        descriptor_norm = np.linalg.norm(descriptor)
        if descriptor_norm != 0:
            descriptor /= descriptor_norm

        descriptors.append(descriptor)

    return descriptors

def image_match_with_ransac(img1, img2, threshold=0.7, Type='harris'):
    ## Compute Harris keypoints and descriptors for both images
    if Type == 'sift':
        # keypoints1 = get_sift_keypoints(img1)
        # keypoints2 = get_sift_keypoints(img2)
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()

    elif Type == 'orb':
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_ = bf.knnMatch(descriptors1, descriptors2, k=2)

    matches=[]
    for m, n in matches_:
        if m.distance < 0.75 * n.distance:
            kpt1=[int(keypoints1[m.queryIdx].pt[1]), int(keypoints1[m.queryIdx].pt[0])]
            kpt2=[int(keypoints2[m.trainIdx].pt[1]), int(keypoints2[m.trainIdx].pt[0])]
            matches.append([kpt1, kpt2])

    # print(matches)
    ransac_matches = ransac(matches, iterCount=10000, threshold=2)
    
    return ransac_matches

