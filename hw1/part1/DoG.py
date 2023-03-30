import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
       	
        gaussian_images = []
        for i in range(self.num_octaves): #(0, 1)
            octave=[]
            resize = True
            if i == 1 and resize == True: ### resize
                image = cv2.resize(gaussian_images[0][self.num_DoG_images_per_octave], (image.shape[1]//2, image.shape[0]//2),  interpolation=cv2.INTER_NEAREST)
                octave.append(image)
                resize = False
            elif i ==0 :
                octave.append(image)

            for j in range(self.num_guassian_images_per_octave-1): ###(0,1,2,3)
                img = cv2.GaussianBlur(image, (0,0), sigmaX=self.sigma**(j+1), sigmaY=self.sigma**(j+1))
                octave.append(img)
            
            gaussian_images.append(octave)
	
 
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(self.num_octaves): ### (0,1)
            octave=[]
            for j in range(self.num_DoG_images_per_octave): ### (0,1,2,3)
                dog = cv2.subtract(gaussian_images[i][j+1], gaussian_images[i][j])
                
                ## plot DoG image
                # cv2.imwrite(f"image/DoG{i+1}-{j+1}.png", dog)

                octave.append(dog)
                
            dog_images.append(octave)
        
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for i in range(self.num_octaves):  # (0,1)
            for j in range(1, self.num_DoG_images_per_octave-1): ## (1,2)
                prev_img, curr_img, next_img = dog_images[i][j-1], dog_images[i][j], dog_images[i][j+1]
                h, w = curr_img.shape[:2]
                for x in range(1, h-1):
                    for y in range(1, w-1):
                        patch = np.stack([
                            prev_img[x-1:x+2, y-1:y+2],
                            curr_img[x-1:x+2, y-1:y+2],
                            next_img[x-1:x+2, y-1:y+2]
                        ])
                        # print("center: " + str(patch[1, 1, 1]))
                        # print("count: " + str(np.count_nonzero(patch == patch[1, 1, 1])))
                        # print("================================")
                        
                        if np.isnan(patch[1, 1, 1])==False and np.count_nonzero(patch == patch[1, 1, 1])==1:
                            center_value = patch[1, 1, 1]
                            if center_value == np.nanmax(patch) or center_value == np.nanmin(patch):
                                if abs(center_value) >= self.threshold:
                                    if i == 0:
                                        keypoints.append((x, y))
                                    else:
                                        keypoints.append((2*x, 2*y))
        
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
