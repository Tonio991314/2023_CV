import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        
        ### TODO ###
        ## Look up table for range kernel
        LUT_range_kernel = np.exp(-(np.arange(256)/255) * (np.arange(256)/255) / (2*self.sigma_r**2)) ## (256, 256)

        ## Look up table for spectial kernel
        x, y = np.meshgrid(np.arange(2 * self.pad_w + 1) - self.pad_w, np.arange(2 * self.pad_w + 1) - self.pad_w)
        spetial_kernel =  np.exp(-(x**2 + y**2) / (2*self.sigma_s**2)) ## (19, 19)

        output = np.zeros_like(img)
        for x in range(self.pad_w, img.shape[0] + self.pad_w):
            for y in range(self.pad_w, img.shape[1] + self.pad_w):
                if len(guidance.shape) == 3: ## RGB
                    numerator = LUT_range_kernel[abs(padded_guidance[x - self.pad_w : x + self.pad_w + 1, y - self.pad_w : y + self.pad_w + 1, 0] - padded_guidance[x, y, 0])] * \
                                LUT_range_kernel[abs(padded_guidance[x - self.pad_w : x + self.pad_w + 1, y - self.pad_w : y + self.pad_w + 1, 1] - padded_guidance[x, y, 1])] * \
                                LUT_range_kernel[abs(padded_guidance[x - self.pad_w : x + self.pad_w + 1, y - self.pad_w : y + self.pad_w + 1, 2] - padded_guidance[x, y, 2])] * \
                                spetial_kernel
                elif  len(guidance.shape) == 2: ## Gray
                    numerator = LUT_range_kernel[abs(padded_guidance[x - self.pad_w : x + self.pad_w + 1, y - self.pad_w : y + self.pad_w + 1] - padded_guidance[x, y])] * spetial_kernel
                
                Denominator = np.sum(numerator)
                output[x - self.pad_w, y - self.pad_w, 0] = np.sum(numerator * padded_img[x - self.pad_w : x + self.pad_w + 1, y - self.pad_w : y + self.pad_w + 1, 0]) / Denominator
                output[x - self.pad_w, y - self.pad_w, 1] = np.sum(numerator * padded_img[x - self.pad_w : x + self.pad_w + 1, y - self.pad_w : y + self.pad_w + 1, 1]) / Denominator
                output[x - self.pad_w, y - self.pad_w, 2] = np.sum(numerator * padded_img[x - self.pad_w : x + self.pad_w + 1, y - self.pad_w : y + self.pad_w + 1, 2]) / Denominator

        return np.clip(output, 0, 255).astype(np.uint8)
    
