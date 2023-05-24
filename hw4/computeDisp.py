import numpy as np
import cv2.ximgproc as xip

def computeCensus(img):
    h, w, ch = img.shape
    census = np.zeros((h, w, ch), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            for c in range(ch):
                center_val = img[y, x, c]
                census_code = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if x + dx < 0 or x + dx >= w or y + dy < 0 or y + dy >= h:
                            continue
                        if img[y + dy, x + dx, c] < center_val:
                            census_code |= 1
                        census_code <<= 1
                census[y, x, c] = census_code >> 1

    return census

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    Il_census = computeCensus(Il)
    Ir_census = computeCensus(Ir)

    cost_left = np.zeros((max_disp+1, h, w), dtype=np.float32)
    cost_right = np.zeros((max_disp+1, h, w), dtype=np.float32)

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)

    for d in range(max_disp+1):
        for x in range(w):
            x_left = max(x-d, 0)
            x_right = min(x+d, w-1)
            for y in range(h):
                ## Census cost (Hamming distance)
                cost_left[d, y, x] = np.sum(Il_census[y, x, :] != Ir_census[y, x_left, :]) 
                cost_right[d, y, x] = np.sum(Ir_census[y, x, :] != Il_census[y, x_right, :])
                
        ## Joint bilateral filter
        sigmaColor = 48
        sigmaSpace = 10
        cost_left[d, :, :] = xip.jointBilateralFilter(Il, cost_left[d, :, :], sigmaColor, sigmaSpace, sigmaSpace)
        cost_right[d, :, :] = xip.jointBilateralFilter(Ir, cost_right[d, :, :], sigmaColor, sigmaSpace, sigmaSpace)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    winner_left = np.argmin(cost_left, axis=0)
    winner_right = np.argmin(cost_right, axis=0)
    
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    
    ## Left-right consistency check
    for y in range(h):
        for x in range(w):
            if x-winner_left[y, x] < 0 or winner_right[y, x-winner_left[y, x]] != winner_left[y, x]:
                winner_left[y, x] = -1
                             
    ## Hole filling
    for y in range(h):
        for x in range(w):
            if winner_left[y, x] == -1:
                l, r = 0, 0
                while x - l >= 0 and winner_left[y, x - l] == -1:
                    l += 1
                Fl = max_disp if x - l < 0 else winner_left[y, x - l]

                while x + r <= w - 1 and winner_left[y, x + r] == -1:
                    r += 1
                Fr = max_disp if x + r > w - 1 else winner_left[y, x + r]

                winner_left[y, x] = min(Fl, Fr)
                    
    ## Weighted median filtering
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), winner_left.astype(np.uint8), 18, 0.5)
    
    return labels.astype(np.uint8)


