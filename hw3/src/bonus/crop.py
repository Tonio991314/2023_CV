import cv2
from utils import *

type="harris"
result = cv2.imread(f"data/blend_{type}.png")
result = crop_image(result)
cv2.imwrite(f'data/result_{type}.png', result)
print("crop done")