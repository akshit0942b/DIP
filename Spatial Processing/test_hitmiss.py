import cv2
import numpy as np

img = cv2.imread("images/lines.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
binary_inv = cv2.bitwise_not(binary)

print("binary unique values:", np.unique(binary))
print("binary_inv unique values:", np.unique(binary_inv))

B1 = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0]
], dtype=np.uint8)

B2 = np.array([
    [1,0,1],
    [0,0,0],
    [1,0,1]
], dtype=np.uint8)

# Try hitmiss on binary
e1 = cv2.erode(binary, B1)
e2 = cv2.erode(255 - binary, B2)
hm = cv2.bitwise_and(e1, e2)
print("hm on binary max:", np.max(hm))

# Try hitmiss on binary_inv
e1 = cv2.erode(binary_inv, B1)
e2 = cv2.erode(255 - binary_inv, B2)
hm_inv = cv2.bitwise_and(e1, e2)
print("hm on binary_inv max:", np.max(hm_inv))

