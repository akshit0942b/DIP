import os
import math
import cv2 
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt

img = cv2.imread("dog.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, c = img.shape

# IMPROVED: Vectorized RGB to YCbCr conversion (much faster)
R = img[:, :, 0].astype(np.float32)
G = img[:, :, 1].astype(np.float32)
B = img[:, :, 2].astype(np.float32)

# Playing with Intensity
Y  = np.clip(0.299*R + 0.587*G + 0.114*B, 0, 255).astype(np.uint8)

log = 1 * np.log(Y + 1)
negative = 255 - Y
gamma = 1 * (Y**0.0001)

# plt.subplot(1, 2, 1)
# plt.imshow(Y, cmap='gray')
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(gamma, cmap='gray')
# plt.axis("off")

 
# hist = cv2.calcHist([Y], [0], None, [256], [0, 256])
# cdf = hist.cumsum()
# cdfNorm = cdf * float(hist.max()) / cdf.max()


# equImg = cv2.equalizeHist(Y)
# equHist = cv2.calcHist([equImg], [0], None, [256], [0, 256])
# equCdf = equHist.cumsum()
# equCdfNorm = equCdf * float(equHist.max()) / equCdf.max()


# claheObj = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
# claheImg = claheObj.apply(Y)
# claheHist = cv2.calcHist([claheImg], [0], None, [256], [0, 256])
# claheCdf = claheHist.cumsum()
# claheCdfNorm = claheCdf * float(claheHist.max()) / claheCdf.max()

  
# plt.subplot(3, 2, 1)
# plt.imshow(Y, cmap='gray')
# plt.ylabel("Original")

# plt.subplot(3, 2, 2)
# plt.plot(hist)
# plt.plot(cdfNorm, color='y')
# plt.ylabel('# of pixels')

# plt.subplot(3, 2, 3)
# plt.imshow(equImg,cmap='gray')
# plt.ylabel("equ")

# plt.subplot(3, 2, 4)
# plt.plot(equHist)
# plt.plot(equCdfNorm, color='y')
# plt.ylabel('# of pixels(equ)')

# plt.subplot(3, 2, 5)
# plt.imshow(claheImg, cmap='gray')
# plt.ylabel("Clahe")

# plt.subplot(3, 2, 6)
# plt.plot(claheHist)
# plt.plot(claheCdfNorm, color='y')
# plt.xlabel('pixel intensity(clahe)')
# plt.ylabel('# of pixels(clahe)')


# plt.show()



diff = (Y - gamma + 255) / 2 

plt.subplot(2, 2, 1)
plt.imshow(Y, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gamma, cmap='gray')
plt.title("Gamma")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(diff, cmap='gray')
plt.title("Difference")
plt.axis("off")

plt.show()

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 