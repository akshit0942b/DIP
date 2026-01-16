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

# Luminance
Y  = np.clip(0.299*R + 0.587*G + 0.114*B, 0, 255).astype(np.uint8)

# Multiple noise addition 
noise = np.zeros(Y.shape, dtype = np.float32)
N = 5
s = 1

noisy = Y.astype(np.float32) + noise

for i in range (4):
    l = N
    
    # plt.subplot(3, 2, s)
    # plt.imshow(noisy, cmap='gray')
    # plt.axis("off")  
    
    while(l > 0):
        cv2.randn(noise, mean=0, stddev=25)
        noisy = noisy + noise
        l = l - 1
        
    s = s + 1
    N = N + 5           


# Color edge detection

color_img = cv2.imread("prasidh.png")
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)


Rc = color_img[:, :, 0].astype(np.float32)
Gc = color_img[:, :, 1].astype(np.float32)
Bc = color_img[:, :, 2].astype(np.float32)

sobelx_R = cv2.Sobel(Rc, cv2.CV_32F, 1, 0, ksize=3)
sobelx_G = cv2.Sobel(Gc, cv2.CV_32F, 1, 0, ksize=3)
sobelx_B = cv2.Sobel(Bc, cv2.CV_32F, 1, 0, ksize=3)

sobely_R = cv2.Sobel(Rc, cv2.CV_32F, 0, 1, ksize=3)
sobely_G = cv2.Sobel(Gc, cv2.CV_32F, 0, 1, ksize=3)
sobely_B = cv2.Sobel(Bc, cv2.CV_32F, 0, 1, ksize=3)

Gx = np.sqrt(sobelx_R**2 + sobelx_G**2 + sobelx_B**2)
Gy = np.sqrt(sobely_R**2 + sobely_G**2 + sobely_B**2)

edge_mag = np.sqrt(Gx**2 + Gy**2)

edge_mag = cv2.normalize(edge_mag, None, 0, 255, cv2.NORM_MINMAX)
edge_mag = edge_mag.astype(np.uint8)

plt.subplot(2, 1, 1)
plt.imshow(color_img)
plt.axis("off")

plt.subplot(2, 1, 2)
plt.imshow(edge_mag)
plt.axis("off")

plt.show()









