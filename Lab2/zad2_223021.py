import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use as mpl_use
mpl_use('TkAgg')

img = cv2.imread('messi.jpg', 0)
#img = cv2.imread('lena.png', 0)

compassOperatorKernels = [np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]]),
                          np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
                          np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]),
                          np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
                          np.array([]),
                          np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                          np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]),
                          np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
                          np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
                          ]
filteredImages = []


plt.figure(1)

for i, kernel in enumerate(compassOperatorKernels):
    if i == 4:
        continue
    filteredImg = cv2.filter2D(img, -1, kernel)
    filteredImages.append(filteredImg)
    plt.subplot(3, 3, i+1)
    plt.imshow(filteredImg, cmap='gray')


combinedFiltersImg = np.zeros_like(filteredImages[0])
for filteredImg in filteredImages:
    combinedFiltersImg = np.maximum(combinedFiltersImg, filteredImg)
plt.subplot(3, 3, 5)
plt.title("Combined Filters")
plt.imshow(combinedFiltersImg, cmap='gray')

plt.figure(2)

ret15, thresholdCombined15 = cv2.threshold(combinedFiltersImg, 15, 255, cv2.THRESH_BINARY)
ret30, thresholdCombined30 = cv2.threshold(combinedFiltersImg, 30, 255, cv2.THRESH_BINARY)
ret45, thresholdCombined45 = cv2.threshold(combinedFiltersImg, 45, 255, cv2.THRESH_BINARY)
ret60, thresholdCombined60 = cv2.threshold(combinedFiltersImg, 60, 255, cv2.THRESH_BINARY)

plt.subplot(2,2,1)
plt.title("Threshold 15")
plt.imshow(thresholdCombined15, cmap='gray')

plt.subplot(2,2,2)
plt.title("Threshold 30")
plt.imshow(thresholdCombined30, cmap='gray')

plt.subplot(2,2,3)
plt.title("Threshold 45")
plt.imshow(thresholdCombined45, cmap='gray')

plt.subplot(2,2,4)
plt.title("Threshold 60")
plt.imshow(thresholdCombined60, cmap='gray')

plt.show()