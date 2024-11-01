import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use as mpl_use
mpl_use('TkAgg')

# Izbraniot algoritam prilicno dobro gi segmentira slikite so mnogu minimalni greski

# Namestete ja patekata kaj vas za da vi raboti, jas staviv vo zipot od nekolku sliki rezultati
img = cv2.imread('../database/10309.jpg', 0)
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret, thresholdedImg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresholdedInvertedImg = 255 - thresholdedImg

# Dilatacija pa erozija - opening
opened = cv2.morphologyEx(thresholdedImg, cv2.MORPH_OPEN, kernel=np.ones((5, 5)))
contoursOpened, hierarchy1 = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoursOpenedImage = cv2.drawContours(np.zeros_like(img), contoursOpened, -1, (255, 255, 255), 1)

# Vtor nacin - cisto da probam :)
# Erozija pa dilatacija - closing
closed = cv2.morphologyEx(thresholdedInvertedImg, cv2.MORPH_CLOSE, kernel=np.ones((5, 5)))
contoursClosed, hierarchy2 = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoursClosedImage = cv2.drawContours(np.zeros_like(img), contoursClosed, -1, (255, 255, 255), 1)


plt.figure(1)
plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1,4,2)
plt.title("Thresholded")
plt.imshow(thresholdedImg, cmap='gray')

plt.subplot(1,4,3)
plt.title("Opening - Erosion -> Dilation")
plt.imshow(opened, cmap='gray')

plt.subplot(1,4,4)
plt.title("Contours")
plt.imshow(255-contoursOpenedImage, cmap='gray')

plt.show()





