import cv2
import numpy as np
def pixel_val(pix, coords):
    for i in range(0, len(coords), 2):
        xCurr, yCurr = coords[i], coords[i + 1]
        xNext, yNext = coords[i + 2:i + 4] if i + 2 < len(coords) - 1 else (255, 255)

        if i == 0:
            if 0 <= pix and pix <= xCurr:
                return (yCurr / xCurr) * pix

        if xCurr <= pix and pix <= xNext:
            return ((yNext - yCurr) / (xNext - xCurr)) * (pix - xCurr) + yCurr


img = cv2.imread('img_small.png')

b, g, r = cv2.split(img)

# vrednostite se soodvetno x1, y1, x2, y2, x3, y3...
coords = [50, 0, 100, 50, 150, 200]
#coords = [50, 30, 150, 200, 255, 255]


b_contrast_stretched = np.zeros_like(b)
g_contrast_stretched = np.zeros_like(g)
r_contrast_stretched = np.zeros_like(r)

# probav i so vectorize da napravam kade go isprakjav sekoj kanal posebno ama davase error pri povik na len(coords)
for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        b_contrast_stretched[i, j] = pixel_val(b[i, j], coords)
        g_contrast_stretched[i, j] = pixel_val(g[i, j], coords)
        r_contrast_stretched[i, j] = pixel_val(r[i, j], coords)


contrast_stretched = cv2.merge((b_contrast_stretched, g_contrast_stretched, r_contrast_stretched))
contrast_stretched = np.array(contrast_stretched, dtype=np.uint8)

cv2.imshow('normal', img)
cv2.imshow('contrast_stretched', contrast_stretched)
cv2.waitKey(0)
cv2.destroyAllWindows()


