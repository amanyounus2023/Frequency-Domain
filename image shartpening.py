import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread(r'C:\Users\SAHYADRI\Desktop\valo.jfif', 0)

# Compute FFT and shift zero frequency to the center
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Define a sharpening kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Convolve the image with the sharpening kernel in the frequency domain
fshift_sharpened = fshift * np.fft.fft2(kernel, image.shape)
f_ishift = np.fft.ifftshift(fshift_sharpened)
image_sharpened = np.abs(np.fft.ifft2(f_ishift))

# Display the original and sharpened images
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(image_sharpened, cmap='gray')
plt.title('Sharpened Image'), plt.xticks([]), plt.yticks([])
plt.show()
