import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread(r'C:\Users\SAHYADRI\Desktop\valo.jfif', 0)

# Compute FFT and shift zero frequency to the center
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Create a Gaussian low-pass filter
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
d0 = 30  # cutoff frequency
gaussian_filter = np.exp(-((np.square(cols - ccol) + np.square(rows - crow)) / (2 * d0 ** 2)))

# Apply the filter and perform inverse FFT
fshift_filtered = fshift * gaussian_filter
f_ishift = np.fft.ifftshift(fshift_filtered)
image_filtered = np.abs(np.fft.ifft2(f_ishift))

# Display the original and filtered images
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(image_filtered, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
