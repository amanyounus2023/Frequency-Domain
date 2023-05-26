import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread(r'C:\Users\SAHYADRI\Desktop\valo.jfif', 0)

# Add random noise to the image
noisy_image = image + np.random.normal(0, 50, image.shape)

# Compute FFT and shift zero frequency to the center
f = np.fft.fft2(noisy_image)
fshift = np.fft.fftshift(f)

# Apply a low-pass filter for denoising
rows, cols = noisy_image.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols), np.uint8)
r = 50  # radius of the circular mask
mask[crow - r: crow + r, ccol - r: ccol + r] = 0
fshift_filtered = fshift * mask

# Perform inverse FFT
f_ishift = np.fft.ifftshift(fshift_filtered)
denoised_image = np.abs(np.fft.ifft2(f_ishift))

# Display the original and denoised images
plt.subplot(121), plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(denoised_image, cmap='gray')
plt.title('Denoised Image'), plt.xticks([]), plt.yticks([])
plt.show()
