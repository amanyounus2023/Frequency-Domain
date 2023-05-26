import numpy as np

# Generate a random signal
signal = np.random.random(1024)

# Compute FFT
fft_result = np.fft.fft(signal)
print(fft_result)