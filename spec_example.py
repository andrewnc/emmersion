import matplotlib.pyplot as plt
import numpy as np


from DataLoader import DataLoader
# Fixing random state for reproducibility

data_loader = DataLoader(truncate_secs=10, dtype=np.float32)

x = data_loader._data[0]

dt = 0.0005

NFFT = 1024       # the length of the windowing segments
Fs = int(1.0 / dt)  # the sampling frequency

# Pxx is the segments x freqs array of instantaneous power, freqs is
# the frequency vector, bins are the centers of the time bins in which
# the power is computed, and im is the matplotlib.image.AxesImage
# instance

ax1 = plt.subplot(211)
plt.plot(x)
plt.subplot(212, sharex=ax1)
Pxx, freqs, bins, im = plt.specgram(x, NFFT=NFFT, Fs=2, noverlap=900)
plt.show()