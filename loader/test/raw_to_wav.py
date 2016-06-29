import numpy as np
from scipy.io.wavfile import write

a = np.fromfile('/tmp/file.raw', dtype='int16')
write('/tmp/file.wav', 16000, a)
