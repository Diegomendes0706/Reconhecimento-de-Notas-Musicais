from os import listdir
import os
import scipy.io.wavfile as wavfile
import numpy as np
from numpy.fft import fft

if __name__ == '__main__':
    # pegar arquivos de som
    path_sounds = r'.\Violão'
    sound_files = listdir(path_sounds)

    n = 65536  # 16 bits

    frequency = 16_000
    time_interval = np.linspace(0, 10, n)

    sound_data = [
        wavfile.read(os.path.join(path_sounds, s))[1]
        for s in sound_files
    ]

    f_hats = [abs(fft(d)) for d in sound_data]
