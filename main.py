from glob import glob
import scipy.io.wavfile as wavfile
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.fftpack import fft
import pandas as pd

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    # pegar arquivos
    notes = pd.read_csv('notas.csv')
    sound_files = glob(r'./Violão/*.wav')
    frequency = 16_000
    period = 1 / frequency

    sound_data = [
        wavfile.read(s)[1]
        for s in sound_files
    ]
    audio_lengths = [
        len(sound_data[d]) / frequency
        for d in range(len(sound_data))
    ]
    time_intervals = [
        np.arange(0, audio_lengths[a], period)
        for a in range(len(audio_lengths))
    ]
    frequencies = [
        (np.fft.fftfreq(len(sound_data[d]), period))
        for d in range(len(sound_data))
    ]
    f_hats = [fft(d) for d in sound_data]
    spectral_densities = [abs(fh) for fh in f_hats]

    plt.figure()
    plt.plot(frequencies[0], spectral_densities[0])
    plt.show()
