import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import librosa


@jit(nopython=True)
def f_pitch(p, pitch_ref=69, freq_ref=440.0):
    """Computes the center frequency/ies of a MIDI pitch

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        p (float): MIDI pitch value(s)
        pitch_ref (float): Reference pitch (default: 69)
        freq_ref (float): Frequency of reference pitch (default: 440.0)

    Returns:
        freqs (float): Frequency value(s)
    """
    return 2 ** ((p - pitch_ref) / 12) * freq_ref

@jit(nopython=True)
def pool_pitch(p, Fs, N, pitch_ref=69, freq_ref=440.0):
    """Computes the set of frequency indices that are assigned to a given pitch

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        p (float): MIDI pitch value
        Fs (scalar): Sampling rate
        N (int): Window size of Fourier fransform
        pitch_ref (float): Reference pitch (default: 69)
        freq_ref (float): Frequency of reference pitch (default: 440.0)

    Returns:
        k (np.ndarray): Set of frequency indices
    """
    lower = f_pitch(p - 0.5, pitch_ref, freq_ref)
    upper = f_pitch(p + 0.5, pitch_ref, freq_ref)
    k = np.arange(N // 2 + 1)
    k_freq = k * Fs / N  # F_coef(k, Fs, N)
    mask = np.logical_and(lower <= k_freq, k_freq < upper)
    return k[mask]

def compute_spectrogram(x, Fs, N, H, w='hann', pad_mode='constant', center=True, mag=False ):
    """Computes a magnitude or power spectrogram

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate
        N (int): Window size of Fourier fransform
        H (int): Hopsize
        window (str): Type of window function
        mag (bool): True for magnitude spectrogram, False for power spectrogram

    Returns:
        Y (np.ndarray): Magnitude or power spectrogram
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N,
                     window=w, pad_mode=pad_mode, center=center)
    if mag:
        X = np.abs(X)**2
    F_coef = librosa.fft_frequencies(sr=Fs, n_fft=N)
    T_coef = librosa.frames_to_time(np.arange(X.shape[1]), sr=Fs, hop_length=H)
    return X, T_coef, F_coef

@jit(nopython=True)
def compute_spec_log_freq(Y, Fs, N):
    """Computes a log-frequency spectrogram

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        Y (np.ndarray): Magnitude or power spectrogram
        Fs (scalar): Sampling rate
        N (int): Window size of Fourier fransform

    Returns:
        Y_LF (np.ndarray): Log-frequency spectrogram
        F_coef_pitch (np.ndarray): Pitch values
    """
    Y_LF = np.zeros((128, Y.shape[1]))
    for p in range(128):
        k = pool_pitch(p, Fs, N)
        Y_LF[p, :] = Y[k, :].sum(axis=0)
    F_coef_pitch = np.arange(128)
    return Y_LF, F_coef_pitch


@jit(nopython=True)
def compute_chromagram(Y_LF):
    """Computes a chromagram

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        Y_LF (np.ndarray): Log-frequency spectrogram

    Returns:
        C (np.ndarray): Chromagram
    """
    C = np.zeros((12, Y_LF.shape[1]))
    p = np.arange(128)
    for c in range(12):
        mask = (p % 12) == c
        C[c, :] = Y_LF[mask, :].sum(axis=0)
    return C

def chromagram(wavfile, fs, N, H):
    x, _ = librosa.load(wavfile, sr=fs)
    Y ,_,_= compute_spectrogram(x, fs, N, H,mag=True)
    Y_LF, _ = compute_spec_log_freq(Y, fs, N)  
    chroma = compute_chromagram(Y_LF)
    return chroma

if __name__ == '__main__':
    wavfile = "/home/usuari/Desktop/SMC-Master/MIR/FINALPROJECT/MajorChords-Flute.wav"
    fs = 22050
    x, _ = librosa.load(wavfile, sr=fs)

    N = 4096
    H = 1024
    chroma = chromagram(wavfile, fs, N, H)

    eps = np.finfo(float).eps

    fig = plt.figure(figsize=(10, 3))
    chroma_label = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    plt.imshow(10 * np.log10(eps + chroma), origin='lower', aspect='auto', cmap='gray_r',
            )
    plt.clim([0, 60])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Chroma')
    cbar = plt.colorbar()
    cbar.set_label('Magnitude (dB)')
    plt.yticks(np.arange(12) + 0.5, chroma_label)
    plt.tight_layout()
    plt.show()