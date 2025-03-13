from featureExtraction import chromagram, chromaOnsets
import numpy as np
import glob
import os
import json
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
DATA_FOLDER = "/home/usuari/Desktop/SMC-Master/MIR/FINALPROJECT" # Change this to data folder
SEPARATE_CHORDS = False
FS = 22050
N = 4096
H = 1024

def get_wav_files(directory):
    """
    Recursively finds all .wav files in the given directory and subdirectories.
    
    Parameters:
    directory (str): The root directory to search.
    
    Returns:
    List of strings: A list of full file paths to .wav files.
    """
    return glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True)



if __name__ == '__main__':
    # Load audio files
    wav_files = get_wav_files(DATA_FOLDER)
    chromagrams = {}

    # Compute chromagrams
    for wavfile in tqdm(wav_files, desc='Computing chromagrams'):
        chroma = chromagram(wavfile, fs=FS, N=N, H=H)
        chroma_onsets = chromaOnsets(chroma)
        if SEPARATE_CHORDS:
            # Separate into different chords
            for i, onset in enumerate(chroma_onsets):
                if i == 0:
                    start = 0
                else:
                    start = chroma_onsets[i - 1]
                end = onset
                chord = chroma[:, start:end]
                chromagrams[wavfile + f"_{i}"] = (chord.tolist())
        else:
            chromagrams[wavfile] = chroma.tolist()


    print(len(chromagrams))
    # Save chromagrams into json file
    with open("chromagrams.json", "w") as f:
        json.dump(chromagrams, f, indent=2)

    # Load chromagrams from json file
    with open("chromagrams.json", "r") as f:
        chromagrams = json.load(f)

    # Choose a random chromagram
    random_key = random.choice(list(chromagrams.keys()))
    chroma = np.array(chromagrams[random_key])
    # Plot some chromagrams
    eps = np.finfo(float).eps
    fig = plt.figure(figsize=(10, 3))
    chroma_label = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    plt.imshow(10 * np.log10(eps + chroma), origin='lower', aspect='auto', cmap='gray_r')
    plt.clim([0, 60])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Chroma')
    cbar = plt.colorbar()
    cbar.set_label('Magnitude (dB)')
    plt.yticks(np.arange(12) + 0.5, chroma_label)
    
    plt.tight_layout()
    plt.show()
    print(chroma.shape)



