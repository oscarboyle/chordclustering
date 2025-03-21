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


def build_chord_templates():
    """
    Build a dictionary of chord templates for major and minor triads
    across all roots (12 semitones).
    
    Returns:
        chord_templates (dict): keys = chord name (str),
                                values = 12D numpy array
    """
    # For convenience, we index pitch classes as:
    # C=0, C#=1, D=2, D#=3, E=4, F=5,
    # F#=6, G=7, G#=8, A=9, A#=10, B=11
    # major triad = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] (root, major 3rd, perfect 5th)
    # minor triad = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] but with a flat 3rd

    # base major triad starting at C
    major_base = np.zeros(12)
    major_base[0] = 1   # root
    major_base[4] = 1   # major 3rd
    major_base[7] = 1   # perfect 5th
    
    # base minor triad starting at C
    minor_base = np.zeros(12)
    minor_base[0] = 1   # root
    minor_base[3] = 1   # minor 3rd
    minor_base[7] = 1   # perfect 5th

    # base diminished triad starting at C
    diminished_base = np.zeros(12)
    diminished_base[0] = 1   # root
    diminished_base[3] = 1   # minor 3rd
    diminished_base[6] = 1   # diminished 5th

    # base augmented triad starting at C
    augmented_base = np.zeros(12)
    augmented_base[0] = 1   # root
    augmented_base[4] = 1   # major 3rd
    augmented_base[8] = 1   # augmented 5th

    chord_templates = {}
    note_names = ['C', 'C#', 'D', 'Eb', 'E', 
                  'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    
    # Construct major/minor templates for all 12 possible roots
    for root in range(12):
        # Rotate the base pattern by 'root' semitones
        major_pattern = np.roll(major_base, root)
        minor_pattern = np.roll(minor_base, root)
        diminished_pattern = np.roll(diminished_base, root)
        augmented_pattern = np.roll(augmented_base, root)

        # Name the chord e.g. "C Maj", "C# Maj", ... "C Min", ...
        chord_templates[f"{note_names[root]}_Major"] = major_pattern
        chord_templates[f"{note_names[root]}_Minor"] = minor_pattern
        chord_templates[f"{note_names[root]}_Diminished"] = diminished_pattern
        chord_templates[f"{note_names[root]}_Augmented"] = augmented_pattern
    
    return chord_templates


