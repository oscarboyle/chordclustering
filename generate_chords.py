import os
from music21 import stream, chord, instrument, pitch
from midi2audio import FluidSynth

# Define all root notes
root_notes = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Define chord structures (Major and Minor inversions)
chord_patterns = {
    "Major": [[0, 4, 7], [4, 7, 12], [7, 12, 16]],
    "Minor": [[0, 3, 7], [3, 7, 12], [7, 12, 15]],
}

# Define instruments
instruments = [
    instrument.Piano(),
    instrument.Violin(),
    instrument.Viola(),
    instrument.Violoncello(),
    instrument.Contrabass(),
    instrument.Harp(),
    instrument.Guitar(),
    instrument.Flute(),
    instrument.Piccolo(),
    instrument.Clarinet(),
    instrument.Oboe(),
    instrument.Bassoon(),
    instrument.Saxophone(),
    instrument.Trumpet(),
    instrument.Trombone(),
]

# Define specific octave ranges for instruments
instrument_octaves = {
    "Piano": [3, 4, 5, 6],
    "Violin": [4, 5],
    "Viola": [3, 4, 5],
    "Violoncello": [2, 3, 4],
    "Contrabass": [1, 2],
    "Harp": [3, 4, 5],
    "Guitar": [3, 4, 5],
    "Flute": [5, 6],
    "Piccolo": [6],
    "Clarinet": [4, 5],
    "Oboe": [4, 5],
    "Bassoon": [3, 4],
    "Saxophone": [4, 5],
    "Trumpet": [4, 5],
    "Trombone": [3, 4],
}

# Directories to store outputs
midi_output_dir = "generated_chords/midi"
wav_output_dir = "generated_chords/wav"
os.makedirs(midi_output_dir, exist_ok=True)
os.makedirs(wav_output_dir, exist_ok=True)

# Initialize FluidSynth once outside the loop
fs = FluidSynth(sound_font="MuseScore_General.sf2")  # wget ftp://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2

# Function to generate MIDI file for a chord with specific octaves for instruments
def create_midi(root, intervals, instrument_obj, filename, octave):
    s = stream.Stream()
    s.append(instrument_obj)  # Set instrument
    
    # Compute pitch names with selected octaves
    root_pitch = pitch.Pitch(root + str(octave))  # Use selected octave
    chord_notes = [root_pitch.transpose(interval).nameWithOctave for interval in intervals]

    c = chord.Chord(chord_notes)
    c.quarterLength = 2  # Set duration
    s.append(c)

    midi_fp = os.path.join(midi_output_dir, filename)
    s.write("midi", fp=midi_fp)
    return midi_fp

# Function to convert MIDI file to WAV
def convert_mid_to_wav(mid_file, wav_file):
    fs.midi_to_audio(mid_file, wav_file)

# Generate all MIDI files and store their paths
midi_files = []
for instrument_obj in instruments:
    instrument_name = instrument_obj.instrumentName
    print(f"Generating MIDI for: {instrument_name}")
    available_octaves = instrument_octaves.get(instrument_name, [])
    for root in root_notes:
        for octave in available_octaves:
            for i, (chord_type, variations) in enumerate(chord_patterns.items()):
                for j, intervals in enumerate(variations):
                    midi_filename = f"{instrument_name}_{root}_{chord_type}_{octave}_{j}.mid"
                    midi_path = create_midi(root, intervals, instrument_obj, midi_filename, octave)
                    midi_files.append(midi_path)

print(f"Generated {len(midi_files)} MIDI files.")

# Convert all MIDI files to WAV
print("Starting MIDI to WAV conversion...")
for midi_path in midi_files:
    wav_filename = os.path.basename(midi_path).replace(".mid", ".wav")
    wav_path = os.path.join(wav_output_dir, wav_filename)
    convert_mid_to_wav(midi_path, wav_path)

print("All chords have been generated and converted to WAV.")
