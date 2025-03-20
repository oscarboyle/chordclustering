import logging
import os
import subprocess

from concurrent.futures import ThreadPoolExecutor
from midi2audio import FluidSynth
from music21 import stream, chord, instrument, pitch

# Configure logging for debugging and tracking progress
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define all root notes for chord generation
ROOT_NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Define different chord structures with multiple inversions
CHORD_PATTERNS = {
    "Major": [[0, 4, 7], [4, 7, 12], [7, 12, 16]],
    "Minor": [[0, 3, 7], [3, 7, 12], [7, 12, 15]],
    "Augmented": [[0, 4, 8], [4, 8, 12], [8, 12, 16]],
    "Diminished": [[0, 3, 6], [3, 6, 12], [6, 12, 15]],
}

# Define available instruments and their corresponding octave ranges
INSTRUMENTS = {
    "Piano": instrument.Piano(),
    "Violin": instrument.Violin(),
    "Viola": instrument.Viola(),
    "Violoncello": instrument.Violoncello(),
    "Contrabass": instrument.Contrabass(),
    "Harp": instrument.Harp(),
    "Guitar": instrument.Guitar(),
    "Flute": instrument.Flute(),
    "Piccolo": instrument.Piccolo(),
    "Clarinet": instrument.Clarinet(),
    "Oboe": instrument.Oboe(),
    "Bassoon": instrument.Bassoon(),
    "Saxophone": instrument.Saxophone(),
    "Trumpet": instrument.Trumpet(),
    "Trombone": instrument.Trombone(),
}

# Define the octave ranges for each instrument
INSTRUMENT_OCTAVES = {
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

# Directories for storing MIDI and WAV files
MIDI_OUTPUT_DIR = "generated_chords/midi"
WAV_OUTPUT_DIR = "generated_chords/wav"
os.makedirs(MIDI_OUTPUT_DIR, exist_ok=True)
os.makedirs(WAV_OUTPUT_DIR, exist_ok=True)

# Initialize FluidSynth with the specified soundfont
SOUND_FONT = "MuseScore_General.sf2"  # wget ftp://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2
if not os.path.exists(SOUND_FONT):
    logging.error(f"SoundFont '{SOUND_FONT}' not found! Please provide a valid path.")
    exit(1)
fs = FluidSynth(sound_font=SOUND_FONT)


def create_midi(root, intervals, instrument_name, filename, octave):
    """Generate a MIDI file for a given chord and instrument."""
    try:
        s = stream.Stream()
        s.append(INSTRUMENTS[instrument_name])  # Assign instrument to the stream
        
        # Compute pitch names based on the root note and intervals
        root_pitch = pitch.Pitch(root + str(octave))
        chord_notes = [root_pitch.transpose(interval).nameWithOctave for interval in intervals]
        
        c = chord.Chord(chord_notes)
        c.quarterLength = 2  # Set chord duration
        s.append(c)

        # Save the generated MIDI file
        midi_fp = os.path.join(MIDI_OUTPUT_DIR, filename)
        s.write("midi", fp=midi_fp)
        return midi_fp
    except Exception as e:
        logging.error(f"Failed to create MIDI for {filename}: {e}")
        return None


def convert_mid_to_wav(mid_file):
    """Convert a MIDI file to WAV format while suppressing FluidSynth output."""
    try:
        wav_filename = os.path.basename(mid_file).replace(".mid", ".wav")
        wav_path = os.path.join(WAV_OUTPUT_DIR, wav_filename)
        
        # Run FluidSynth using subprocess while suppressing its output
        with open(os.devnull, 'wb') as devnull:
            subprocess.run(
                ["fluidsynth", "-ni", SOUND_FONT, mid_file, "-F", wav_path, "-r", "44100"],
                stdout=devnull, stderr=devnull, check=True
            )
        
        return wav_path
    except Exception as e:
        logging.error(f"Failed to convert {mid_file} to WAV: {e}")
        return None


def generate_chords():
    """Generate all chord variations for each instrument and store as MIDI files."""
    midi_files = []
    for instrument_name in INSTRUMENTS:
        logging.info(f"Generating MIDI for: {instrument_name}")
        available_octaves = INSTRUMENT_OCTAVES.get(instrument_name, [])
        
        # Iterate through all root notes, octaves, chord types, and variations
        for root in ROOT_NOTES:
            for octave in available_octaves:
                for chord_type, variations in CHORD_PATTERNS.items():
                    for j, intervals in enumerate(variations):
                        midi_filename = f"{instrument_name}_{root}_{chord_type}_{octave}_{j}.mid"
                        midi_path = create_midi(root, intervals, instrument_name, midi_filename, octave)
                        if midi_path:
                            midi_files.append(midi_path)
    
    logging.info(f"Generated {len(midi_files)} MIDI files.")
    return midi_files


def convert_all_midi_to_wav(midi_files):
    """Convert all generated MIDI files to WAV format using multiple threads."""
    logging.info("Starting MIDI to WAV conversion...")
    with ThreadPoolExecutor() as executor:
        executor.map(convert_mid_to_wav, midi_files)
    logging.info("All conversions completed.")


if __name__ == "__main__":
    # Generate MIDI files for all chords and instruments
    midi_files = generate_chords()
    
    # Convert generated MIDI files to WAV format
    convert_all_midi_to_wav(midi_files)
