# chordclustering
MIR final project: chord classification using semisupervised clustering


### Modules

- *featureExtraction*: contains the functions to compute chromagram and chromagram onsets using cosine similarity
- *dataProcessing*: reads all .wav files from selected folder and computes their chromagram, if SEPARATE_CHORDS=True, computes separates the chromagram into separate chords
- *testing.ipynb*: for visualizing chromagrams

### How to run
chromagrams.json is already computed for testing if not:

- Download wav files you want to process and add them to selected folder in _dataProcessing.py_ 
- Run _dataProcessing.py_ 
- Visualize in testing.ipynb
