# %%
%load_ext autoreload
%autoreload 2

equivalent_aug_chords = {
    "E_Augmented": "C_Augmented",
    "Ab_Augmented": "C_Augmented",
    
    "F_Augmented": "C#_Augmented",
    "A_Augmented": "C#_Augmented",
    
    "F#_Augmented": "D_Augmented",
    "Bb_Augmented": "D_Augmented",
    
    "G_Augmented": "Eb_Augmented",
    "B_Augmented": "Eb_Augmented",
}

# %%
import json
import os
from madmom.audio.chroma import DeepChromaProcessor
from featureExtraction import chromagram
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


wavfolder = "generated_chords/wav"

processor = DeepChromaProcessor()


deepchromas = list()
chromas = list()

instruments = list()
chords = list()
octaves = list()
inversions = list()
img_paths = list()

for audio in tqdm(os.listdir(wavfolder)):
    if audio.endswith(".wav"):
        
        path = os.path.join(wavfolder, audio)
        
        chroma = processor(path)
        deepchromas.append(chroma)

        img_path = os.path.join("imgs", audio.replace(".wav", ".png"))
        if not os.path.exists(img_path) and 0:
            plt.imshow(chroma, origin='lower', aspect='auto', resample=True)
            plt.savefig(img_path, bbox_inches='tight')
        img_paths += [img_path] * chroma.shape[0]

        try:
            instrument, chord, octave, inversion = re.findall(r"(.*)_(.*_.*)_(\d)_(\d).wav", audio)[0]
            instruments += [instrument] * chroma.shape[0]
            chords += [chord] * chroma.shape[0]
            octaves += [octave] * chroma.shape[0]
            inversions += [inversion] * chroma.shape[0]
        except:
            pass

# Account for augmented chords simmetry
chords = [equivalent_aug_chords.get(chord, chord) for chord in chords]

# Build the data matrix
X = np.vstack(deepchromas)

# %%
%load_ext autoreload
%autoreload 2
from dataProcessing import build_chord_templates
chord_templates = build_chord_templates()
templates = np.array(list(chord_templates.values()))
plt.imshow(templates.T, origin='lower', aspect='auto', cmap='gray_r')
plt.title("Chord templates")
plt.xlabel("Chroma")
plt.ylabel("Template")
template_labels = list(chord_templates.keys())

# %% [markdown]
# ### Baseline: Random Forest

# %%
# Import random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

RF = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)

# Evaluate the model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
scores = cross_val_score(RF, X, chords, cv=cv, scoring='accuracy')
print(f"Mean accuracy: {np.mean(scores):.3f}")

# %% [markdown]
# ### Assess the optimal number of clusters based on coherence

# %%
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, silhouette_score
from sklearn.mixture import GaussianMixture

from tqdm import trange
rands = list()
mis = list()
fmis = list()
sil = list()
bics = list()
aics = list()



# %%

for n in trange(51, 60):
    #kmeans = KMeans(n_clusters=n, random_state=123)
    gmm = GaussianMixture(n_components=n, covariance_type='full', init_params='kmeans', max_iter=1000, random_state=42)
    clusters = gmm.fit_predict(X)

    # Compute AIC and BIC
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    bics.append(bic)
    aics.append(aic)
    rands.append(adjusted_rand_score(chords, clusters))
    mis.append(normalized_mutual_info_score(chords, clusters))
    fmis.append(fowlkes_mallows_score(chords, clusters))
    #sil.append(silhouette_score(X, clusters))

# %%

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].plot(rands, label="Adjusted Rand Index")
axs[0].plot(mis, label="Mutual Information Score")
axs[0].plot(fmis, label="Fowlkes-Mallows Index")

# Mean of all metrics
axs[0].plot(np.mean([rands, mis, fmis], axis=0), label="Mean")

# Add vline for optimal number of clusters
optim_n = np.argmax(np.mean([rands, mis, fmis], axis=0))
axs[0].axvline(optim_n, color='black', linestyle='--', label="Optimal number of clusters")

axs[1].plot(bics, label="BIC")
axs[1].plot(aics, label="AIC")
axs[1].plot(np.mean([bics, aics], axis=0), label="Mean")

# Add vline for optimal number of clusters
optim_n = np.argmax(np.mean([1-np.array(aics), 1-np.array(bics)], axis=0))
axs[1].axvline(optim_n, color='black', linestyle='--', label="Optimal number of clusters")

#plt.plot(sil, label="Silhouette Score")
axs[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

# %% [markdown]
# ### Fit the GMM with the optimal number of classes

# %%
gmm = GaussianMixture(n_components=optim_n, covariance_type='full', init_params="kmeans", max_iter=1000, random_state=42)
clusters = gmm.fit_predict(X)

# %% [markdown]
# ### Plot the clustering

# %%
from MulticoreTSNE import MulticoreTSNE as TSNE
import seaborn as sns

def plot_clusters(coords, template_coords, clusters):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=clusters, palette='tab20')
    #sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=clusters, palette='tab20')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    sns.scatterplot(x=template_coords[:, 0], y=template_coords[:, 1], color='black', s=100, marker='x')
    plt.show()

# Plot the cluster assignments
tsne = TSNE(n_jobs=-1)
coords = tsne.fit_transform(np.vstack([X, templates]))
sample_coords = coords[:-len(templates)]
template_coords = coords[-len(templates):]

plot_clusters(sample_coords, template_coords, clusters)

# %% [markdown]
# ### Assess the content of each cluster

# %%
import pandas as pd
# Count the proportion of each chord in each cluster
cluster_df = pd.DataFrame({'cluster': clusters, 'chord': chords})
cluster_df = cluster_df.groupby(['cluster'])["chord"].value_counts().reset_index()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(cluster_df)

# %% [markdown]
# ### Create cluster labels

# %%
from sklearn.cluster import KMeans

template_clusters = gmm.predict(templates)

cluster_mapping = dict()
for cluster, label in zip(template_clusters, template_labels):
    cluster_mapping[cluster] = label

# %%
len(np.unique(template_clusters))

# Find repeated template_clusters
from collections import Counter
c = Counter(template_clusters)
repeated_clusters = [cluster for cluster, count in c.items() if count > 1]
repeated_clusters

# %%
for c, l in zip(template_clusters, template_labels):
    print(c, l)

# %% [markdown]
# ### Compute the "accuracy" of the clustering

# %%
gmm_y = list(map(lambda x: cluster_mapping.get(x, "None"), clusters))

from sklearn.metrics import accuracy_score
print(accuracy_score(chords, gmm_y))

# %% [markdown]
# ### Try it with a sequence of chords

# %%
predictions = gmm.predict(X[:150])
plt.plot(predictions)

# %% [markdown]
# ### Interactive TSNE

# %%
from faerun import Faerun
f = Faerun(clear_color="#111111", view="front", coords=False)

instruments = np.array(instruments).flatten()
img_paths = np.array(img_paths).flatten()
cat_chords, chord_data = Faerun.create_categories(chords)
cat_clusters, cluster_data = Faerun.create_categories(gmm_y)
cat_instruments, instruments_data = Faerun.create_categories(instruments)

misclassified = (np.array(chords) == np.array(gmm_y)).astype(str)
cat_misclassified, misclassified = Faerun.create_categories(misclassified)

labels = list()

for i in range(len(chord_data)):
    labels.append(f'{chords[i]}_{cluster_data[i]}__{instruments_data[i]}__<img src="{img_paths[i]}" width=500px>')

f.add_scatter(
    "embeddings",
    {
        "x": coords[:, 0], 
        "y": coords[:, 1], 
        "c": [chord_data, cluster_data, instruments_data, misclassified],
        "labels": labels,
    },
    colormap=["tab20", "tab20", "tab20"],
    shader="smoothCircle",
    point_scale=2,
    max_point_size=8,
    has_legend=[True, True, True, True],
    legend_labels=[cat_chords, cat_clusters, cat_instruments, cat_misclassified],
    categorical=[True, True, True, True],
    series_title=["True class", "Cluster", "Instruments", "Correctly classified"],
    selected_labels=["Cluster", "Instruments"],
    label_index=0,
    title_index=0
)

f.plot("TSNE_GMM", template="default")

# %%
cat_instruments

# %% [markdown]
# # TODO
# 
# * Asignar a cada cluster su etiqueta según el template más cercano
# * Analizar los que están mal clasificados
# * Hacerlo solo con clarinete
# * Hacerlo con todos los audios de music21
# * Plottear el chord tracking para una secuencia de acordes
# * Aplicar HMM/viterbi decoding (opcional)
# * A ver si mejora poniendo un cluster más (para cuando no suena nada)

# %%
import numpy as np
from hmmlearn import hmm

n_states = 25  # Number of hidden states
n_observations = 25  # Number of observed classes

# Transition probability matrix (random uniform example)
transition_probs = np.full((n_states, n_states), 1/n_states)

# Emission probability matrix (random but normalized)
emission_probs = np.random.rand(n_states, n_observations)
emission_probs /= emission_probs.sum(axis=1, keepdims=True)  # Normalize rows

# Initial state probabilities (assuming equal probability)
start_probs = np.full(n_states, 1/n_states)

# Example observed sequence (indices of classes from 0 to 11)
observed_sequence = kmeans.predict(X[:100]).reshape(-1,1)

# Define and initialize HMM model
model = hmm.CategoricalHMM(n_components=n_states, n_iter=10000)
model.startprob_ = start_probs
model.transmat_ = transition_probs
model.emissionprob_ = emission_probs

# Apply Viterbi decoding
hidden_states = model.predict(observed_sequence)

print("Most likely hidden state sequence:", hidden_states)

plt.plot(hidden_states)


