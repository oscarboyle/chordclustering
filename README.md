# chordclustering

MIR final project: chord classification using semisupervised clustering

## Overview

This repository investigates unsupervised approaches to chord recognition by directly learning harmonic structures from audio without extensive reliance on annotated data. We extract chroma features using a deep learning-based method and apply Gaussian Mixture Models to cluster these high-dimensional representations. The resultant clusters are analyzed both quantitatively—using metrics such as the Adjusted Rand Index, Normalized Mutual Information, and Fowlkes-Mallows Index—and qualitatively through t-SNE visualizations that reveal the underlying organization of chord spaces. Comparative evaluation against a supervised Random Forest baseline highlights that while unsupervised clustering can capture essential harmonic characteristics, its performance deteriorates in scenarios involving diverse chord types, inversions, and instrument timbres. These findings underscore the promise and limitations of unsupervised strategies in chord recognition, suggesting that further integration of invariance principles may enhance their applicability in low-annotation settings.

## Experiments

Three experiments have been run for this project:
* ```experiment_violin_major_chords.ipynb```: contains the analysis only for those samples of major chords played with the violin.
* ```experiment_violin_all_chords.ipynb```: contains the analysis only for those samples played with the violin, regardless the chord type.
* ```experiment_all_instruments.ipynb```: contains the analysis using the whole dataset.



