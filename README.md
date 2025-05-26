## The 2025 challenge

Encoding models of neural responses are increasingly used as predictive and explanatory tools in computational neuroscience ([Kay et al., 2008](https://doi.org/10.1038/nature06713); [Kell et al., 2018](https://doi.org/10.1016/j.neuron.2018.03.044); [Kriegeskorte and Douglas, 2019](https://doi.org/10.1016/j.conb.2019.04.002); [Naselaris et al., 2011](https://doi.org/10.1016/j.neuroimage.2010.07.073); [Tuckute et al., 2023](https://doi.org/10.1371/journal.pbio.3002366); [Van Gerven, 2017](https://doi.org/10.1016/j.jmp.2016.06.009); [Wu et al., 2006](https://doi.org/10.1146/annurev.neuro.29.051605.113024); [Yamins and DiCarlo, 2016](https://doi.org/10.1038/nn.4244)). They consist of algorithms, typically based on deep learning architectures, that take stimuli as input, and output the corresponding neural activations, effectively modeling how the brain responds to (i.e., encodes) these stimuli. Thus, the goal of the 2025 challenge is to provide a platform for biological and artificial intelligence scientists to cooperate and compete in developing cutting-edge functional magnetic resonance imaging (fMRI) encoding models. Specifically, these models should predict fMRI response to multimodal naturalistic movies, and generalize outside their training distribution.
The challenge is based on data from the Courtois Project on Neuronal Modelling ([CNeuroMod](https://www.cneuromod.ca/)), which has acquired the dataset that, to date, most intensively samples single-subject fMRI responses to a variety of naturalistic tasks, including movie watching. For more details on the challenge you can visit the [website](https://algonautsproject.com/), read the [paper](https://doi.org/10.48550/arXiv.2501.00504), or watch [this video](https://youtu.be/KvLDpsIO2eg).

# Algonauts 2025 Challenge - Project Overview

This repository contains code and resources for the Algonauts 2025 Challenge, which involves building encoding models to predict fMRI responses to naturalistic stimuli such as movies. The project leverages multimodal features (visual, audio, and language) extracted from movie stimuli to train and evaluate encoding models.

## Files in the Repository

### 1. `new.ipynb`
This notebook is the main pipeline for the project. It includes the following functionalities:

- **Feature Extraction**:
  - Extracts **audio features** using the Wav2Vec2 model.
  - Extracts **visual features** using a pre-trained SlowFast video model.
  - Handles **language features** using pre-computed embeddings.

- **fMRI Data Loading**:
  - Loads fMRI responses for multiple subjects and movies.
  - Aligns fMRI responses with stimulus features using parameters like HRF delay and stimulus window.

- **Model Training**:
  - Trains encoding models (e.g., Ridge regression) to predict fMRI responses from stimulus features.
  - Saves trained models for future use.

- **Validation**:
  - Aligns validation data and evaluates the trained models on unseen movies.
  - Computes encoding accuracy using Pearson's correlation and visualizes results on a brain atlas.

- **Submission Preparation**:
  - Prepares predictions for Friends Season 7 episodes for submission to the challenge.

### 2. `analysis.ipynb`
This notebook focuses on analyzing the fMRI data and its functional connectivity. Key functionalities include:

- **fMRI Data Loading**:
  - Loads fMRI data for a specific session (e.g., `s01e09b`) from HDF5 files.

- **Stimulus Timing**:
  - Extracts movie chunk start times based on the fMRI repetition time (TR).

- **Functional Connectivity**:
  - Computes the correlation matrix of fMRI responses using the Nilearn library.
  - Visualizes the correlation matrix and connectome using Schaefer 2018 brain atlas.

- **Voxel Analysis**:
  - Computes explainable variance for each voxel.
  - Plots the time series of the voxel with the highest explainable variance.

- **Confound Analysis**:
  - Evaluates the presence of confounders in the fMRI data and their impact on functional connectivity.

