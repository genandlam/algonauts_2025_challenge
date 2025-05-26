import os
from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd
import h5py
import string
from scipy.stats import pearsonr
from nilearn import plotting
import zipfile
from sklearn.model_selection import check_cv
from voxelwise_tutorials.utils import generate_leave_one_run_out
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.backend import set_backend
import pickle

root_data_dir = '/home/ckj24/genevieve/algonauts_2025_challenge/data/algonauts_2025.competitors'
initial_dir = os.getcwd() 
backend = set_backend("torch_cuda", on_error="warn")
print(backend)
subjects = [1,2,3]  # ["1", "2", "3", "5"] 

modality = "all"  #["visual", "audio", "language", "all"]

excluded_samples_start = 5  # { min:0, max:20, }

excluded_samples_end = 1  # {min:0, max:20}

hrf_delay = 0  #{ min:0, max:10}

stimulus_window = 1  #{min:1, max:20}

movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05","friends-s06", "movie10-bourne", "movie10-figures", "movie10-life","movie10-wolf"] 

movies_val = [ "movie10-wolf"]

movies_test = [ "movie10-wolf"] 

def load_stimulus_features(root_data_dir, modality, train_test):
    """
    Load the stimulus features.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    modality : str
        Used feature modality.

    Returns
    -------
    features : dict
        Dictionary containing the stimulus features.

    """

    features = {}

    ### Load the visual features ###
    if modality == 'visual' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir,'stimulus_features', 'pca',
            'friends_movie10', 'visual', train_test)
        features['visual'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Load the audio features ###
    if modality == 'audio' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir,'stimulus_features', 'pca',
            'friends_movie10', 'audio', train_test)
        features['audio'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Load the language features ###
    if modality == 'language' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
            'friends_movie10', 'language', train_test)
        features['language'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Output ###
    return features

def load_fmri(root_data_dir, subject):
    """
    Load the fMRI responses for the selected subject.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    subject : int
        Subject used to train and validate the encoding model.

    Returns
    -------
    fmri : dict
        Dictionary containing the  fMRI responses.

    """

    fmri = {}

    ### Load the fMRI responses for Friends ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
                            #'_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_dir = os.path.join(root_data_dir,#,'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_friends = h5py.File(fmri_dir, 'r')
    for key, val in fmri_friends.items():
        fmri[str(key[13:])] = val[:].astype(np.float32)
    del fmri_friends

    ### Load the fMRI responses for Movie10 ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_dir = os.path.join(root_data_dir,#'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_movie10 = h5py.File(fmri_dir, 'r')
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10
    # Average the fMRI responses across the two repeats for 'figures'
    keys_all = fmri.keys()
    figures_splits = 12
    for s in range(figures_splits):
        movie = 'figures' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]
    # Average the fMRI responses across the two repeats for 'life'
    keys_all = fmri.keys()
    life_splits = 5
    for s in range(life_splits):
        movie = 'life' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    ### Output ###
    return fmri

def align_features_and_fmri_samples(features, fmri, excluded_samples_start,
    excluded_samples_end, hrf_delay, stimulus_window, movies):
    """
    Align the stimulus feature with the fMRI response samples for the selected
    movies, later used to train and validate the encoding models.

    Parameters
    ----------
    features : dict
        Dictionary containing the stimulus features.
    fmri : dict
        Dictionary containing the fMRI responses.
    excluded_trs_start : int
        Integer indicating the first N fMRI TRs that will be excluded and not
        used for model training. The reason for excluding these TRs is that due
        to the latency of the hemodynamic response the fMRI responses of first
        few fMRI TRs do not yet contain stimulus-related information.
    excluded_trs_end : int
        Integer indicating the last N fMRI TRs that will be excluded and not
        used for model training. The reason for excluding these TRs is that
        stimulus feature samples (i.e., the stimulus chunks) can be shorter than
        the fMRI samples (i.e., the fMRI TRs), since in some cases the fMRI run
        ran longer than the actual movie. However, keep in mind that the fMRI
        timeseries onset is ALWAYS SYNCHRONIZED with movie onset (i.e., the
        first fMRI TR is always synchronized with the first stimulus chunk).
    hrf_delay : int
        fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
        that reflects changes in blood oxygenation levels in response to
        activity in the brain. Blood flow increases to a given brain region in
        response to its activity. This vascular response, which follows the
        hemodynamic response function (HRF), takes time. Typically, the HRF
        peaks around 5–6 seconds after a neural event: this delay reflects the
        time needed for blood oxygenation changes to propagate and for the fMRI
        signal to capture them. Therefore, this parameter introduces a delay
        between stimulus chunks and fMRI samples for a better correspondence
        between input stimuli and the brain response. For example, with a
        hrf_delay of 3, if the stimulus chunk of interest is 17, the
        corresponding fMRI sample will be 20.
    stimulus_window : int
        Integer indicating how many stimulus features' chunks are used to model
        each fMRI TR, starting from the chunk corresponding to the TR of
        interest, and going back in time. For example, with a stimulus_window of
        5, if the fMRI TR of interest is 20, it will be modeled with stimulus
        chunks [16, 17, 18, 19, 20]. Note that this only applies to visual and
        audio features, since the language features were already extracted using
        transcript words spanning several movie chunks (thus, each fMRI TR will
        only be modeled using the corresponding language feature chunk). Also
        note that a larger stimulus window will increase compute time, since it
        increases the amount of stimulus features used to train and test the
        fMRI encoding models.
    movies: list
        List of strings indicating the movies for which the fMRI responses and
        stimulus features are aligned, out of the first six seasons of Friends
        ["friends-s01", "friends-s02", "friends-s03", "friends-s04",
        "friends-s05", "friends-s06"], and the four movies from Movie10
        ["movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"].

    Returns
    -------
    aligned_features : float
        Aligned stimulus features for the selected movies.
    aligned_fmri : float
        Aligned fMRI responses for the selected movies.

    """

    ### Empty data variables ###
    aligned_features = []
    aligned_fmri = np.empty((0,1000), dtype=np.float32)

    ### Loop across movies ###
    for movie in movies:

        ### Get the IDs of all movies splits for the selected movie ###
        if movie[:7] == 'friends':
            id = movie[8:]
        elif movie[:7] == 'movie10':
            id = movie[8:]
        movie_splits = [key for key in fmri if id in key[:len(id)]]

        ### Loop over movie splits ###
        for split in movie_splits:

            ### Extract the fMRI ###
            fmri_split = fmri[split]
            # Exclude the first and last fMRI samples
            fmri_split = fmri_split[excluded_samples_start:-excluded_samples_end]
            aligned_fmri = np.append(aligned_fmri, fmri_split, 0)

            ### Loop over fMRI samples ###
            for s in range(len(fmri_split)):
                # Empty variable containing the stimulus features of all
                # modalities for each fMRI sample
                f_all = np.empty(0)

                ### Loop across modalities ###
                for mod in features.keys():

                    ### Visual and audio features ###
                    # If visual or audio modality, model each fMRI sample using
                    # the N stimulus feature samples up to the fMRI sample of
                    # interest minus the hrf_delay (where N is defined by the
                    # 'stimulus_window' variable)
                    if mod == 'visual' or mod == 'audio':
                        # In case there are not N stimulus feature samples up to
                        # the fMRI sample of interest minus the hrf_delay (where
                        # N is defined by the 'stimulus_window' variable), model
                        # the fMRI sample using the first N stimulus feature
                        # samples
                        if s < (stimulus_window + hrf_delay):
                            idx_start = excluded_samples_start
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s + excluded_samples_start - hrf_delay \
                                - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        # In case there are less visual/audio feature samples
                        # than fMRI samples minus the hrf_delay, use the last N
                        # visual/audio feature samples available (where N is
                        # defined by the 'stimulus_window' variable)
                        if idx_end > (len(features[mod][split])):
                            idx_end = len(features[mod][split])
                            idx_start = idx_end - stimulus_window
                        f = features[mod][split][idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())

                    ### Language features ###
                    # Since language features already consist of embeddings
                    # spanning several samples, only model each fMRI sample
                    # using the corresponding stimulus feature sample minus the
                    # hrf_delay
                    elif mod == 'language':
                        # In case there are no language features for the fMRI
                        # sample of interest minus the hrf_delay, model the fMRI
                        # sample using the first language feature sample
                        if s < hrf_delay:
                            idx = excluded_samples_start
                        else:
                            idx = s + excluded_samples_start - hrf_delay
                        # In case there are fewer language feature samples than
                        # fMRI samples minus the hrf_delay, use the last
                        # language feature sample available
                        if idx >= (len(features[mod][split]) - hrf_delay):
                            f = features[mod][split][-1,:]
                        else:
                            f = features[mod][split][idx]
                        f_all = np.append(f_all, f.flatten())

                 ### Append the stimulus features of all modalities for this sample ###
                aligned_features.append(f_all)

    ### Convert the aligned features to a numpy array ###
    aligned_features = np.asarray(aligned_features, dtype=np.float32)

    ### Output ###
    return aligned_features, aligned_fmri

def compute_encoding_accuracy(fmri_val, fmri_val_pred):
    """
    Compare the  recorded (ground truth) and predicted fMRI responses, using a
    Pearson's correlation. The comparison is perfomed independently for each
    fMRI parcel. The correlation results are then plotted on a glass brain.

    Parameters
    ----------
    fmri_val : float
        fMRI responses for the validation movies.
    fmri_val_pred : float
        Predicted fMRI responses for the validation movies
    subject : int
        Subject number used to train and validate the encoding model.
    modality : str
        Feature modality used to train and validate the encoding model.

    """

    ### Correlate recorded and predicted fMRI responses ###
    encoding_accuracy = np.zeros((fmri_val.shape[1]), dtype=np.float32)
    for p in range(len(encoding_accuracy)):
        encoding_accuracy[p] = pearsonr(fmri_val[:, p],
            fmri_val_pred[:, p])[0]
    mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)

    return mean_encoding_accuracy

def model(cv):
    alphas = np.logspace(1, 20, 20)
    pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=False),
            Delayer(delays=[1, 2, 3, 4]),
            KernelRidgeCV(
                alphas=alphas,
                cv=cv,
                solver_params=dict(
                    n_targets_batch=500, n_alphas_batch=5, n_targets_batch_refit=100
                ),
            ),
        )
    return pipeline

if __name__== "__main__":
    """
    Main function to run the encoding model training and validation.

    Returns
    -------
    None
        The function does not return anything, but it prints the results of the
        encoding model training and validation.
    """

    # Load the stimulus features for training and validation
    features = load_stimulus_features(root_data_dir, modality, 'features_train.npy')
    #features_val = load_stimulus_features(root_data_dir, modality, 'features_test.npy')

    # Load the fMRI responses for each subject
    fmri_subjects = {subject: load_fmri(root_data_dir, subject) for subject in subjects}

    for i in subjects:
        # Align the stimulus features with the fMRI responses for the training movies
        features_train, fmri_train = align_features_and_fmri_samples(features,fmri_subjects[i],
            excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window,
            movies_train)
        if i == 1:
            features_train_all=features_train
            fmri_train_all=fmri_train
        else:
            features_train_all=np.concatenate((features_train_all, features_train), axis=0)
            fmri_train_all=np.concatenate((fmri_train_all,fmri_train), dtype=np.float32)

    # Print the shape of the training fMRI responses and stimulus features: note
    # that the two have the same sample size!
    print("Training fMRI responses shape:")
    print(fmri_train_all.shape)
    print('(Train samples × Parcels)')
    print("\nTraining stimulus features shape:")
    print(features_train_all.shape)
    print('(Train samples × Features)')
    # Load the fMRI responses for validation
    # Align the stimulus features with the fMRI responses for the validation movies
    fmri_val = load_fmri(root_data_dir, 3)
    features_val, fmri_val = align_features_and_fmri_samples(features, fmri_val,
        excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window,
        movies_val)
    # Print the shape of the test fMRI responses and stimulus features 
    print("Validation fMRI responses shape:", fmri_val.shape)
    print('(Validation samples × Parcels)')
    print("\nValidation stimulus features shape:", features_val.shape)
    print('(Validation samples × Features)')
    # Remove unused variables from memory
    del features, fmri_subjects
    backend = set_backend("torch_cuda", on_error="warn")
    run_onsets = [x for x in np.arange(0, fmri_train_all.shape[0], stimulus_window)][:-1]
    n_samples_train = fmri_train_all.shape[0]
    cv = generate_leave_one_run_out(n_samples_train, run_onsets)
    cv = check_cv(cv) 
    print("running model")
    model = model(cv)
    model.fit(features_train_all, fmri_train_all)
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)

    # with open('model.pkl', 'rb') as f:
    #    clf2 = pickle.load(f)
    
    # Predict the fMRI responses for the validation set
    fmri_val_pred = model.predict(features_val)
    fmri_val_pred=fmri_val_pred.numpy()
    np.savetxt('test1.txt', fmri_val_pred)
    #b = np.loadtxt('test1.txt', dtype=int)
    mean_acc=compute_encoding_accuracy(fmri_val, fmri_val_pred, 3, modality)
    with open("demofile.txt", "a") as f:
        f.write("Subject: " + str(3) + "\n")
        f.write("Modality: " + modality + "\n")
        f.write("Mean encoding accuracy: " + str(mean_acc) + "\n")
    print("Mean encoding accuracy:", mean_acc)
    print("Results saved to demofile.txt")
 