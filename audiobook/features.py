#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create and load stimulus features.

Created on Thu Nov  4 17:23:06 2021

@author: Hugo Weissbart <hugo.weissbart@donders.ru.nl>
"""
import os.path as op
import sys
import numpy as np

from tqdm import tqdm
import h5py

from audiobook.utils import  STORIES
from audiobook.utils import get_acoustic_envelope
from audiobook.text import get_wordlevel_aligned

storynames = [s[:-4] for s in STORIES.filename]

# Word level features available
wl_feats = {0: 'wordonsets',
            1: 'surprisal',
            2: 'entropy',
            3: 'KL',
            4: 'PE',
            5: 'wordfrequency',
            6: 'depth',
            7: 'close',
            8: 'open'}

wl_feats_id = {v: k for k, v in wl_feats.items()}

def generate_wordlevel_feats(story, srate=100, N=None):
    """
    Produce the word level feature matrix word onset aligned values (spike time
                                                                     series).
    The length (duration) of the time series is defined from the length of the 
    envelope at the specified sampling rate, or by the input argument `N`.
    
    The current word features are (in that order):
        
        - Always on feature
            - word onset
        - Valued features
            - surprisal
            - entorpy
            - KL divergence
            - prediction error (= surprisal/entropy)
            - word frequency
            - depth in constiutuency tree
            - number of closing branches for current word
            - number of opening branches for current word

    Parameters
    ----------
    story : str
        Story name.
    srate : float, optional
        Sampling rate of spiketime series. The default is 100.
    N : int, optional
        Length of time series, if None will compute it from length of envelope
        feature. The default is None.

    Returns
    -------
    wordlvl : ndarray (N, 5)
        Each column is one feature (see description).

    """
    if 'normalized' not in story:
        story_acc = story + '_normalized'
        story_bare = story
    else:
        story_acc = story # acoustic name
        story_bare = story.split('_normalized')[0] # bare name (without "normalized")
    if N is None:
        N = len(get_acoustic_envelope(story_acc, srate=srate))
    
    # TODO: the switch between old and new annot is hard-coded for no, need to 
    # allow the user to use an argument to chose between versions.
    # data = np.c_[get_wordlevel_aligned(story).loc[:, ['onset', 'surprisal', 'entropy', 'KL', 'PE']].fillna(0).values,
    #              get_wordlevel_aligned(story, annot='wf').loc[:, ['freq']].fillna(0).values,
    #              get_wordlevel_aligned(story, annot='tree').loc[:, ['depth', 'close', 'open']].fillna(0).values]
    
    # For now switch base dir and some parameters for naming for French or Dutch stories:
    if any(s in story_bare for s in ['ANGE', 'BALL', 'EAUV']): # then it's a French story part
        print("Loading and aligning French word-level features...")
        # tree features -> in All/story_bare-syntfeats.csv <-> aligned with tg from French_timed (_normalized-timed.csv)
        # WF feature -> in All/story_bare_normalized_apost-wf.csv <-> aligned with tg from French_timed
        # GPT2 -> in former annotations files (default parameters) <-> aligned with old timed data (project folder annotations)
        annot_dir = '/home/lacnsg/hugwei/Documents/Scripts/AudioBook_Experiment/Annotations_cleaned/All'
        tg_dir = '/home/lacnsg/hugwei/Documents/Scripts/AudioBook_Experiment/French_timed'
        params = {'tg_dir': tg_dir, 'annot_dir': annot_dir, 'transcripts':{'wordlevel':'', 'textgrid':'normalized'}}
        # Option 1 is to align DataFrames using the exact same TextGrid df for all features (as in Dutch below)
        # Option 2 if I use another -timed.csv for each features is to merge the df based on onset columns before concatenating all (for now I keep this to keep the GPT2 faithful to the alignment wihtout apostrophe separation)
        df_gpt = get_wordlevel_aligned(story_bare, annot='gpt2', transcripts='normalized').loc[:, ['onset', 'surprisal', 'entropy', 'kl', 'pe']].fillna(0)
        df_wf = get_wordlevel_aligned(story_bare, annot='wf', **(params | {'transcripts':{'wordlevel':'normalized_apost', 'textgrid':'normalized'}})).loc[:, ['onset', 'logwf']].fillna(15) # filling NaN with max logwf more or less
        # Fix possible np.inf
        df_wf.loc[df_wf.logwf.isin([np.inf]), 'logwf'] = 15
        df_tree = get_wordlevel_aligned(story_bare, annot='tree', **params).loc[:, ['onset', 'depth', 'close', 'open']].fillna(0)
        df_all = df_gpt.merge(df_wf, on='onset').merge(df_tree, on='onset')
        data = df_all.loc[:, ['onset', 'surprisal', 'entropy', 'kl', 'pe', 'logwf', 'depth', 'close', 'open']].fillna(0.).values
    else:
        print("Loading and aligning Dutch word-level features...")
        base_dir = '/home/lacnsg/hugwei/Documents/Scripts/AudioBook_Experiment/Annotations_cleaned/All'
        params = {'tg_dir': base_dir, 'annot_dir': base_dir, 'transcripts':{'wordlevel':'cleaned_lined', 'textgrid':'cleaned'}}
        data = np.c_[get_wordlevel_aligned(story_bare, annot='gpt2', **params).loc[:, ['onset', 'surprisal', 'entropy', 'kl', 'pe']].fillna(0).values,
                     get_wordlevel_aligned(story_bare, annot='wf', **(params | {'transcripts':{'wordlevel':'cleaned_lined_nopunct', 'textgrid':'cleaned'}})).loc[:, ['logwf']].fillna(0).values,
                     get_wordlevel_aligned(story_bare, annot='tree', **(params | {'transcripts':{'wordlevel':'cleaned_line', 'textgrid':'cleaned'}})).loc[:, ['depth', 'close', 'open']].fillna(0).values]
    
    onsets = (data[:, 0] * srate).astype(int)
    wordlvl = np.zeros((N, data.shape[1]))
    wordlvl[onsets, 0] = 1
    for k in range(1, data.shape[1]):
        wordlvl[onsets, k] = data[:, k]
    return wordlvl

def write_h5_dataset(srate=100, datadir='/project/3027007.01/Stimuli/', fname='predictors.hdf5',
                     transcript_key='transcripts_v2'):
    """
    Create an HDF5 dataset with word and acoustic features for each story at 
    a given sampling rate.

    Parameters
    ----------
    srate : float, optional
        Sampling rate. The default is 100.
    datadir : str, optional
        Path to file. The default is '/project/3027007.01/Stimuli/'.
    fname : str, optional
        filename. The default is 'predictors.hdf5'.

    Returns
    -------
    None.

    """
    write_stories_to_dataset(storynames, srate=srate, datadir=datadir, fname=fname, transcript_key=transcript_key)
            
def write_stories_to_dataset(stories, srate=100, datadir='/project/3027007.01/Stimuli/', fname='predictors.hdf5',
                     transcript_key='transcripts_v2'):
    """
    Create an HDF5 dataset with word and acoustic features for some stories as passed as an 
    argument at a given sampling rate.

    Parameters
    ----------
    stories: list<str>
        List of story names to be (re)written in the hdf5 file.
    srate : float, optional
        Sampling rate. The default is 100.
    datadir : str, optional
        Path to file. The default is '/project/3027007.01/Stimuli/'.
    fname : str, optional
        filename. The default is 'predictors.hdf5'.

    Returns
    -------
    None.

    """
    print(f"Creating a dataset using sampling rate of {srate} Hz and {transcript_key} transcripts data.")
    print('Writing the following stories:')
    print(stories)
    with h5py.File(op.join(datadir, fname), mode='a') as f:
        root = f.require_group(transcript_key)
        gp = root.require_group(str(srate))
        acou = gp.require_group('acoustic')
        word = gp.require_group('wordlevel')
        for s in tqdm(stories, file=sys.stdout):
            env = get_acoustic_envelope(s, srate)
            wl = generate_wordlevel_feats(s, srate, len(env))
            # Overwrite dataset if it already exists
            if s in acou:
                del acou[s]
            if s in word:
                del word[s]
            acou.create_dataset(s, data=env)
            word.create_dataset(s, data=wl)
            
def read_h5_data(feat_type='acoustic', srate=100, transcript_key='transcripts_v2',
                 datadir='/project/3027007.01/Stimuli/', fname='predictors.hdf5',
                 stories=storynames):
    """
    Read, for a given sampling rate and if available, the stimulus representation:
    acoustic (continuous) or word-level data for all story parts.

    Parameters
    ----------
    feat_type : str, optional
        Type of stimulus representation. The default is 'acoustic'.
    srate : float, optional
        Sampling frequency of the time-aligned features. The default is 100.
    datadir : str, optional
        Path to file. The default is '/project/3027007.01/Stimuli/'.
    fname : str, optional
        filename. The default is 'predictors.hdf5'.

    Raises
    ------
    KeyError
        When desired sampling rate has no HDF5 dataset group.

    Returns
    -------
    out : list<ndarray>
        Numpy array of time aligned features for each story parts.

    """
    out = []
    with h5py.File(op.join(datadir, fname), 'r') as f:
        try:
            for s in stories:
                out.append(f[transcript_key][str(srate)][feat_type][s][()])
        except KeyError:
            raise KeyError(f"The required sampling rate ({srate}), feature type ({feat_type}) or {transcript_key} transcript family is not available")
    return out

def list_h5_data(fullpath='/project/3027007.01/Stimuli/predictors.hdf5'):
    """
    Describe the content of the h5 data file recursively.

    Parameters
    ----------
    fullpath : Path-like (str), optional
        The default is '/project/3027007.01/Stimuli/predictors.hdf5'.

    """
    def show_hierarchy(member, obj):
        #space =  '    '
        branch = '│   '
        tee =    '├── '
        #last =   '└── '
        basename = op.basename(member)
        depth = len(member.split('/'))
        if basename == member:
            print(member)
        else:
            if depth<=2:
                print(tee + basename)
            else:
                if isinstance(obj, h5py.Dataset):
                    print(branch*(depth-2) + tee + basename, obj.shape)    
                else:
                    print(branch*(depth-2) + tee + basename)
    with h5py.File(fullpath, 'r') as f:
        f.visititems(show_hierarchy)
            

def get_feature_signal(feats=['acoustic', 'wordonsets'], srate=100, transcripts='transcripts_v2',
                       stories=storynames, normalise='all', verbose=True):
    """
    Extract the final design matrix from features data (as stored in HDF5 file).

    Parameters
    ----------
    feats : list
        Can contain any of ['acoustic', 'wordonsets', 'wordfrequency', 'surprisal', 'entropy',
        'kl', 'PE', 'depth', 'close', 'open']
    srate : float
        Sampling rate requested.
    transcripts : str
        Which transcripts to use (old or new). See output of :func:`list_h5_data` to see available options.
    stories : list of str
        Which stories to extract
    normalise : str
        'all' or 'story'. Whether to normalise per story or across all stories jointly.

    Returns
    -------
    X : list of ndarray
    """
    assert normalise in ['all', 'story'], "Normalisation must per story ('story') or across all stories ('all')"
    if verbose: print("Load stimulus features")
    envs = read_h5_data(srate=srate, feat_type='acoustic', transcript_key=transcripts, stories=stories)
    wordlevels = read_h5_data(srate=srate, feat_type='wordlevel', transcript_key=transcripts, stories=stories)
    if verbose: print(f"Loading feature signal for : {feats}")
    X = []
    for k, s in enumerate(stories):
        x_ = []
        for f in feats:
            if f == 'acoustic':
                x_.append(envs[k])
            else:
                x_.append(wordlevels[k][:, wl_feats_id[f]])
        X.append(np.vstack(x_).T)
    # normalise feature to unit variance (so we do not remove the mean here...)
    var = np.var(np.vstack(X), 0)
    for x in X:
        if normalise=='story':
            x /= np.sqrt(np.var(x, 0)) # per story
        elif normalise=='all':
            x /= np.sqrt(var) # across all stories
    if verbose: print(f"Done. X shape: {X[0].shape}")
    return X