#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:19:47 2020

@author: hugwei
"""
import os
from typing import List, Union, Tuple
try:
    import h5py
    h5py_available = True
except ImportError:
    print("h5py not installed, some functions will not work.")
    h5py_available = False
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    tg_available = True
    import textgrid as tg
except ImportError:
    tg_available = False
    print("textgrid not installed, some functions will not work.")
from tqdm import tqdm

from scipy.signal import correlate
from scipy.io.wavfile import read as wavread
from scipy.stats import zscore

from pyeeg.utils import signal_envelope, lag_matrix, lag_span
import mne

from audiobook.config import DATA_PATH, STIM_PATH, CH_TYPES, STORIES, subjects
# DATA_PATH = '/project/3027007.01/'
# STIM_PATH = '/project/3027007.01/Stimuli/'
# CH_TYPES = {
#     # EOG
#     'EEG057':'eog', 
#     'EEG058':'eog',
#     # EKG
#     'EEG059':'ecg',
#     # STIM - audio
#     'UADC001':'misc',
#     'UADC002':'misc',
#     # Triggers
#     'UPPT001':'stim',
#     'UPPT002':'resp', # response
#     }
# STORIES = pd.read_csv(os.path.join(STIM_PATH, 'story_ids.csv'), delim_whitespace=True, names=['id', 'filename', 'lang'])
# subjects = [d for d in os.listdir(DATA_PATH) if (os.path.isdir(os.path.join(DATA_PATH, d)) and d.startswith('sub'))]

data_folders = {}
for subj in subjects:
    subj_dashed = subj[:3] + '-' + subj[3:]
    ds_folders = [d for d in os.listdir(os.path.join(DATA_PATH, 'raw', subj_dashed, 'ses-meg01', 'meg')) if d.endswith('ds')]
    if len(ds_folders) > 1: # several recordings available, checking if some are "empty"
        # Due to errors during experiments, some ds folder contain 0byte .meg4 data, we can discard those
        for ds in ds_folders:
            # while some participants had to pause recording for some reasons, in that
            # case we have two recordings that will need to be concatenated
            megfile = [fname for fname in os.listdir(os.path.join(DATA_PATH, 'raw', subj_dashed, 'ses-meg01', 'meg', ds)) if 'meg4' in fname][0]
            if os.path.getsize(os.path.join(DATA_PATH, 'raw', subj_dashed, 'ses-meg01', 'meg', ds, megfile)) <= 8: #1 byte
                ds_folders.remove(ds)
                flag_both_valid = False
    
    data_folders[subj] = ds_folders
    
def correct_subject_name(subj):
    if '-' not in subj:
        subj = subj[:-3] + '-' + subj[-3:]
    return subj
   
SUBJECTS = [correct_subject_name(s) for s in subjects]
    
def get_bad_channels(subject, flat=True, noisy=True):
    chans = []
    if flat and os.path.exists(os.path.join(DATA_PATH, 'processed', correct_subject_name(subject), 'meg', 'flat_channels.csv')):
        with open(os.path.join(DATA_PATH, 'processed', correct_subject_name(subject), 'meg', 'flat_channels.csv'), 'r') as f:
            chans += f.read().split(',')
    if noisy and os.path.exists(os.path.join(DATA_PATH, 'processed', correct_subject_name(subject), 'meg',  'noisy_channels.csv')):
        with open(os.path.join(DATA_PATH, 'processed', correct_subject_name(subject), 'meg', 'noisy_channels.csv'), 'r') as f:
            chans += f.read().split(',')
    # Clean empty lines
    chans = [c for c in chans if len(c)>0]
    return chans

def story_to_triggers(story):
    """
    Returns trigger onset and offset values.

    Parameters
    ----------
    story : str
        Story part name (e.g. "Anderson_S01_P01_normalized")

    Returns
    -------
    tuple (int, int)
        Onset and offset event ids for this story.

    """
    sid = str(STORIES.loc[STORIES.filename == story  +'.wav', 'id'].to_list()[0])
    return int('1' + sid[::-1]), int('2' + sid[::-1])

def find_lags(signal, pattern):
    """
    Find the lag at which to signal aligned most (based on cross-correlation).

    Parameters
    ----------
    signal : ndarray
        Longer signal.
    pattern : ndarray
        Signal pattern to be matched.

    Returns
    -------
    int
        Lag (sample for ``signal`` at which ``pattern`` starts.

    """
    return np.argmax(correlate(signal, pattern, 'valid'))

def get_audiopath(story):
    sid = str(STORIES.loc[STORIES.filename == story  +'.wav', 'id'].to_list()[0])
    fname = 'story%s_part%s_normalized.wav'%(sid[0], sid[1])
    return os.path.join(STIM_PATH, 'materials', fname)

def get_acoustic_envelope(story, map_names=True, srate=120):
    """
    Get the acoustic envelope from a given story. This loads the wav-file and 
    compute the envelope from the raw audio. The envelope is computed using 
    the ``abs()`` of the Hilbert transformed signal. This broadband power is 
    then passed trhough a low-pass filter (20 Hz cutoff) and compressed by 
    raising values to the power :math:`\\frac{1}{3}` which mimicks the 
    non-linear compression of human early auditory processing.

    Parameters
    ----------
    story : str
        e.g. "Anderson_S01_P01_normalized".
    srate : float, optional
        Target sampling rate. The default is 120.

    Returns
    -------
    y : ndarray (nsamples,)
        Acoustic envelope.

    """
    fs, y = wavread(get_audiopath(story) if map_names else story)
    y = signal_envelope(y, fs, resample=srate, verbose=0)
    return  y

def get_boundaries(story, onset=True, phonemes=False):
    assert tg_available, "textgrid not installed, please install it to use this function."
    tgdata = tg.TextGrid.fromFile(os.path.join(STIM_PATH, 'Alignments', story + '.TextGrid'))
    intervals = tgdata.getFirst('MAU' if phonemes else 'ORT-MAU')
    wordlist = []
    times = []
    for k in range(len(intervals)):
        w, ton, toff = intervals[k].mark, intervals[k].minTime, intervals[k].maxTime
        if w is not None and (w != '<p:>') and (w != ''):
            wordlist.append(w)
            times.append(ton if onset else toff)
    return wordlist, times

story_data = {} # try to have this as a local variable for memoize_data?
def memoize_data(func):
    def helper(subj):
        if subj not in story_data:
            #ds_folder = os.path.join(DATA_PATH, subj, data_folders[subj])
            ds_folder = os.path.join(DATA_PATH, 'raw', subj[:3] + '-' + subj[3:], 'ses-meg01', 'meg', data_folders[subj][0])
            raw = mne.io.read_raw_ctf(ds_folder, clean_names=True)
            raw = raw.set_channel_types(CH_TYPES)
            story_data[subj] = {}
            story_data[subj]['raw'] = raw.copy()
            #story_data[subj]['events'] = mne.find_events(raw, shortest_event=1) # this takes time
            story_data[subj]['events'] = mne.find_events(raw) # this takes time
        return story_data[subj]['raw'] 
    return helper
            
            
@memoize_data
def get_raw(subject):
    """
    Loading data from disk.

    Parameters
    ----------
    subject : str
        Subject folder name.

    Returns
    -------
    raw : :class:`mne.io.Raw`
        Raw data.

    """
    return story_data[subject]['raw'] # data are not loaded at this level

def read_head_position(subject, return_dig=False, unit='m'):
    """
    Read head shape digitised position from Polhemus file.
    """
    data = pd.read_csv(os.path.join(DATA_PATH, subject, subject + '.pos'), delim_whitespace=True,
                skiprows=1, names=['pnts', 'x', 'y', 'z'])
    pos_col = ['x', 'y', 'z']
    if unit == 'cm':
        scale = 1e2
    elif unit == 'mm':
        scale = 1e3
    else:
        scale = 1
    
    nasion = data.loc[data.pnts == 'nasion', pos_col].to_numpy().ravel()/1e2
    lpa = data.loc[data.pnts == 'left', pos_col].to_numpy().ravel()/1e2
    rpa = data.loc[data.pnts == 'right', pos_col].to_numpy().ravel()/1e2
    hsp = data.loc[data.pnts.str.match(r'\d'), pos_col].to_numpy()/1e2
    
    dig =  {'nasion': nasion, 'rpa': rpa, 'lpa': lpa, 'hsp': hsp}
    
    if return_dig:
        return mne.channels.make_dig_montage(**dig)
        
    return {k: v*scale for k,v in dig.items()}
                
def get_storyPartData(subject, story_id=1, story_part=1):
    trigger_start = int('1' + str(story_part) + str(story_id))
    raw = get_raw(subject)
    events = story_data[subject]['events']
    tstart = events[np.argwhere(events[:, 2] == trigger_start), 0].squeeze()/raw.info['sfreq']
    tend = events[np.argwhere(events[:, 2] == trigger_start+100), 0].squeeze()/raw.info['sfreq']
    if not np.any(events[:, 2] == trigger_start):
        raise ValueError("Missing events for story %d part %d"%(story_id, story_part))
    raw = raw.copy().crop(tmin=tstart, tmax=tend-1) # remove 1sec here cause we have a delay in the presentation code
    return raw

def load_and_process(raw, new_fs=120, highpass=1, lowpass=12, picks=None):
    raw.load_data()
    raw.pick_types(meg=True, ref_meg=False)
    raw.resample(new_fs)
    raw.filter(highpass, lowpass)
    return raw
    
def load_stimSound(story_id=1, story_part=1):
    trig = str(story_id) + str(story_part)
    fname = STORIES.loc[STORIES.id==int(trig), 'filename'].iloc[0]
    fs, y = wavread(os.path.join(DATA_PATH, 'Stimuli', fname))
    return y, fs

def load_raw(subject, events=True, ica=False, picks=None) -> Union[mne.io.Raw, Tuple[mne.io.Raw, np.ndarray]]:
    """
    Read raw data from subject file. If ICA filters exist will apply removal
    or load existing fitlered "ICAed" data if present.
    """
    if '-' not in subject: subject = subject[:3] + '-' + subject[3:]
    data_folder = os.path.join(DATA_PATH, 'processed', subject, 'meg')
    if ica:
        if os.path.exists(os.path.join(data_folder, 'audioBook-filtered-ICAed-raw.fif')):
            raw = mne.io.read_raw(os.path.join(data_folder, 'audioBook-filtered-ICAed-raw.fif'), preload=True, verbose=False)
        else:
            raw = mne.io.read_raw(os.path.join(data_folder, 'audioBook-raw.fif'), preload=True, verbose=False)
            ica = mne.preprocessing.ica.read_ica('audioBook-ica.fif')
            raise NotImplementedError("Need to implement the rejection in this function")
    else:
        raw = mne.io.read_raw(os.path.join(data_folder, 'audioBook-raw.fif'), preload=True, verbose=False)
        
    if picks is not None:
        raw = raw.pick(picks)
    if events:
        events = mne.read_events(os.path.join(data_folder, 'audioBook-eve.fif'), verbose=False)
        # Need to shift the first sample according to the original crop!
        events[:, 0] += raw.first_samp
        return raw, events
    else:
        return raw
    

def extract_resting_states(subject, story_id=None, ica=False, filter=(0.3, 40), picks=None):
    """
    Extract all (or one) resting state period(s) for a given subject.
    
    Parameters
    -----------
    subject : str or mne.io.Raw
        Subject id
    story_id : None or str
        If None will extract all resting state, if story_id is given,
        then a string is expected as "ij" where i is the story id and j the
        story part.
        
    Returns
    -------
    epochs : mne.Epochs
        Resting state period (10 sec epochs)
    """
    if isinstance(subject, mne.io.Raw):
        raw = subject
        if picks is not None: raw = raw.pick(picks)
        events = mne.read_events('/'.join(raw.filenames[0].split('/')[:-1]) + '/audioBook-eve.fif')
        events[:, 0] += raw.first_samp
    else:
        raw, events = load_raw(subject, events=True, ica=ica, picks=picks)
    if filter is not None:
        raw = raw.filter(*filter)
    
    #events[:, 0] += raw.first_samp # (now this is done in load_raw)
    if story_id is None: # Extract all resting state epochs
        epochs = mne.Epochs(raw, events, event_id=list(range(1, 7)),
                            tmin=0., tmax=10., baseline=None, on_missing='ignore', )
    else:
        epochs = mne.Epochs(raw, events, event_id=100 + int(story_id[::-1]),
                            tmin=-12., tmax=-2., baseline=None)
    return epochs

def extract_story_parts_data(subject, filter=(0.3, 40), ica=False, picks=None) -> List[mne.io.Raw]:
    """
    Extract each story part data and return a list of correspponding mne.Raw
    instances.
    Will take each story in the order they appear in `STORIES`.

    Parameters
    ----------
    subject: str | instace of mne.Raw
        Either the subject id ("sub-005") or directly the raw data if preloaded.
    """
    if type(subject) is str:
        raw, events = load_raw(subject, events=True, ica=ica, picks=picks)
    else:
        assert isinstance(subject, mne.io.Raw), "Wrong argument, should be an instance of Raw or a subject identifier"
        raw = subject
        events = mne.find_events(raw)
        if picks is not None: raw = raw.pick(picks)
        #events = mne.read_events('/'.join(raw.filenames[0].split('/')[:-1]) + '/audioBook-eve.fif')
        #events[:, 0] += raw.first_samp
    if filter is not None:
        raw = raw.filter(*filter)
        
    events[:, 0] -= raw.first_samp # now this is done in load_raw directly, NEED TO REMOVE IT AS I AM TAKING BELOW THE TSTART TEND AS INDICES
    raw_list = []
    for story in STORIES.filename:
        trig_on, trig_off = story_to_triggers(story[:-4])
        tstart = events[np.argwhere(events[:, 2] == trig_on), 0]/raw.info['sfreq']
        tend = events[np.argwhere(events[:, 2] == trig_off), 0]/raw.info['sfreq']
        # we take the last assuming that for such subject a problem
        # occured in the first presented story with the double event
        tstart = tstart[-1]
        tend = tend[-1]
        raw_list.append(raw.copy().crop(tmin=tstart.squeeze(), tmax=tend.squeeze(), include_tmax=False))
    return raw_list

def concatenate_raws_and_epoch(raws, lang='both', duration=5.):
    """
    Take a list of raw object, concatenate into one long raw instance.
    Then the long continuous instance is "chopped" into equal length epochs.
    Boundaries are annotated as BAD, so when epoching will be dropped.
    
    One can select only French or only Dutch story parts, or both (default).
    """
    raw = mne.concatenate_raws([r for r,l in zip(raws, STORIES.lang) if (l==lang or lang=='both')])
    epochs = mne.make_fixed_length_epochs(raw, duration=5.)
    return epochs

def compute_noise_cov(subject, filter=(0.3, 40), zscored=False, ica=False, picks=None):
    """
    Compute a covariance matrix on the resting state periods of a subject data.
    """
    epochs = extract_resting_states(subject, filter=filter, ica=ica, picks=picks)
    epochs.load_data()
    #if zscored:

    #epochs = epochs.drop_bad()
    # epochs = epochs.filter(*filter)
    return mne.compute_covariance(epochs)

def compute_data_cov(subject, filter=(0.3, 40), ica=False, picks=None):
    """
    Compute a covariance matrix on the story parts of a subject data.
    """
    concatenated = mne.concatenate_raws(extract_story_parts_data(subject, filter, ica=ica, picks=picks))
    return mne.compute_raw_covariance(concatenated)

def write_hdf_raws(meg_data, fpath='/project/3027007.01/analysis/Hugo/Data/raws.h5', subj=None, srate=50, fband='delta', overwrite=False):
    """
    Write raw data to hdf5 file.
    If meg_data is a dict, then the keys are the subject ids and the values
    are the corresponding raw data.

    Parameters
    ----------
    meg_data: dict | list
        Either a dict with subject ids as keys and raw data as values, or a list
        of raw data.
    fpath: str
    subj: str
        If meg_data is a list, then this argument is required to specify the
        subject id.
    srate: int
        The sampling rate of the data.
    fband: str
        The frequency band of the data.
    overwrite: bool
        If True, will overwrite the file if it already exists.
    """
    assert h5py_available, "h5py not installed, please install it to use this function."
    with h5py.File(fpath, mode='a') as f:
        if str(srate) not in f.keys():
            gp = f.create_group(str(srate))
        else:
            gp = f.require_group(str(srate))
        if fband not in gp.keys():
            gpfreq = gp.create_group(fband)
        else:
            gpfreq = gp.require_group(fband)
        if isinstance(meg_data, dict):
            for subj, data in tqdm(meg_data.items()):
                s = gpfreq.require_group(subj)
                for k, d in enumerate(data):
                    if STORIES.loc[k, 'filename'].split('.')[0] not in s:
                        s.create_dataset(STORIES.loc[k, 'filename'].split('.')[0], data=d._data)
        elif isinstance(meg_data, list):
            assert subj is not None, "If subject level data supplied, please inform subject name"
            s = gpfreq.require_group(subj)
            for k, d in enumerate(meg_data):
                if STORIES.loc[k, 'filename'].split('.')[0] not in s:
                    s.create_dataset(STORIES.loc[k, 'filename'].split('.')[0], data=d._data)
                else:
                    if overwrite:
                        if s[STORIES.loc[k, 'filename'].split('.')[0]].shape != d._data.shape:
                            del s[STORIES.loc[k, 'filename'].split('.')[0]]
                            s.create_dataset(STORIES.loc[k, 'filename'].split('.')[0], data=d._data)
                        else:
                            s[STORIES.loc[k, 'filename'].split('.')[0]][:] = d._data
                    else:
                        print(f"Dataset {STORIES.loc[k, 'filename'].split('.')[0]} already exists for subject {subj} and frequency band {fband}. Use overwrite=True to overwrite.")
            
                    
def read_hdf_raws(subject, story, srate=50, fband='delta',
                  fpath='/project/3027007.01/analysis/Hugo/Data/raws.h5'):
    
    assert h5py_available, "h5py not installed, please install it to use this function."
    with h5py.File(fpath, mode='r') as f:
        return f[str(srate)][fband][subject][story][()]
    

########### MAIN IS AN EXAMPLE OF TRF ON EACH STORY PART + ACCUMULATING ######
if __name__ == '__main__':
    lags = lag_span(-0.2, 0.6, 120)
    subj = subjects[2]
    if not os.path.exists(os.path.join('figures', subj)):
        os.mkdir(os.path.join('figures', subj))
    XtX = np.zeros((96, 96))
    XtY = np.zeros((96, 269))
    alpha = 10000
    for story_parts in STORIES.id:
        print("\nNew story part")
        story_id = int(str(story_parts)[0])
        part_id = int(str(story_parts)[1])
        try:
            raw = get_storyPartData(subj, story_id=story_id, story_part=part_id)
        except ValueError:
            print("This story part was not in data.. skipping...")
            continue
        print("Processing")
        raw = load_and_process(raw)
        print("Loading stim")
        y, fs = load_stimSound(story_id, part_id)
        env = signal_envelope(y, fs, resample=120)
        print("Accumulating matrices")
        X = lag_matrix(zscore(env), lags, filling=0.)
        Y = zscore(raw._data.T.copy()[:len(X), :])
        if len(Y) < len(X): X = X[:len(Y), :]
        XtX += X.T @ X
        XtY += X.T @ Y
        
        # Single story TRF
        betas = np.linalg.inv(X.T@X + alpha*10*np.eye(X.shape[1])) @ X.T @ Y
        trf = mne.EvokedArray(betas.T, raw.info, tmin=-0.2)
        trf.plot(spatial_colors=True)
        plt.savefig('figures/%s/%s_trf_story_%d.png'%(subj, subj, story_parts))
        plt.close()
        
    # Compute TRF
    print("Computing TRF")
    betas = np.linalg.inv(XtX+ alpha*np.eye(X.shape[1])) @ XtY
    trf = mne.EvokedArray(betas.T, raw.info, tmin=-0.2)
    trf.plot(spatial_colors=True)
    trf.plot_joint(times=[0.11, 0.2])
    plt.savefig('figures/%s/%s_trf_global.png'%(subj, subj))
    plt.close()
    
