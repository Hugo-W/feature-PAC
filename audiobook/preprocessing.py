#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:19:47 2020

@author: hugwei
"""
import os
from copy import deepcopy
import fire
import numpy as np
import mne
import time

from .utils import DATA_PATH, CH_TYPES, subjects, data_folders, read_head_position

def load_raw_data(subject):
    """
    Loads the raw data together with the events from the .ds CTF folder.
    
    Parameters
    ----------
    subject : str
        Subject name.
        
    Returns
    -------
    raw : mne.io.Raw
        The raw data.
    events : ndarray (nevents, 3)
        Events structure (first column onset, second duration, third id).
    """
    # If two recordings exist, will simply append one to the other
    if len(data_folders[subject]) > 1:
        # load each and append
        ds_folder = os.path.join(DATA_PATH, 'raw', (subject if ('-' in subject) else subject[:3] + '-' + subject[3:]), 'ses-meg01', 'meg', data_folders[subject][0])
        raw = mne.io.read_raw_ctf(ds_folder, clean_names=True)
        for ds in data_folders[subject][1:]:
            ds_folder = os.path.join(DATA_PATH, 'raw', (subject if ('-' in subject) else subject[:3] + '-' + subject[3:]), 'ses-meg01', 'meg', ds)
            raw.append(mne.io.read_raw_ctf(ds_folder, clean_names=True))
    else:
        ds_folder = os.path.join(DATA_PATH, 'raw', (subject if ('-' in subject) else subject[:3] + '-' + subject[3:]), 'ses-meg01', 'meg', data_folders[subject][0])
        raw = mne.io.read_raw_ctf(ds_folder, clean_names=True)
    raw = raw.set_channel_types(CH_TYPES)
    
    # Drop all unused "EEG" channels
    eeg_ch_names = [raw.ch_names[k] for k in mne.pick_types(raw.info, eeg=True)]
    raw = raw.drop_channels(eeg_ch_names)
    
    events = mne.find_events(raw) 
    return raw, events

def add_headshape_points(raw, subject, overwrite=False):
    """
    This function will modify the :py:class:`~mne.channels.DigMontage` in the
    raw instance to add the Polhemus extra headshape points.
    These points are useful to coregister MRI coordinate frame with the MEG
    device coordinate frame (e.g. when using ``mne coreg``).

    Parameters
    ----------
    subject : str
        Subject id.
    raw : mne.io.Raw
        Raw instance.
    overwrite : bool (optional)
        Whether to force resetting montage in raw, note that if it seems that
        no headshape points are present in the raw instance,
        the montage will be updated anyway. (Default: False)

    Returns
    -------
    raw : mne.io.Raw
        The modified raw instance (modified in place).

    Notes
    -----
    .. note::
        The **current version** simply adds the headhsape points, transformed in head
        coordinates using Polhemus' fiducials positions, to the list of digitised
        points in raw.info['dig'] (which is used by mne to extract the DigMontage).
    .. note::
        The commented code ("**old version**") had a different behaviour:
        
        The dig points of head shape are transformed in "head" coordinate using the 
        fiducials point digitized by Polhemus.
        However those points are then passed as HPI (head position indicator) along
        with fiducials from CTF recording to the montage.
        By default we keep the polhemus fiducials rather than CTF ones, the latter
        are simply converted to extra HPI points too.
    """
    # Read Polhemus data
    dig = read_head_position(subject.replace('-', ''), return_dig=True)
    # Read original montage (CTF)
    orig = raw.get_montage()
    
    # Only reset montage if overwrite is True or if less dig points are present
    if len(orig.dig) < len(dig.dig) or overwrite:
        # Transform to head coordinate frame
        dig_head = mne.channels.montage.transform_to_head(dig)
        raw.info['dig'] += dig_head.dig[3:] # not adding fiducials, only HSP
        
        ''' Old version
        # Fids -> HPI
        for d in orig.dig[:3]:
            d['kind'] = mne.channels.montage.FIFF.FIFFV_POINT_HPI
        # Add CTF montage to Polhemus
        dig.dig += orig.dig
        dig_fids = deepcopy(dig.dig[:3])
        # Add Polhemus' fiducials as extra HPIs
        for d in dig_fids:
            d['kind'] = mne.channels.montage.FIFF.FIFFV_POINT_HPI
        dig.dig += dig_fids
        # Transform to head coordinate frame
        dig_head = mne.channels.montage.transform_to_head(dig)
        # Match EEG names
        dig_head.ch_names = orig.ch_names
        # Remove EEG channels (those are not used, and not digitized, just dummy 
        # dummy information that might lead to error in head origin fitting algos?)
        for k in range(len(dig_head.ch_names)): dig_head.dig.pop()
        dig_head.ch_names = []
        # Set montage onto raw object
        raw.set_montage(dig_head, on_missing='warn')
        '''
    return raw

def maxwellfilter(raw, find_bad_channels=True, apply_filter=False, save_dir='./'):
    """
    Remove software gradient compensation and then apply maxwell filter to data.
    The maxwell filter can optionally be used to proceed to an automated "bad"
    channel repair.
    
    Parameters
    ----------
    raw : mne.Raw
        data
    find_bad_channels: bool
        (Defulat: True) Whether to automatically repair flat and noisy channels.
        If done, the name of those channels will be save on disk to allow further
        inspection (make sure :py:obj:`save_dir` is valid path).
    remove_gradcomp : bool
        (Default: True) Whether to remove the software based gradient compensation.
    
    Returns
    -------
    The compensated data.
    """
    former_compgrade = 0
    if raw.compensation_grade != 0:
        former_compgrade = raw.compensation_grade 
        raw = raw.apply_gradient_compensation(0)
        
    if find_bad_channels:
        noisy, flats = mne.preprocessing.find_bad_channels_maxwell(raw, skip_by_annotation='BAD_Behav_Resp',
                                                                   ignore_ref=True, duration=10.)
        with open(os.path.join(save_dir, 'flat_channels.csv'), 'w') as f:
            f.write(','.join(flats))
        with open(os.path.join(save_dir, 'noisy_channels.csv'), 'w') as f:
            f.write(','.join(noisy))
            
        # Mark bad channels
        raw.info['bads'] = raw.info['bads'] + noisy + flats
    
    #mf_kwargs = dict(origin=(0., 0., 0.), st_duration=10.)
    if apply_filter:
        chpi_dic = mne.chpi.extract_chpi_locs_ctf(raw)
        pos = mne.chpi.compute_head_pos(raw.info, chpi_dic)
        return mne.preprocessing.maxwell_filter(raw, head_pos=pos, skip_by_annotation='BAD_Behav_Resp')
    else:
        if former_compgrade != 0:
            raw = raw.apply_gradient_compensation(former_compgrade)
        return raw

def split_audioLoc_audioBook(raw, events, savedir=None):
    """
    Split the two parts of data. Audio localizer data will be saved as is (not
    resampled) while the audiobook part will be passed on (returned by this function)
    for further preprocessing.

    Parameters
    ----------
    raw : mne.Raw
        Raw data.
    events : ndarray (nevents, 3)
        Events data.
    savedir  : string
        Path to where the audioLoc data will be saved. If None (default), they 
        will not be saved.

    Returns
    -------
    raw : mne.Raw
        Raw data of audiobook part.
    events : ndarray
        New events structure (without audio loc, and shifted in time).
    """
    event_ids = events[:, -1]
    last_audioloc = np.argwhere(np.logical_and(10 < event_ids, event_ids < 24))[-1][0]
    first_audiobook = last_audioloc + 1
    
    # AudioLoc
    if savedir is not None:
        print("Saving audioLoc data...")
        raw_audioloc = raw.copy().crop(tmin=0., tmax=raw.times[events[last_audioloc, 0]] +0.5)
        mne.event.write_events(os.path.join(savedir, 'audioLocalizer-eve.fif'), events[:first_audiobook])
        raw_audioloc.save(os.path.join(savedir, 'audioLocalizer-raw.fif'), overwrite=True)
        
    # Audiobook
    times_session = raw.times.copy()
    # Crop with 3 seconds margin on each side
    raw.crop(tmin=raw.times[events[first_audiobook, 0]]- 3,
                    tmax=raw.times[events[-1, 0]] + 3)
    # shift events:
    events = mne.event.shift_time_events(events, None, -(times_session[events[first_audiobook, 0]]- 3), raw.info['sfreq'])[first_audiobook:]
    return raw, events 

def annotate_badsegment(raw, events, margin=2, orig_time=None):
    """Will make a rough annotations by tagging segment of data as "BAD_Behav_Resp"
    for portions of data between stimulus presentation.
    
    We assume that the subject movements (e.g. during behavioural response) occur
    between the end of stimulus presentation and the onset of the resting state
    recording preceding stimulus presentation (10 sec, with three seconds delay 
    between end of resting state period and onset of sound).

    Trigger identification:
        - all resting state onset are single digit 0<id<10
        - Stimulus onset 1xx
        - Stimulus offset 2xx
        
    See also
    --------
    :func:`utils.story_to_triggers`
    """
    fs = float(raw.info['sfreq'])
    #start_resting_state_sample = np.where(events[:, 2] <10)[0]
    story_end_sample = np.where(events[:, 2] > 200)[0]
    
    # Ideally this annotation task occurs after data split AudioLoc/Audiobook
    # So we only have 3sec before and after firt resting state and last sory
    # end respecitvely, we won't annotate those as "bad" segments
    
    onsets = events[story_end_sample[:-1]][:, 0] / fs # last story end is NOT an onset of bad segment
    #duration = events[start_resting_state_sample[1:]][:, 0]/fs - margin - (onsets + margin)
    duration = events[story_end_sample[:-1]+1][:, 0]/fs - margin - (onsets + margin)
    
    return raw.set_annotations(mne.Annotations(onsets + margin, duration, 'BAD_Behav_Resp', orig_time=orig_time))

def process(subject, annotate=True, maxwell=False,
            find_bad_channels=True, resample=200., n_jobs=1):
    """
    Apply the processing chain to a single subject.

    Parameters
    ----------
    subject : str
        Subject to be processed.
    annotate : bool, optional
        Whether to annotate segment outside listening time (as "BAD_Behav_Resp").
        The default is True.
    maxwell : bool, optional
        Whether to apply maxwell filtering. The default is True.
    find_bad_channels : bool, optional
        See :func:`maxwellfilter`. The default is True.
    resample : float
        New sampling rate (data will be low-passed to avoid aliasing).

    Returns
    -------
    None.

    """
    # Dirty hack to accomodate for discrepancy in folder naming
    if '-' in subject:
        subject_dir = subject
    else:
        subject_dir = subject[:3] + '-' + subject[-3:]
    savedir = os.path.join(DATA_PATH, 'processed', subject_dir, 'meg')
    # Create "meg" subdirectory if it does not already exist
    os.makedirs(savedir, exist_ok=True)
    
    # Loading data
    tstart = time.time()
    # raw = get_raw(subject)
    # events = story_data[subject]['events']
    raw, events = load_raw_data(subject)
    # Save raw original events (in case sub msec resolution needed for timing
    # of epoching, etc...)
    mne.event.write_events(os.path.join(savedir, 'raw-eve.fif'), events)
    
    # Add Polhemus data into it (if not already present)
    if len(raw.info['dig']) <= 11:
        raw = add_headshape_points(raw, subject)
    
    # Extract only audiobook part
    raw, events = split_audioLoc_audioBook(raw, events, savedir=savedir)
    tend = time.time()
    print(f"\nData loading and splitting done.\nElapsed time: {tend - tstart:.3f} sec\n")
        
    # Resample
    print("Resampling...")
    raw, events = raw.resample(resample, events=events, n_jobs=n_jobs, verbose=True, npad='auto')
    tend2 = time.time()
    print(f"\nResampling done.\nElapsed time: {(tend2 - tend)//60:.0f} min and {(tend2 - tend)%60:.1f} sec\n")
    
    if annotate:
        print("Adding annotations...")
        raw = annotate_badsegment(raw, events)
        
    if maxwell or find_bad_channels:
        tstart_maxwell = time.time()
        print("Maxwell filtering and detection of flat and noisy channels...")
        raw = maxwellfilter(raw, find_bad_channels=find_bad_channels, apply_filter=maxwell, save_dir=savedir)
        tend_maxwell = time.time()
        print(f"\Maxwell done.\nElapsed time: {(tend_maxwell - tstart_maxwell)//60:.0f} min and {(tend_maxwell - tstart_maxwell)%60:.1f} sec\n")
    
    # gradient Compensation grade 3
    # raw = raw.apply_gradient_compensation(3)
    
    # Pick channels (with exclude=[] I am keeping the bad channels and can interpolate them later?)
    audio_chans = raw.copy().pick_channels(['UADC001', 'UADC002'])
    raw = raw.pick_types(meg=True, stim=True, eeg=False, eog=True, ecg=True, ref_meg='auto', exclude=[])
    # Write to disk
    raw.save(os.path.join(savedir, 'audioBook-raw.fif'), overwrite=True)
    audio_chans.save(os.path.join(savedir, 'audioBook-audioChannels.fif'), overwrite=True)
    mne.event.write_events(os.path.join(savedir, 'audioBook-eve.fif'), events)
    tend = time.time()
    print(f"\Preprocessing done.\nTotal elapsed time: {(tend - tstart)//60:.0f} min and {(tend - tstart)%60:.1f} sec\n")

def main(subject=None, annotate=True, maxwell=False,
         find_bad_channels=True, resample=200.):
    """
    Run the processing pipeline on all subjects.

    Parameters
    ----------
    subject : None or str
        If None all subjects are processed, else subject to be processed.
    annotate : bool, optional
        Whether to annotate segment outside listening time (as "BAD_Behav_Resp").
        The default is True.
    maxwell : bool, optional
        Whether to apply maxwell filtering. The default is False.
    find_bad_channels : bool, optional
        See :func:`maxwellfilter`. The default is True.
    resample : float
        New sampling rate (data will be low-passed to avoid aliasing).

    Returns
    -------
    None.

    """
    if subject is None:
        for subj in subjects:
            print("\n" + "=" * 50)
            print("Processing subject " + subj)
            print("=" * 50  + "\n")
            try:
                process(subj, annotate, maxwell, find_bad_channels, resample)
            except Exception as e:
                print("An errror occured while processing subject %s"%subj)
                print(e)
                print("Skipping...\n")
                continue
    else:
        print("\n" + "=" * 50)
        print("Processing subject " + subject)
        print("=" * 50  + "\n")
        process(subject, annotate, maxwell, find_bad_channels, resample)
        
if __name__ == '__main__':
    fire.Fire(main)
    
    
