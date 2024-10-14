#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 10 Nov 2021

@author: hugwei
"""
import os
import pandas as pd


DATA_PATH = '../data/meg/'  # This relative path works if the notebook is ran from the notebook folder, use absolute path if ran from another folder
STIM_PATH = '../data/stim/'  #  same comment
print(f'Reading data and stim from {os.path.abspath(DATA_PATH)} and {os.path.abspath(STIM_PATH)} respectively')
CH_TYPES = {
            # EOG
            'EEG057':'eog', 
            'EEG058':'eog',
            # EKG
            'EEG059':'ecg',
            # STIM - audio
            'UADC001':'misc',
            'UADC002':'misc',
             # Triggers
            'UPPT001':'stim',
            'UPPT002':'resp', # response
             }

STORIES = pd.read_csv(os.path.join(STIM_PATH, 'story_ids.csv'), delim_whitespace=True, names=['id', 'filename', 'lang'])
subjects = [d for d in os.listdir(DATA_PATH) if (os.path.isdir(os.path.join(DATA_PATH, d)) and d.startswith('sub'))]

