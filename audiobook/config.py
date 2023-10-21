#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 10 Nov 2021

@author: hugwei
"""
import os
import pandas as pd


DATA_PATH = '/project/3027007.01/'
STIM_PATH = '/project/3027007.01/Stimuli/'
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

