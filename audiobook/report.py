#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report making
=============

This script will generate basic HTML reports with:
    - raw data info
    - PSD
    - Covariances matrices (noise and normal)
    - projections if any
    - ICA ?
    - Alignment of trans- files
    - evoked for auditory localisers
    - corresponding source reconstructions
    
Created on Tue Jul 13 12:15:45 2021

@author: Hugo Weissbart <hugo.weissbart@donders.ru.nl>
"""
import os
import fire
from mne import Report

DATA_PATH = '/project/3027007.01/processed'

def process(subject):
    """
    Process a single subject.
    
    Will create a report in /project/3027007.01/analysis/reports/<subject>.html
    """
    info_fname = os.path.join(DATA_PATH, subject, 'meg',
                          'audioLocalizer-raw.fif')
    report = Report(verbose=True, title=subject, info_fname=info_fname,
                    subject=subject, subjects_dir=DATA_PATH,
                    raw_psd=True, projs=True)

    # Parse raw
    report.parse_folder(f'/project/3027007.01/processed/{subject}/meg/',
                        pattern=['*-raw.fif',
                                 '*-trans.fif',
                                 '*-cov.fif'],
                        render_bem=True)
    
    # Parse evoked/epochs and source stc
    
    # save
    report.save(f"/project/3027007.01/analysis/reports/{subject}.html", overwrite=True)

if __name__ == '__main__':
    fire.Fire(process)

