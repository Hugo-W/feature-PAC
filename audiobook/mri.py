#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process MRI files:
    
    - convert DICOM to NII
    - Call to `recon-all`
    

Example
-------

To run `recon-all` on all subjects data using 8 parallel cpu cores (one per 
                                                                    subject):

``python mri.py --do_convert=False --do_recon=True --n_jobs=8``

To run only the conversion to nii format from DICOM images:

``python mri.py``


Created on Mon Mar 22 13:43:36 2021

@author: hugwei
"""
import os
import sys
import subprocess
import fire
from joblib import Parallel, delayed
import glob

from .utils import DATA_PATH, subjects

def convert2nii(subject):
    subj = 'sub-' + subject[-3:]
    # find anatomical MRI directory
    if os.path.exists(os.path.join(DATA_PATH, 'raw', subj, 'ses-mri01')):
       print("Found an MRI...")
       T1 = [d for d in os.listdir(os.path.join(DATA_PATH, 'raw', subj, 'ses-mri01')) if 't1' in d]
       if len(T1) > 0:
           T1 = T1[0]
           output_dir = os.path.join(DATA_PATH, 'raw', subj, 'anat')
           input_dir = os.path.join(DATA_PATH, 'raw', subj, 'ses-mri01', T1)
           os.makedirs(output_dir, exist_ok=True)
           cmd = ['dcm2niix', "-o", output_dir +'/',"-a","y", input_dir]
           print(f"""
===================================
Running dcm2niix for subject {subj}
===================================
>> {' '.join(cmd)}""")
           subprocess.call(cmd)
    else:
        print("No MRI for subject " + subject)
           
def run_reconall(subject):
    subj = 'sub-' + subject[-3:]
    try:
        if not os.path.exists(os.path.join(DATA_PATH, 'raw', subj, 'anat')):
            raise FileNotFoundError(".nii.gz anatomical MRI not found")
        root_dir = os.path.join(DATA_PATH, 'raw', subj, 'anat')
        niifiles = glob.glob(root_dir + '/*nii*')
        if len(niifiles) == 0: raise FileNotFoundError(".nii.gz missing")
    except FileNotFoundError:
        print(f"Error for subject {subj}")
        return 0
    cmd = ['recon-all',
           '-i', niifiles[0],
           '-s', subj + "-reconall",
           '-all', '-parallel']
    print(f"""
===================================
Running recon-all for subject {subj}
===================================
>> {' '.join(cmd)}
          """, file=sys.stderr)
    subprocess.call(cmd)
    return 1

def main(listsubjects=None, do_convert=True, do_recon=False, n_jobs=4):
    if listsubjects is None:
        listsubjects = subjects
    for sub in listsubjects:
        print("Subject: "+ sub)
        if do_convert: convert2nii(sub)
    if do_recon:
        os.environ['SUBJECTS_DIR'] = os.path.join(DATA_PATH, 'processed')
        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_reconall)(sub) for sub in listsubjects
            )
        
if __name__ == '__main__':
    fire.Fire(main)
