#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/07/21

Entry script for creating necessary files for source space analysis. Namely we 
create here the source space for an individual subject based on a morphed version
of fsaverage, then calculate BEM solutions and forward lead-field matrix.

Notes:
    
    - ``mne coreg`` to create -trans.fif file
    - ``mne watershed_bem``to compute the requried inner_skull.surf etc. (4 min)
    - ``mne make_scalp_surfaces`` also generates one of the surface file but not all (1 min)
    

@author: Hugo Weissbart <hugo.weissbart@donders.ru.nl>
"""
import os
import fire

import mne
from mne import pick_types, pick_info
from mne.io import read_info
from mne import read_source_spaces, read_forward_solution, read_cov
from mne.beamformer import make_lcmv
from mne.forward import convert_forward_solution

import conpy

subjects_dir = '/project/3027007.01/processed/'
trans_fname = lambda subject: "/project/3027007.01/processed/%s/meg/%s-trans.fif"%(subject, subject)

def compute_source_space(subject, from_fsaverage=True, subjects_dir=subjects_dir,
                         spacing='oct6'):
    """
    Compute the source space for a given subject.
    Can be morphed from fsaverage (useful to keep vertices indices in order
    for connectivity analysis) or simply created from subject specific meshes
    directly.
    """
    # Create BEM directories if needed
    os.makedirs(os.path.join(subjects_dir, 'fsaverage', 'bem'), exist_ok=True)
    os.makedirs(os.path.join(subjects_dir, subject, 'bem'), exist_ok=True)
    
    if from_fsaverage:
        if not os.path.exists(os.path.join(subjects_dir, 'fsaverage', 'bem', '%s-src.fif'%spacing)):
            # Define source space on average brain
            src_avg = mne.setup_source_space('fsaverage', spacing=spacing)
            mne.write_source_spaces(os.path.join(subjects_dir, 'fsaverage', 'bem', '%s-src.fif'%spacing), src_avg)
        else:
            src_avg = mne.read_source_spaces(os.path.join(subjects_dir, 'fsaverage', 'bem', '%s-src.fif'%spacing))

        # Morph source space to individual subject
        src_sub = mne.morph_source_spaces(src_avg, subject_to=subject, subjects_dir=subjects_dir)
    else:
        src_sub = mne.setup_source_space(subject, spacing=spacing, subjects_dir=subjects_dir, add_dist=True)
    
    # Discard deep sources
    info = mne.io.read_info('/project/3027007.01/processed/%s/meg/audioBook-raw.fif'%subject)
    verts = conpy.select_vertices_in_sensor_range(
        src_sub, dist=0.07, info=info, trans=trans_fname(subject))
    src_sub = conpy.restrict_src_to_vertices(
        src_sub, verts)
    
    return src_sub


def compute_bem_solution(subject, subjects_dir=subjects_dir):
    # Create a one-layer BEM model
    bem_model = mne.make_bem_model(
        subject, ico=4, conductivity=(0.3,), subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(bem_model)

    return bem

def compute_forward_solution(subject, src, bem, subjects_dir=subjects_dir):
    info = mne.io.read_info('/project/3027007.01/processed/%s/meg/audioBook-raw.fif'%subject)
    # Make the forward model
    fwd = mne.make_forward_solution(
        info, trans_fname(subject), src, bem, meg=True, eeg=False)
    
    # Only retain orientations tangential to a sphere
    # approximation of the head
    fwd = conpy.forward_to_tangential(fwd)
    return fwd

def get_lcmv_filters(subject, data_cov, noise_cov=None, fixed_orientation=False, bads=None,
                     pick_ori='max-power', reduce_rank=True, verbose=False):
    """
    Generate the LCMV beamformer filters for a given subject.
    This will use precomputed (loading from file) covariance matrices, unless a covariance
    matrix is passed as a parameter (instead of a filepath string).
    """
    info = read_info(f'/project/3027007.01/processed/{subject}/meg/audioBook-filtered-ICAed-raw.fif')
    picks = pick_types(info, meg=True, ref_meg=False, exclude=bads if bads is not None else [])
    info = pick_info(info, picks)
    fwd_file = os.path.join(subjects_dir, f'{subject}', 'meg', 'oct6-fwd.fif') # or sub-005-oct6-fwd.fif
    src_file = os.path.join(subjects_dir, f'{subject}', 'bem', 'oct6-src.fif') # or sub-005-oct6-src.fif
    fwd = read_forward_solution(fwd_file, verbose=False)
    src = read_source_spaces(src_file)
    if fixed_orientation:
        fwd = convert_forward_solution(fwd, surf_ori=True)
    
    # Load covariance matrices
    if isinstance(data_cov, str): data_cov = read_cov(data_cov)
    if noise_cov is not None and isinstance(noise_cov, str): noise_cov = read_cov(noise_cov)
    
    # Remove unwanted channels if any:
    if bads is not None:
        fwd.pick_channels([c for c in info.ch_names if c not in bads])
        data_cov.pick_channels([c for c in info.ch_names if c not in bads])
        if noise_cov is not None: noise_cov.pick_channels([c for c in info.ch_names if c not in bads])
    
    print("Computing beamformer filters...")
    return make_lcmv(info, fwd, data_cov, noise_cov=noise_cov, pick_ori=pick_ori, reduce_rank=reduce_rank, verbose=verbose)
    
def process(subject, subjects_dir=subjects_dir, spacing='oct6', from_fsaverage=True):
    if '-' not in subject: subject = subject[:3] + '-' + subject[3:]
    print("Computing source model")
    src = compute_source_space(subject, from_fsaverage, subjects_dir, spacing)
    print("Computing BEM solution")
    bem = compute_bem_solution(subject, subjects_dir)
    print("Computing forward model")
    fwd = compute_forward_solution(subject, src, bem, subjects_dir)
    
    # Saving to disk:
    print("Saving to disk...")
    mne.write_source_spaces(os.path.join(subjects_dir, subject, 'bem', '%s-src.fif'%spacing), src, overwrite=True)
    mne.write_bem_solution(os.path.join(subjects_dir, subject, 'bem', 'bem-sol.h5'), bem, overwrite=True)
    mne.write_forward_solution(os.path.join(subjects_dir, subject, 'meg', '%s-fwd.fif'%spacing), fwd, overwrite=True)


if __name__ == '__main__':
    fire.Fire(process)
