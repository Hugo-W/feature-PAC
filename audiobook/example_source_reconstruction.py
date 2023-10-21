#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 20:52:47 2021

Notes:
    
    - ``mne coreg`` to create -trans.fif file
    - ``mne watershed_bem``to compute the requried inner_skull.surf etc. (4 min)
    - ``mne make_scalp_surfaces`` also generates one of the surface file but not all (1 min)
    

@author: hugwei
"""
import os
import matplotlib.pyplot as plt
from mne.minimum_norm import make_inverse_operator, apply_inverse

import mne

#%% Data Paths

subjects_dir = '/project/3027007.01/processed/'
subject = 'sub-020'
fig_dir = '/home/lacnsg/hugwei/Documents/Scripts/AudioBook_Experiment/figures/%s'%subject
os.makedirs(fig_dir, exist_ok=True)

raw_fname = '/project/3027007.01/processed/%s/meg/audioLocalizer-raw.fif'%subject
eve_fname = '/project/3027007.01/processed/%s/meg/audioLocalizer-eve.fif'%subject
trans_fname = "/project/3027007.01/processed/%s/meg/%s-trans.fif"%(subject, subject)

#%% Loading

# Example on Audio Localiser data for one subject
#================================================

raw = mne.io.read_raw(raw_fname)
events = mne.read_events(eve_fname)

# Quick processing
raw.load_data()
raw = raw.filter(1, 40)

#%% Epoching

baseline = (None, 0)
reject = dict(mag=4e-12)

epochs = mne.Epochs(raw, events, event_id=None, tmin=-0.3, picks='meg',
                    reject=reject, baseline=baseline)

# Resample for lighter processing later on (better to do it after epoching)
epochs.load_data()
epochs = epochs.resample(200, n_jobs=4)

#%% Compute noise covariance
data_cov = mne.compute_covariance(epochs, tmin=0.01, tmax=0.25,
                                  method='empirical')
noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)

# # Plot
# fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)
# fig_cov, fig_spectra = mne.viz.plot_cov(data_cov, raw.info)
# plt.show()

#%% Compute CSD (for DICS beamformer later)

#csd = mne.time_frequency.csd_fourier(epochs, fmin=3, fmax=35)

#%% Evoked

evoked = epochs.average()
evoked.plot(time_unit='s')
f = evoked.plot_topomap(times=[0., 0.07, 0.115, 0.22], time_unit='s', sphere=0.095)
f.savefig(os.path.join(fig_dir, 'evoked_topomap.png'))

 # Have a glance at the whitened data
# evoked.plot_white(noise_cov, time_unit='s')
del epochs, raw  # to save memory

#%% Forward model
#================

#%% 1. Sanity checks
#-----------------

# First we look at the MRI data (for pleasure)
# mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
#                  brain_surfaces='white', orientation='coronal')

# Second we can double check the coregistration (that's important!)
info = mne.io.read_info(raw_fname)
# mne.viz.plot_alignment(info, trans_fname, subject=subject, dig=True,
#                        meg=['helmet', 'sensors'], subjects_dir=subjects_dir,
#                        surfaces='head-dense')

# %%2. Source Space (a few seconds)
#-----------------

# Can be volumetric/discrete (filling up the space with some grid) or surface based
# (following the cortical sheet surface).

# Surface-based
src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch',
                             subjects_dir=subjects_dir)
print(src)

# Volumetric
# sphere = (0.0, 0.0, 0.04, 0.09) # for volumetric we define a sphere in which to compute dipoles
# vol_src = mne.setup_volume_source_space(subject, sphere=sphere', add_dist='patch',
                             # subjects_dir=subjects_dir)
# The above give a very rough placement of sourcces as some will be outside the grey matter
# A more precise grid can be computed using BEM surfaces as follow:
surface = os.path.join(subjects_dir, subject, 'bem', 'inner_skull.surf')
vol_src = mne.setup_volume_source_space(subject, subjects_dir=subjects_dir,
                                        surface=surface) # default to 5mm spacing

# We can double check the computed source location on the BEM surfaces
# mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
#                  brain_surfaces='white', src=src, orientation='coronal')

# 3D brain
# fig = mne.viz.plot_alignment(subject=subject, subjects_dir=subjects_dir,
#                              surfaces='white', coord_frame='head',
#                              src=src)
# mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
#                     distance=0.30, focalpoint=(-0.03, -0.01, 0.03))

#%% 3. Forward solution
#--------------------

# We just need to give a CONDUCTIVITY model (3 layers for EEG, 1 is sufficient for MEG).
conductivity = (0.3,)
# conductivity = (0.3, 0.006, 0.3)  # for three layers
# First we compute the actual boundary element model, and then solution 
# the latter only depend on MRI geometry so far
# This will need the inner_skull.surf file created by `mne watershed_bem` 
model = mne.make_bem_model(subject=subject, ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

# Now we compute the actual LEADFIELD (or GAIN, or simply FORWARD) matrix:
# This takes in account the -trans matrix mapping coord. from MEG to MRI
fwd = mne.make_forward_solution(raw_fname, trans=trans_fname, src=src, bem=bem,
                                meg=True, eeg=False, mindist=5.0, n_jobs=1,
                                verbose=True)
print(fwd)
fwd_vol = mne.make_forward_solution(raw_fname, trans=trans_fname, src=vol_src, bem=bem,
                                meg=True, eeg=False, mindist=5.0, n_jobs=1,
                                verbose=True)
print(fwd_vol)

# Some vertices are sometimes removed because to far from skull, some function will need
# fwd['src'] or inv['src'] to account for that removal
print(f'Before: {src}')
print(f'After:  {fwd["src"]}')

# Let's save the forward model
mne.write_forward_solution('/project/3027007.01/processed/%s/meg/%s-oct-6-fwd.fif'%(subject, subject), fwd)
mne.write_forward_solution('/project/3027007.01/processed/%s/meg/%s-volumetric-5mm-fwd.fif'%(subject, subject), fwd_vol)


#%% Inverse Operator

# We are gonna compute one inverse operator for a GIVEN noise covariance matrix
# plus other parameters. Of course it might be convenient to save to disk
# different inverse operator
# One important parameter here is "loose" which constrain the orientation of dipoles
# loose = 0. => fixed-orientation
# loose= 1. = > Free orientation
inverse_operator = make_inverse_operator(
    evoked.info, fwd, noise_cov, loose=0.2, depth=0.8)
del fwd
mne.minimum_norm.write_inverse_operator('/project/3027007.01/processed/%s/meg/%s-oct-6-inv.fif'%(subject, subject),
                                        inverse_operator)

#%% Inverse Solution

# Different approaches: MNE, LORETA, dSPM, Beamforming (and of course dipole fit :D)

# Minimum norm types
#---------------------
method = "dSPM" # MNE, dSPM, eLORETA, sLORETA
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(evoked, inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)

# Visualise source time cousrses
fig, ax = plt.subplots()
ax.plot(1e3 * stc.times, stc.data[::100, :].T)
ax.set(xlabel='time (ms)', ylabel='%s value' % method)

# Plot residuals
fig, axes = plt.subplots(1, 1)
evoked.plot(axes=axes)
for line in axes.lines:
    line.set_color('#98df81')
residual.plot(axes=axes)

# Visualise activation pattern at the peak (marking the vertex of the peak too)
vertno_max, time_max = stc.get_peak(hemi='rh')
surfer_kwargs = dict(
    hemi='both', subjects_dir=subjects_dir, surface='white',
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)
brain.save_image(os.path.join(fig_dir, 'dSPM.png'))
#%% Beamforming
#------------
# Beamformer are basically a kind of spatial filering
# To compute those spatial filters we need the covariance of the data (and hence all epochs)
# The forward model used for beamformer though is usually a VOLUMETRIC one, as
# only estimating cortical surface activation might misrepresent the data.

from mne.beamformer import make_lcmv, apply_lcmv, make_dics, apply_dics
# Another kind would DICS (time-frequency beamformer)

# Now we will look at LCMV (minimum variance) source reconstruction
fwd_vol = mne.read_forward_solution('/project/3027007.01/processed/%s/meg/%s-volumetric-5mm-fwd.fif'%(subject, subject))
filters = make_lcmv(evoked.info, fwd_vol, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='max-power', #pick_ori='vector' for volume vector source estimate
                    weight_norm='unit-noise-gain', rank=None)

# The following line is slow (one spatial filter computation per frequency bin!)
# dics = make_dics(evoked.info, fwd_vol, csd)

# You can save the filter for later use with:
filters.save('/project/3027007.01/processed/%s/meg/filters-vol-lcmv.h5'%subject, overwrite=True)

# Those filters can be applied to covariance matrices, evoked data, epochs and Raw!
stc = apply_lcmv(evoked, filters, max_ori_out='signed')

# Visualisation
lims = [0.3, 0.5, 0.6]
kwargs = dict(src=vol_src, subject=subject, subjects_dir=subjects_dir,
              initial_time=0.087, verbose=True)
# For surface source estimate
#stc.plot(surface='white', clim=dict(kind='value', lims=lims), **kwargs)
# For volume source estimate
f = stc.plot(mode='glass_brain', clim=dict(kind='value', lims=lims), **kwargs)
f.savefig(os.path.join(fig_dir, 'lcmv_beamformer.png'))
# param "mode='glass_brain' works only for volumetric source estimate (or mode='stat_map' for overlay on MRI slices)

# This is also only for volum.
# brain = stc.plot_3d(
#     clim=dict(kind='value', lims=lims), hemi='both',
#     views=['coronal', 'sagittal', 'axial'], size=(800, 300),
#     view_layout='horizontal', show_traces=0.3,
#     brain_kwargs=dict(silhouette=True), **kwargs)

#%% Morph to average Brain
