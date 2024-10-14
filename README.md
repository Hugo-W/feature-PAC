# Feature-informed Phase-amplitude coupling estimation

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![DOI](https://zenodo.org/badge/708032317.svg)](https://zenodo.org/doi/10.5281/zenodo.12667683)

This repository contains the necessary code and a minimal working example of preprocessed data to reproduce the results of:
> " The structure and statistics of language jointly shape cross-frequency neural dynamics during spoken language comprehension", H. Weissbart & AE. Martin, 2024 ([Nature Communications]([https://www.biorxiv.org/content/10.1101/2023.10.06.561087v1.full](https://www.nature.com/articles/s41467-024-53128-1))).
> [![DOI](https://img.shields.io/badge/DOI-10.1038/s41467--024--53128--1-blue.svg)](https://doi.org/10.1038/s41467-024-53128-1)

## Content

### Outline

- [Content](#content)
- [Setting up](#installation-and-usage)
- [Dependencies](#requirements)
- [Data](#dataset)
- [Reference](#citation-)

Folders content:
- `data`
  - preprocessed MEG file: sample files are provided in as a figshare dataset (see [`data/meg/README.md`](https://github.com/Hugo-W/feature-PAC/blob/main/data/meg/README.md) for more information)
  - Stimulus:
    - Audio files: `data/stim/` contains the audio files used in the experiment
    - Annotations: All annotations are stored in the `data/stim/annotations` folder
- `audiobook`: a custom made python package containing some utility functions specific to our "audiobook MEG" dataset (see [the dataset section](#raw-data) for more information)
- `pyeeg`: a custom made python package containing some utility functions for computing TRFs (listed as submodule here)
- `notebooks`: example Jupyter Notebook reproducing some of the main results and figures of the paper

## Installation and usage

### Install dependencies

The overall installation process is simple and fast, given that the main dependencies are already pre-installed in the provided conda environment. Installing from scratch in a new conda environment should take only a few minutes. It is mainly about setting up the right path for some of the utility functions.

Firstly, clone the repository and navigate to the root folder of the repository:

```bash
git clone https://github.com/Hugo-W/feature-PAC
cd feature-PAC 
```

#### 1. MNE

The main requirement is the `mne` library. You can install it using with `conda`:

```bash
conda install -c conda-forge mne
```

More information on how to install `mne` can be found [here](https://mne.tools/stable/install/mne_python.html).

#### 2. pyEEG

Following the installation of `mne`, you must install the custom package `pyeeg`. This package contains utility functions for computing TRFs and is included as a submodule in this repository. To install it, run the following commands from the root directory (making sure you have the correct conda environment activated):

```bash
git submodule update --init
cd pyeeg
pip install .
cd ..
```

#### 3. `audiobook` utility functions

The present code depends on python modules within the `audiobook` folder. This is not packed as a package, so you must add the path to the `audiobook` folder to your `PYTHONPATH`. The easiest is to run code directly within a directory containing `audiobook` folder. Alternatively, you can add it to the path by running the following command from the root directory:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/audiobook
```

Or from within python (either in a script or in a Jupyter notebook) you can add the path to the `audiobook` folder as follows:

```python
import sys
sys.path.append('path/to/feature-PAC/audiobook')
```

### Run the code

Example code to reproduce the results of the paper is provided in the `notebooks` folder. You can run the notebooks directly from the command line or from a Jupyter notebook server (e.g., Jupyter Lab `jupyter lab ./` from the root directory, then navigate to the `notebook` folder within Jupyter).

## Additional information

### Requirements

The code was tested with Python 3.12.2 and 3.9.13 on CentoOS 7 (kernel Linux-5.4.230-1.el7.elrepo.x86_64-x86_64-with-glibc2.17) and Windows 10 respectively with the following dependencies (mostly scientific python libraries):

- mne               1.6.1
- numpy             1.26.4
- scipy             1.13.0
- matplotlib        3.8.4
- sklearn           1.4.2
- pandas            2.2.2
- joblib            1.4.0
- seaborn           0.13.2
- statsmodels       0.14.1
- h5py              >=3.6.0
- h5io              >=0.1.0
- fooof             >=1.0.0

### Dataset

#### Preprocessed data

A minimal working example of the preprocessed data is provided as a figshare dataset. The data is stored in a MNE `Raw` object, which can be loaded using the `mne.io.read_raw_fif` function. The data is stored in a file named ` sub037-audioBook-filtered-ICAed-raw.fif`. The file contains the MEG data, the channel names, the sampling frequency. Events are stored in `audioBook-eve.fif`, and a covariance matrix is stored in `story-cov.fif`. These files allow for a single subject analysis.

#### Raw data

The raw files of the full dataset used in this study are published and available in the [Donders repository](https://data.ru.nl/collections/di/dccn/DSC_3027007.01_206):

> Martin, A.E. (2023): Constructing sentence-level meaning: an MEG study of naturalistic language comprehension. Version 1. Radboud University. (dataset).
[https://doi.org/10.34973/a65x-p009](https://doi.org/10.34973/a65x-p009)

## Citation 📚

Paper:

> The Structure and Statistics of Language jointly shape Cross-frequency Neural Dynamics during Spoken Language Comprehension;
> Hugo Weissbart, Andrea E. Martin;
> Nature Comms. (2024); [![DOI](https://img.shields.io/badge/DOI-10.1038/s41467--024--53128--1-blue.svg)](https://doi.org/10.1038/s41467-024-53128-1)

Code:
> Weissbart, H. (2024). Code and example for TRF-PAC (v0.1.0). Zenodo. [![DOI](https://zenodo.org/badge/708032317.svg)](https://zenodo.org/doi/10.5281/zenodo.12667683)
