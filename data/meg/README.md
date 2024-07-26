# MEG data

## Raw data

Raw data are accessible from the public dataset in the [Donders repository](https://data.ru.nl/collections/di/dccn/DSC_3027007.01_206).

## Preprocessed data

Preprocessed data for 13 of the 25 subjects (due to quota limitations) are available in the [Figshare repository](https://doi.org/10.6084/m9.figshare.24236512), as a `.zip` archive.

## Sample subject

An example of preprocessed data on a single subject is supplied in the [figshare repository](https://doi.org/10.6084/m9.figshare.24236512).

The data is stored in a MNE `Raw` object, which can be loaded using the `mne.io.read_raw_fif` function.
Content of the figshare upload:
- MEG data `sub037-audioBook-filtered-ICAed-raw.fif` sampeld at 200Hz
- Story onsets (events) stored in `audioBook-eve.fif`
- a covariance matrix in `story-cov.fif` (e.g. to compute beamformer filters)

## Source data

Note that the source data to reproduce all figures of the article are also present in the [Figshare repository](https://doi.org/10.6084/m9.figshare.24236512), archived in `Source Data.zip`.
