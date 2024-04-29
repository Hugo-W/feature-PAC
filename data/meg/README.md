# MEG data

## Raw data

Raw data are accessible from the public dataset in the [Donders repository](https://data.ru.nl/collections/di/dccn/DSC_3027007.01_206).

## Sample subject

An example of preprocessed data on a single subject (as of 29th of April 2024) is supplied in the figshare repository [here (private link)](https://figshare.com/s/cb73d94eba15bed8b16e?file=42575938).

The data is stored in a MNE `Raw` object, which can be loaded using the `mne.io.read_raw_fif` function.
Content of the figshare upload:
- MEG data `sub037-audioBook-filtered-ICAed-raw.fif` sampeld at 200Hz
- Story onsets (events) stored in `audioBook-eve.fif`
- a covariance matrix in `story-cov.fif` (e.g. to compute beamformer filters)
