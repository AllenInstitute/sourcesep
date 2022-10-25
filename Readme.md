### Description

Source separation for multiplexed spectral fiber photometry neuromodulator imaging.

### Environment

```
conda create -n sourcesep python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install scipy scikit-learn statsmodels jupyterlab pandas seaborn h5py pytables
pip install timebudget rich tqdm autopep8 
pip install dysts sdeint 
pip install librosa 
pip install -e .
```

### Config
Config.toml contents:

```
['all']
root='/data/'
spectra='/data/spectra/'
pilot='/data/pilot/'
```

Data directory structure. (Download data from this [dropbox link](https://www.dropbox.com/sh/tsro86ixhv10ccj/AAD2gsnkY85B7wzYjnB8Vd0Ha?dl=0))

```
data
  ├── pilot
  │   ├── GCaMP8s_1.csv
  │   ├── GCaMP8s_2.csv
  │   ├── GCaMP8s_3.csv
  │   └── test.hdf5
  ├── sim_config.toml
  └── spectra
      ├── EGFP.csv
      ├── HbAbs.csv
      ├── Venus.csv
      └── mApple.csv
```


### Notes

[Deep Learning approaches for source separation in music/speech](https://www.youtube.com/watch?v=AB-F2JmI9U4) that we might adapt for our problem:
 - [Asteroid](https://asteroid-team.github.io/): Augmentations, model components, model implementations
 - [Sigsep](https://sigsep.github.io/): Tutorials, datasets for audio source separation

Particular models:
 - [Open unmix](https://github.com/sigsep/open-unmix-pytorch)
 - [U-net SVS](https://github.com/ws-choi/ISMIR2020_U_Nets_SVS)
 - [Demucs](https://github.com/facebookresearch/demucs)

Miscellaneous:
 
 - `.qmd` files are [quarto](https://quarto.org/) markdown files
 - [FFT for non uniformly sampled data](https://github.com/flatironinstitute/finufft)


### Contributors:
Rohan Gala, Smrithi Sunil, Kaspar Podgorski, Uygar Sümbül