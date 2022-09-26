### Description

Source separation for multiplexed spectral fiber photometry neuromodulator imaging.

### Environment
```
conda create -n sourcesep python=3.8
conda install scikit-learn statsmodels jupyterlab pandas seaborn scipy rich tqdm autopep8 h5py pytables
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install timebudget librosa
pip install -e .
```

### Config
Config.toml contents:
```
['pilot']
data_dir='/SpectralUnmixing'
```

```
SpectralUnmixing
    ├── GCaMP8s_1.csv
    ├── GCaMP8s_2.csv
    ├── GCaMP8s_3.csv
    ├── HbAbs.csv
    └── IndicatorSpectra.csv 
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