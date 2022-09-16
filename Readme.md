### Description

Source separation for multiplexed spectral fiber photometry neuromodulator imaging.

### Environment
```
conda create -n sourcesep python=3.8
conda install scikit-learn statsmodels jupyterlab pandas seaborn scipy rich tqdm autopep8 h5py pytables
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install timebudget
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
 - `.qmd` files are [quarto](https://quarto.org/) markdown files
 - [FFT for non uniformly sampled data](https://github.com/flatironinstitute/finufft)
 

### Contributors:
Rohan Gala, Smrithi Sunil, Kaspar Podgorski, Uygar Sümbül