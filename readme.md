### Description

Pre-processing and source separation algorithms for neuromodulator signals in hyperspectral fiber photometry experiments.

### Environment

```
conda create -n sourcesep python=3.12
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install lightning tensorboard
pip install -e .[dev]
```

### Config
Config.toml contents:

```
['all']
root='/data/'
spectra='/data/spectra/'
pilot='/data/pilot/'
```

Data directory structure. (Download data from this [dropbox link](https://www.dropbox.com/sh/k3650wj14sixmvu/AADKdH3ctglrWlNwygwNGLFMa?dl=0))

```
data
  ├── pilot
  │   ├── GCaMP8s_1.csv
  │   ├── GCaMP8s_2.csv
  │   ├── GCaMP8s_3.csv
  │   └── test.hdf5
  ├── sims
  │   ├── 2023-02-24.h5
  │   └── 2023-02-24.toml
  ├── sim_config.toml
  ├── calibrate_px_to_nm.tif
  └── spectra
      ├── EGFP.csv
      ├── HbAbs.csv
      ├── Venus.csv
      ├── mApple.csv
      └── pathlength.csv
```

### Documentation

`./qdocs` contains source files for the quarto project rendered in `./docs`, which can be viewed as a [github pages website](https://alleninstitute.github.io/sourcesep).

### Contributors:
Rohan Gala, Smrithi Sunil, Kaspar Podgorski, Uygar Sümbül
