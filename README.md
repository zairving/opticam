# opticam_new
A Python package for reducing OPTICAM data.

## Features
- Customisable. Many of `opticam_new`'s reduction methods are fully customisable, allowing for full control over the reduction process.
- Informative. `opticam_new` includes informative logging, allowing for reproducable reduction and information about any errors or warnings.
- Robust. `opticam_new` is designed to catch many common errors and inform the user how they can be resolved.
- Scalable. `opticam_new` can leverage modern multi-core CPUs to drastically speed up reduction.
- Simple. When using `opticam_new`'s reduction methods, the default values should "just work" most of the time. Faint sources and/or crowded fields may require some tailoring, however.

## Requirements

All of `opticam_new`'s dependencies are available as Python packages via your preferred package manager. I personally use [miniforge](https://github.com/conda-forge/miniforge), a fast, lightweight [Conda](https://conda.io/) interface that defaults to the [conda-forge](https://conda-forge.org/) channel. A full [dependencies.yml](dependencies.yml) file is available for automatic installation, but the following dependencies should get you up-and-running:

- `astropy >= 6.0.0`
- `ccdproc` (tested with version `2.4.2`)
- `matplotlib` (tested with version `3.9.2`)
- `numpy >= 2.0.0`
- `pandas` (tested with version `2.2.3`)
- `photutils` (tested with version `2.0.2`)
- `scikit-image` (tested with version `0.24.0`)
- `scipy` (tested with version `1.14.1`)
- `tqdm` (tested with version `4.66.6`)

## Getting Started

To get started with `opticam_new`, there is a [dedicated script for creating some synthetic observations](Tutorials/create_test_data.py), which are used in most of the [guided tutorials](Tutorials). These synthetic observations can be reduced by following the [Basic Usage tutorial](Tutorials/basic_usage.ipynb). 