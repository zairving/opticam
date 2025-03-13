# opticam_new
A Python package for reducing OPTICAM data.

## Features
- Customisable. Many of `opticam_new`'s reduction methods are fully customisable, allowing for full control over the reduction process.
- Informative. `opticam_new` includes informative logging, allowing for reproducable reduction and information about any errors or warnings.
- Robust. `opticam_new` is designed to catch many common errors and inform the user how they can be resolved.
- Scalable. `opticam_new` can leverage modern multi-core CPUs to drastically speed up reduction.
- Simple. When using `opticam_new`'s reduction methods, the default values should "just work" most of the time. Faint sources and/or crowded fields may require some tailoring, however.

## Installation

### From GitHub (recommended)

You can install the latest stable release of `opticam_new` directly from GitHub using:

```
pip install git+https://github.com/zairving/opticam_new.git
```

### Locally

If you have a local copy of `opticam_new`, it can be `pip` installed by navigating to the directory and running:

```
pip install .
```

### Requirements

All of `opticam_new`'s dependencies are available as Python packages via your preferred package manager, and should be handled automatically via `pip`. However, in the unlikely event that the `pip` installation breaks any dependencies, a full copy of the environment I use for development is available as a [dependencies.yml](dependencies.yml) file.

## Getting Started

Documentation for `opticam_new` is available on [Read the Docs](https://opticam-new.readthedocs.io/en/latest/index.html). To get started, I recommend checking out the [basic usage](https://opticam-new.readthedocs.io/en/latest/tutorials/basic_usage.html) tutorial.