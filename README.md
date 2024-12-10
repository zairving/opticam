# opticam_new
A Python package for reducing OPTICAM data.

## Features
- Customisable. Many of `opticam_new`'s reduction methods are fully customisable, allowing for full control over the reduction process.
- Informative. `opticam_new` includes informative logging, allowing for reproducable reduction and information about any errors or warnings.
- Robust. `opticam_new` is designed to catch many common errors and inform the user how they can be resolved.
- Scalable. `opticam_new` can leverage modern multi-core CPUs to drastically speed up reduction.
- Simple. When using `opticam_new`'s reduction methods, the default values should "just work" most of the time. Faint sources and/or crowded fields may require some tailoring, however.

## Getting Started

To get started with `opticam_new`, there is a [dedicated script for creating some synthetic observations](Tutorials/create_test_data.py), which are used in most of the [guided tutorials](Tutorials). These synthetic observations can then be reduced by following the [Basic Usage tutorial](Tutorials/basic_usage.ipynb). 