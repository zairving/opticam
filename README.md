# opticam_new
A Python package for reducing OPTICAM data.

## Feautres
- Scalable. `opticam_new` can leverage modern multi-core CPUs thanks to the `multiprocessing` module. However, the scaling is highly non-linear due to parallelisation being used only in only some parts of the code. `opticam_new` will default to half the number of processors, but this can, of course, be changed at will. Here's a look at the relative performance scaling when using 1, 4, 8, 16, and 32 processors to reduce some test data:

![multiprocess_scaling](https://github.com/zairving/opticam_new/assets/121759971/6d0b259d-5ed1-4120-b89f-0c8f8c347007)
