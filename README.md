# ABP_Flocking


[![Python Version](https://img.shields.io/badge/Python-3.6%2B-FFD43B.svg)](https://www.python.org/)



## Authorship

#### Authors:
- Nathan O. Silvano (maintainer)
- Emilio Hernandez-Garcia
- Cristobal Lopez

## Project Structure

- `Vegetation/Codes/_NonLocal_FC.py`: Contains the implementation of the vegetation model.
- `Logistic/Codes/LogisticV1.py`: Contains the implementation of the logistic model.
- `Swift_Hohenberg/Codes/Run_Flow_Integration.py`: Contains the implementation of the Swift-Hohenberg model.

## Requirements

The following packages are necessary to run the code:

- `numpy`: For numerical operations.
- `scipy`: For scientific computations, mainly FFT.
- `sympy`: For symbolic mathematics.
- `matplotlib`: For plotting results.
- `h5py`: For handling HDF5 files.
- `tqdm`: For displaying progress bars.
- `numba`: For JIT compilation to speed up the code. (optional)

**It is also required to have a `Python` version `3.6+`.** 