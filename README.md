# Proliferating Active Flocking
This project contains the scripts used to generate and visualize the results presented in the fourthcoming paper:

### **_Continuous description of active proliferating matter._**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<!---
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16080825.svg)](https://doi.org/10.5281/zenodo.16080825)

[![ArXiv](https://img.shields.io/badge/ArXiv-2409.04268-b31b1b)](https://arxiv.org/abs/2409.04268)
[![Communication Physics](https://img.shields.io/badge/Communication%20Physics-Nature-8b0000)](https://www.nature.com/articles/s42005-025-02246-3)
-->

It includes a numerical integration of partial differential equations (PDEs) using a pseudospectral approach with the Runge-Kutta 4th order (RK4) method.
It also includes a particle-based simulation of the same system using an agent-based model (ABM) approach.
--------------------

## Softwares used 

[![Python Version](https://img.shields.io/badge/Python-3.6%2B-FFD43B?logo=python&logoColor=black)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/JupyterNotebook-7.5+-F37726?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Wolfram Mathematica](https://img.shields.io/badge/WolframMathematica-11.0+-FF0000?logo=wolfram&logoColor=white)](https://www.wolfram.com/mathematica/)

### Python Modules+Version Used

[![UTF-8](https://img.shields.io/badge/UTF--8-100000?logo=unicode&logoColor=white)](https://www.unicode.org/standard/standard.html)   
 

[![NumPy](https://img.shields.io/badge/NumPy-2.4.4-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.17.1-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.8-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)


[![Tqdm](https://img.shields.io/badge/Tqdm-4.67.3-01C0F6?logo=tqdm&logoColor=white)](https://tqdm.github.io/)
[![h5py](https://img.shields.io/badge/h5py-3.16.0-000000?logo=hdf5&logoColor=white)](https://www.h5py.org/)

[![cv2](https://img.shields.io/badge/cv2-4.9.0-000000?logo=opencv&logoColor=white)](https://opencv.org/)
[![psutil](https://img.shields.io/badge/psutil-6.0.0-000000?logo=python&logoColor=white)](https://pypi.org/project/psutil/)


----------------
## Authorship

#### Authors:
- Nathan O. Silvano (maintainer)
- Emilio Hernandez-Garcia
- Cristobal Lopez

## Project Structure


------------------------------------
## Usage

The `Code` folder contains the main scripts necessary to perform the simulations. 

### `pseudospec-1d.py`: For Quasi-1D simulations.
- Contains 4 key exeutions functions:
 
- `tophat_kernel_1d`: Generates a tophat kernel in 1D necessary for FFT computations.

- `rk4_pseudospectral_1d`: Implements the 4th order Runge-Kutta method for time integration of the pseudospectral method in 1D.

- `update_step_1d`: Execute the time step of the coupled system in 1D.

- `INITIAL_CONDITIONS:` Generates initial distributions for the density and polarization.
              It can be set to either 'random' or 'other'.
If set to 'other', the user can specify the `pe0` and `mu0` initial values for peclet and net growt rate.
This configuration use a already generated initial condition stored in an Data folder.


For a given choice of `pe0` and `mu0`, the script runs the simulation for a given number of steps.
It will generate the data used in the heatmaps of the paper.

### `pseudospec-2d.py`: For 2D simulations.
  - Contains 4 key exeutions functions:
 
- `tophat_kernel`: Generates a tophat kernel in 2D necessary for FFT computations.

- `rk4_pseudospectral`: Implements the 4th order Runge-Kutta method for time integration of the pseudospectral method in 2D.

- `update_step`: Execute the time step of the coupled system in 2D.

- `INITIAL_CONDITIONS:` Generates initial distributions for the density and polarization.
              It can be set to either 'random' or 'other'.
If set to 'other', the user can specify the `pe0` and `mu0` initial values for peclet and net growt rate.
This configuration use a already generated initial condition stored in an Data folder.

For a given choice of `pe0` and `mu0`, the script runs the simulation for a given number of steps.
It will generate the data used in the heatmaps of the paper.


###  `ABB_MD.py`: For MD particles simulations.

 - Requires two inputs: Particles Diffusion `diff` and Particle velocity `v`.
 - The script runs the particle simulation fora given number of steps.
 - As is conceived, it will create and save a image of the particles distributions and orientations at the end of every cycle.



###  `Run_Animation.py` : Is a auxiliary script that generates a video from the images creates by the other scripts.


--------------------------

## Acknowledgments
    N.S. and E.H-G. acknowledge 
	funding by the Spanish Ministerio de Ciencia, 
	Innovación y Universidades
	(MICIU/AEI/10.13039/501100011033) through
	the Maria de Maeztu project CEX2021-001164-M.
	C.L. and E.H-G. acknowledge grant LAMARCA PID2021-123352OBC32
	funded by MCIN/AEI/10.13039/501100011033 and FEDER, UE.

---------------------------
## License

This project is licensed under the MIT License.

Copyright (c) 2026 Nathan O. Silvano

## Citation
 If you use this code, please cite as:
 
```
 @misc{silvano,
  author       = {Silvano, Nathan 
                  Hernández-García, Emilio and
                  Lopez, Cristobal},
  title        = {},
  month        = ,
  year         = ,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {},
  url          = {},
}
```