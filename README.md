
## Table of Contents 
 1. Program's structure
 2. 8 codes
 3. Usage
## 1. Program's structure
This program is written in python and is made up by:
  - 8 separate codes, every one with a specific issue needed to print the results of the calculations on some txt files;
 - a file to insert all the parameters needed to make the 8 codes work called: `parameters.py`;
 - a code in wich are present all the function which are needed to make the 8 codes run, called: `functions.py`;
 - a code to plot the results contained in the txt files, called: `plotter.py`;
- a code which specifies the format to print values on the txt files, called: `format_strings.py`.
The 8 codes which produce various txt files are the following:
 1. `qmdiag.py` ;
 2. `qm.py`;
 3. `qmswich.py`;
 4. `qmcool.py`;
 5. `qmidens.py`;
 6. `rilm.py`;
 7. `rilm_gauss.py`;
 8. `iilm.py`; 
## 2. 8 codes
At the beginning of each codes are specified all the parameters needed to make the program work. The general parameters are set by the txt file `parameters.txt`, while the specific parameters needed for the program are specified in the code.
At the beginning of each code are written as comment all the variables entering in the codes and all the output files.
In order to make the codes write the txt output files we need to put thethe 8 codes and all the other codes or file(`parameters.txt`, `functions.py`, `plotter.py`, `format_strings.py`)  in the same folder and create a new folder called `Data`. In the folder `Data` we have to create 8 folders corresponding to the 8 codes with the following names:
1. `qmdiag` ;
 2. `qm`;
 3. `qmswich`;
 4. `qmcool`;
 5. `qmidens`;
 6. `rilm`;
 7. `rilm_gauss`;
 8. `iilm`;
 
We now give a brief description of each of the 8 codes:
###  1. `qmdiag.py`
This programs computes the spectrum and the eigenfunctions of the anharmonic oscillator. The results are used in order to compute euclidean correlation functions.
###  2. `qm.py`
This programs computes correlation functions of the anharmonic oscillator using Monte-Carlo simulations on a euclidean lattice.
###  3. `qmswich.py`
The program qmswicth.for computes the free energy F = −T log(Z) of the anharmonic oscillator using the method of adiabatic switching between the harmonic and the anharmonic oscillator. The action is  S<sub>&alpha;</sub> = S0 + &alpha;(S − S0). The code switches from &alpha; = 0 to &alpha; = 1 and then back to &alpha; = 0. Hysteresis effects are used in order to estimate errors from incomplete equilibration.
The output file contains many details of the adiabatic switching procedure. The final result for the free energy is given as F = F<sub>0</sub> + &delta; F, where F<sub>0</sub> is the free energy of the harmonic oscillator and &delta;F is the integral over &alpha;. We estimate the uncertainty in the final result as F ± &Delta;F(stat) ±&Delta;F(equ) ±&Delta;F(disc), where &delta;F(stat) is the statistical error, &Delta;F(equ) is due to incomplete equilibration (hysteresis), and &Delta;F(disc) is due to discretizing the &alpha; integral.
###  4. `qmcool.py`
This programs is identical to qm.for except that expectation values are measured both in the original and in cooled configurations.
###  5. `qmidens.py`
The program qmidens.for calculates non-Gaussian corrections to the instanton density using adiabatic switching between the Gaussian action and the full action. The calculation is performed in both the zero and one-instanton-sector. The details of the adiabatic switching procedure are very similar to the method used in `qmswitch.py`. Note that the total length of the euclidean time domain, &beta;=na, cannot be chosen too large in order to suppress transitions between the one-instanton sector and the three, five, etc. instanton sector.
The output file contains many details of the adiabatic switching procedure. The final result for the instanton density is compared to the Gaussian (one-loop) approximation. Note that the method breaks down if f is too small or &beta;  is too large.
###  6. `rilm.py`
This program computes correlation functions of the anharmonic oscillator using a random ensemble of instantons. The multi-instanton configuration is constructed using the sum ansatz. Note that, in contrast to RILM calculations in QCD, the fields and correlation functions are computed on a lattice.
###  7. `rilm_gauss.py`
This program generates the same random instanton ensemble as rilm.for but it also includes Gaussian fluctuations around the classical path. This is done by performing a few heating sweeps in the Gaussian effective potential. Most input parameters are defined as in `rilm.for`. 
###  8. `iilm.py`
This program computes correlation functions of the anharmonic oscillator using an interacting ensemble of instantons. The multi-instanton configuration is constructed using the sum ansatz. The configuration is discretized on a lattice and the total action is computed
using the discretized lattice action. Very close instanton-anti-instanton pairs are excluded by adding an nearest neighbor interaction with a repulsive core.
## 3. Usage
After having set the general parameters in the file `parameters.txt`  (we recomend to set them as indicated in the comment at the beginning of each code ), we run all the 8 codes in order to print the output txt files in each of the correspondent folders of `Data`, and finally we run the `plotter.py` code in order to obtain all the plots, that if we want we can save.
