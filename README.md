## Table of Contents 
 1. Program's structure
 2.  `parameters.txt` 
 3. 8 codes
 4. `plotter.py`
 5. Usage
## 1. Project's structure
This project is written in python and is made up by:
  - a txt file in which are present the python libraries used for the codes compilation called: `requirements.txt`;
  -  a code which creates the folder `Data/`. The folders needed to store the data produced by the 8 codes are created directly into `Data/`.   Such code is called:`make_folder.py`;
 -  a file to insert all the parameters needed to make the 8 codes run called: `parameters.txt`;
 - a code in which are present all the functions needed to make the 8 codes run, called: `functions.py`;
 - a code which specifies the format to print values on the txt files, called: `format_strings.py`;
 - 8 separate codes, every one with a specific issue needed to print the results of the calculations on txt files:
	 1. `qmdiag.py` ;
	 2. `qm.py`;
	 3. `qmswich.py`;
	 4. `qmcool.py`;
	 5. `qmidens.py`;
	 6. `rilm.py`;
	 7. `rilm_gauss.py`;
	 8. `iilm.py`; 

	Each code stores the data in a folder inside `Data/` called with the same code's name.
 E.g. the code `qmdiag.py` will store the data in: `Data/qmdiag/` .
 - a code to plot the results contained in the txt files, called: `plotter.py`.

## 2. `parameters.txt`
In this file are already written the suggested values for a consistent simulation, but you are free to change it in order to reduce the computation time, which in some cases may be more then 10 hours. In order to reduce the computation time is suggested to reduce: 

 - the number of lattice points (n) (for all the codes)
 - the number of Monte Carlo iterations (nmc) (for all the codes)
 - the number of cooling sweeps (ncool) (for `qmcool.py`)
 - the number of steps in adiabatic switching (nalpha) (for `qmswich.py` and `qmidens.py`)

We recomend, as indicated in the commentations of the file, not to modify the values for `kp`, which regulates the writeout of the configurations of the Monte Carlo simulations. If you really have the interess in changing this parameter we remind that you also have to change the code of the plotter in order to plot the configurations.

## 3. 8 codes
At the beginning of each code are written as comment all the variables entering in the codes and all the output files.

###  1. `qmdiag.py`
This programs computes the spectrum and the eigenfunctions of the anharmonic oscillator. The results are used in order to compute euclidean correlation functions.
###  2. `qm.py`
This programs computes correlation functions of the anharmonic oscillator using Monte-Carlo simulations on a euclidean lattice.
###  3. `qmswich.py`
The program `qmswicth.py` computes the free energy F = −T log(Z) of the anharmonic oscillator using the method of adiabatic switching between the harmonic and the anharmonic oscillator. The action is  S<sub>&alpha;</sub> = S<sub>0</sub> + &alpha;(S − S<sub>0</sub>). The code switches from &alpha; = 0 to &alpha; = 1 and then back to &alpha; = 0. Hysteresis effects are used in order to estimate errors from incomplete equilibration.
The output file contains many details of the adiabatic switching procedure. The final result for the free energy is given as F = F<sub>0</sub> + &delta; F, where F<sub>0</sub> is the free energy of the harmonic oscillator and &delta;F is the integral over &alpha;. We estimate the uncertainty in the final result as F ± &Delta;F(stat) ±&Delta;F(equ) ±&Delta;F(disc), where &delta;F(stat) is the statistical error, &Delta;F(equ) is due to incomplete equilibration (hysteresis), and &Delta;F(disc) is due to discretizing the &alpha; integral.
###  4. `qmcool.py`
This programs is identical to `qm.py` except that expectation values are measured both in the original and in cooled configurations.
###  5. `qmidens.py`
The program `qmidens.py` calculates non-Gaussian corrections to the instanton density using adiabatic switching between the Gaussian action and the full action. The calculation is performed in both the zero and one-instanton-sector. The details of the adiabatic switching procedure are very similar to the method used in `qmswitch.py`. Note that the total length of the euclidean time domain, &beta;=na, cannot be chosen too large in order to suppress transitions between the one-instanton sector and the three, five, etc. instanton sector.
The output file contains many details of the adiabatic switching procedure. The final result for the instanton density is compared to the Gaussian (one-loop) approximation. Note that the method breaks down if f is too small or &beta;  is too large.
###  6. `rilm.py`
This program computes correlation functions of the anharmonic oscillator using a random ensemble of instantons. The multi-instanton configuration is constructed using the sum ansatz. Note that, in contrast to RILM calculations in QCD, the fields and correlation functions are computed on a lattice.
###  7. `rilm_gauss.py`
This program generates the same random instanton ensemble as rilm.for but it also includes Gaussian fluctuations around the classical path. This is done by performing a few heating sweeps in the Gaussian effective potential. 
###  8. `iilm.py`
This program computes correlation functions of the anharmonic oscillator using an interacting ensemble of instantons. The multi-instanton configuration is constructed using the sum ansatz. The configuration is discretized on a lattice and the total action is computed
using the discretized lattice action. Very close instanton-anti-instanton pairs are excluded by adding an nearest neighbor interaction with a repulsive core.
## 3. `plotter.py`
This program has to be runned after that all the other programs have finished their computations since it takes the data from the folder `Data/`. 
It is the program deputated to give a graphic representation of the data obtained after the simulations.
## 4. Usage

 1. Download all the pyton's libraries indicated in the file `requirements.txt`;
 2. Download all the files present in `codes/` and place them in the same folder;
 3. Run `make_folder.py`;
 4. Set the parameters for the simulations in `parameters.txt`(the parameters are already sat to obtain consistent simulations but with long computation time, see the section dedicated to `parameters.txt` );
 5. Run all the 8 codes which we have indicated in the previous section;
 6. Run `plotter.py` in order to obtain the graphic representations of the data obtained after the runs of all the simulations (if you, for some reason, have changed the value `kp`, please modify the parts of the code in plotter deputated to write out the configurations coming from the Monte Carlo simulation, otherwise the results will not be plotted).
