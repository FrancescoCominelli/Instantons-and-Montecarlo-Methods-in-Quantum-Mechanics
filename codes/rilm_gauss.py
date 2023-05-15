import format_strings as fs
import numpy as np
import random
from tqdm import tqdm
import functions as fn
import re

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#   Random instanton calculation in quantum mechanics.
#   In this version we add gaussian fluctuations to the classical                     
#   This is done by performing a few monte carlo (heating) sweeps
#   in the gaussian effective potential.
#------------------------------------------------------------------------------
#   action m/2(\dot x)^2+k(x^2-f^2)^2, units 2m=k=1.                       
#------------------------------------------------------------------------------
#   Program follows units and conventions of txt file
#------------------------------------------------------------------------------
#   Imput:
#------------------------------------------------------------------------------
#   f       minimum of harmonic oxillator: (x^2-f^2)^2
#   n       number of lattice points in the euclidean time direction (n=800)
#   a       lattice spacing (a=0.05)
#   N_I+A   number of instantons(has to be even). The program displays the one 
#           and two-loop results for the parameters
#   nmc     number of Monte Carlo sweeps (nmc=10^5)
#   n_p      number of points on which the correlation functions are measured: 
#           <x_i x_(i+1)>,...,<x_i x_(i+np)> (np=20)
#   nmea    number of measurement of the correlation function given MonteCarlo 
#           configuration x_i(nmea=5)
#   npri    number of MonteCarlo configurations between output of averaes to 
#           output file (npri=100)
#   nc      number of correlator measurements in a single configuration                                
#   kp      number of sweeps between writeout of complete configuration     
#------------------------------------------------------------------------------
#   Output:
#------------------------------------------------------------------------------
#   Stot        average total action per configuration
#   Vav, Tav    average potential and kinetic energy
#   <x^n>       expectation value <x^n> (n=1,2,3,4)
#   Pi(tau)     euclidean correlation function Pi(tau)=<O(0)O(tau)>,
#               for O=x,x^2,x^3;results are given in the format: tau, Pi(tau),
#               DeltaPi(tau), dlog(Pi)/dtau, Delta[dlog(Pi)/dtau],where 
#               DeltaPi(tau) is the statistical error in Pi(tau)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#   open files
#------------------------------------------------------------------------------
file16 = open('Data/rilm_gauss/rilm_gauss.dat',  'w')
file17 = open('Data/rilm_gauss/config_gauss.dat','w')
file18 = open('Data/rilm_gauss/trajectory.dat',  'w')
file19 = open('Data/rilm_gauss/qmdist.dat',      'w')
file20 = open('Data/rilm_gauss/rcor_gauss.dat',  'w')
file21 = open('Data/rilm_gauss/rcor2_gauss.dat', 'w')
file22 = open('Data/rilm_gauss/rcor3_gauss.dat', 'w') 
file30 = open('Data/rilm_gauss/zdist.dat',       'w')

#------------------------------------------------------------------------------
#   input parameters                                                                 
#------------------------------------------------------------------------------
# open the file for reading
with open('parameters.txt', 'r') as file:
    # read the contents of the file
    contents = file.read()
   
# search for the values of f and a using regular expressions
f      = re.search(r'f\s*=\s*(\d+\.\d+)', contents).group(1)
n      = re.search(r'n\s*=\s*(\d+)', contents).group(1)
a      = re.search(r'a\s*=\s*(\d+\.\d+)', contents).group(1)
icold  = re.search(r'icold\s*=\s*(\d+)', contents).group(1)
nmc    = re.search(r'nmc\s*=\s*(\d+)', contents).group(1)
delx   = re.search(r'delx\s*=\s*(\d+\.\d+)', contents).group(1)
n_p    = re.search(r'n_p\s*=\s*(\d+)', contents).group(1)
kp     = re.search(r'kp\s*=\s*(\d+)', contents).group(1)


# convert the values to integers
f      = float(f)   #separation of wells f (f=1.4)
n      = int(n)     #grid size n<10000 (n=100)
a      = float(a)   #grid spacing a (dtau=0.05)
icold  = int(icold) #cold/hot start (0,1)
nmc    = int(nmc)   #monte carlo sweeps
delx   = float(delx)#update x (delx)
n_p    = int(n_p)   #number of points in correlator
kp     = int(kp)    #number of sweeps between cooling

nin = 10
nc = 5
nheat= 5
pi  = np.pi

tmax  = n*a
s0    = 4.0/3.0*f**3
dens  = 8*np.sqrt(2.0/pi)*f**2.5*np.exp(-s0)
dens2 = 8*np.sqrt(2.0/pi)*f**2.5*np.exp(-s0-71.0/72.0/s0)
xnin  = dens*tmax 
xnin2 = dens2*tmax
nexp  = int(xnin+0.5)
nexp2 = int(xnin2+0.5)

#------------------------------------------------------------------------------
#     parameters for histograms                                              
#------------------------------------------------------------------------------  
nxhist = 50
xhist_min = -1.5*f
stxhist= 3.0*f/float(nxhist)
nzhist = 40
stzhist= 4.01/float(nzhist)

#------------------------------------------------------------------------------
#     initialize                                                  
#------------------------------------------------------------------------------
x      =  np.zeros(n)
z      =  np.zeros(n)  
x_hot  =  np.zeros(n)
w      =  np.zeros(n)
ix     = np.zeros(nxhist)
iz     = np.zeros(nzhist)
xcor_sum   =  np.zeros(n_p)
xcor2_sum  =  np.zeros(n_p)
xcor_av    =  np.zeros(n_p)
xcor_er    =  np.zeros(n_p)
x2cor_sum  =  np.zeros(n_p)
x2cor2_sum =  np.zeros(n_p)
x2cor_av   =  np.zeros(n_p)
x2cor_er   =  np.zeros(n_p)
x3cor_sum  =  np.zeros(n_p)
x3cor2_sum =  np.zeros(n_p)
x3cor_av   =  np.zeros(n_p)
x3cor_er   =  np.zeros(n_p)
x2sub_av   =  np.zeros(n_p)
x2sub_er   =  np.zeros(n_p)

#------------------------------------------------------------------------------
#     echo input parameters                                                  
#------------------------------------------------------------------------------
file16.write('qm rilm gauss 1.0')   
file16.write('-----------------')   
file16.write(fs.f101.format(f,n,a)) 
file16.write(fs.f102.format(nin,nmc)) 
file16.write(fs.f103.format(n_p,nc))
file16.write(fs.f107.format(delx,nheat)) 
file16.write('\n')
file17.write(fs.f444.format(n, nmc/kp, n*a, f)) 

#------------------------------------------------------------------------------
#     clear summation arrays                                                 
#------------------------------------------------------------------------------
stot_sum  = 0.0
stot2_sum = 0.0
vtot_sum  = 0.0
vtot2_sum = 0.0
ttot_sum  = 0.0
ttot2_sum = 0.0
tvir_sum  = 0.0
tvir2_sum = 0.0
x_sum     = 0.0
x2_sum    = 0.0
x4_sum    = 0.0
x8_sum    = 0.0

#------------------------------------------------------------------------------
#   Read input values from console
#------------------------------------------------------------------------------
while True:
   try:
       seed = int(input("Enter the random seed: ")) #change to int() if expecting int-point input
       break # Break out of the loop if input is numeric
   except ValueError:
       print("Invalid input. Please enter a number.")

#------------------------------------------------------------------------------
#   initialize
#------------------------------------------------------------------------------
nconf= 0
ncor = 0
nhit = 0 
nacc = 0

#------------------------------------------------------------------------------
#   Read input values from console
#------------------------------------------------------------------------------




