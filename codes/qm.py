import numpy as np
import scipy as sp
import matplotlib as mpl
import sympy as sym
from math import factorial
from pickle import APPEND
import random
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#   lattice calculation in quantum mechanics                              
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#   action m/2(\dot x)^2+k(x^2-f^2)^2, units 2m=k=1.                      
#-------------------------------------------------------------------------------------------------
#   periodic b.c. x(0)=x(n-1)                               
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#   Imput:
#-------------------------------------------------------------------------------------------------
#   f       minimum of harmonic oxillator: (x^2-f^2)^2
#   n       number of lattice points in the euclidean time direction (n=800)
#   a       lattice spacing (a=0.05)
#   ih      ih=0: cold start, x_i=-f; ih=1: hot start, x_i=random
#   neq     number of equlibration sweeps before the first measurement (neq=100)
#   nmc     number of Monte Carlo sweeps (nmc=10^5)
#   dx      width of Gaussian distribution used for MonteCarlo update: x_i^(n)-->x_i^(n+1)
#   np      number of points on which the correlation functions are measured: 
#           <x_i x_(i+1)>,...,<x_i x_(i+np)> (np=20)
#   nmea    number of measurement of the correlation function given MonteCarlo configuration x_i
#           (nmea=5)
#   npri    number of MonteCarlo configurations between output of averaes to output file (npri=100)
#   nc      number of correlator measurements in a single configuration                                
#   kp      number of sweeps between writeout of complete configuration     
#-------------------------------------------------------------------------------------------------
#   Output:
#-------------------------------------------------------------------------------------------------
#   Stot        average total action per configuration
#   Vav, Tav    average potential and kinetic energy
#   <x^n>       expectation value <x^n> (n=1,2,3,4)
#   Pi(tau)     euclidean correlation function Pi(tau)=<O(0)O(tau)>, for O=x,x^2,x^3;
#               results are given in the format: tau, Pi(tau), DeltaPi(tau), dlog(Pi)/dtau,
#               Delta[dlog(Pi)/dtau], where DeltaPi(tau) is the statistical error in Pi(tau)
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#     function to include value a in histogram array hist(n)
#     def histogramarray(a, amin, st, m, hist)                
#-------------------------------------------------------------------------------------------------
#     a       value to be added to histogram array                        
#     amin    minimum value in histogram                                  
#     st      bin width                                                   
#     m       number of bins                                              
#     hist(n) histogram array                                             
#-------------------------------------------------------------------------------------------------
def histogramarray(a, amin, st, m, hist):
    j = (a - amin)/st + 1.000001
    if (j < 1):
        j = 1
    if (j > m):
        j = m
    hist[int(j)-1] += 1
    return
#-------------------------------------------------------------------------------------------------
#   Estimate average and error from xtot and x2tot
#-------------------------------------------------------------------------------------------------
#   Input:
#           n: number of measurements
#           xtot: sum of x_i
#           x2tot: sum of x**2
#   Output:
#           xav: average
#           xerr: error estimate
#-------------------------------------------------------------------------------------------------    
def disp(n, xtot, x2tot):
    if n < 1:
        raise ValueError("Number of measurements must be at least 1")
    xav = xtot / float(n)
    del2 = x2tot / float(n*n) - xav*xav / float(n)
    if del2 < 0:
        del2 = 0      
    xerr = np.sqrt(del2)  
    return xav, xerr
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#   set the values
#-------------------------------------------------------------------------------------------------
f   = 1.4
n   = 800
ih  = 0
a   = 0.05
nmc = 1000
neq = 100
delx= 0.5
nc  = 5
n_p  = 20
tmax = n*a
#-------------------------------------------------------------------------------------------------
#   set the histogram parameters
#-------------------------------------------------------------------------------------------------
nxhist    = 50
xhist_min = -1.5*f
stxhist   = 3.0*f/nxhist

       
#-------------------------------------------------------------------------------------------------
#     initialize                                                             
#-------------------------------------------------------------------------------------------------
random.seed(-1234)
xcor_sum  = np.zeros(neq)
x2cor_sum = np.zeros(neq)
x3cor_sum = np.zeros(neq)
xcor2_sum = np.zeros(neq)
x2cor2_sum= np.zeros(neq)
x3cor2_sum= np.zeros(neq)
histo_x   = np.zeros(neq)
stot_sum = 0.0
stot2_sum= 0.0
vtot_sum = 0.0
vtot2_sum= 0.0
ttot_sum = 0.0
ttot2_sum= 0.0
tvir_sum = 0.0
tvir2_sum= 0.0
x = np.zeros(n)
if ih==0:
    for i in range(n):
        x[i]= -f
else:
    for i in range(n):
        x[i] = 2.0 * random.random() * f - f
#-------------------------------------------------------------------------------------------------
#     periodic boundary conditions                                           
#-------------------------------------------------------------------------------------------------
x[0] = x[n-1]
x    = np.append(x, x[1])
#-------------------------------------------------------------------------------------------------
#     initial action                                                       
#-------------------------------------------------------------------------------------------------
stot= 0.0         
for i in range(n):
    xp = (x[i+1]-x[i])/a
    t  = 1.0/4.0*xp**2
    v  = (x[i]**2-f**2)**2
    s  = a*(t+v)
    stot = stot + s
#-------------------------------------------------------------------------------------------------
#    monte carlo sweeps                                                     
#-------------------------------------------------------------------------------------------------
nacc = 0
nhit = 0
nconf = 0
ncor = 0
histo_x = np.zeros(neq)

for i in range(1, nmc):
    nconf += 1
    if i == neq:
        nconf = 0
        ncor = 0
        xcor_sum = np.zeros(neq)
        x2cor_sum = np.zeros(neq)
        x3cor_sum = np.zeros(neq)
        histo_x = np.zeros(neq)
    #---------------------------------------------------------------------------------------------
    #   one sweep thorough configuration                                       
    #---------------------------------------------------------------------------------------------
    for j in range(1,n):
        nhit= nhit + 1
        xpm = (x[j]-x[j-1])/a
        xpp = (x[j+1]-x[j])/a
        t   = 1.0/4.0*(xpm**2+xpp**2)
        v   = (x[j]**2-f**2)**2
        sold = a*(t+v)
        xnew = x[j] + delx*(2.0*random.random()-1.0)
        xpm = (xnew-x[j-1])/a
        xpp = (x[j+1]-xnew)/a
        t   = 1.0/4.0*(xpm**2+xpp**2)
        v   = (xnew**2-f**2)**2
        snew = a*(t+v)
        dels = snew-sold
        dels = min(dels,70.0)
        dels = max(dels,-70.0)
        if np.exp(-dels) > random.random():
            x[j] = xnew
            nacc = nacc + 1
    x[n-1]= x[0]
    x     = np.append(x, x[1])
		
	#---------------------------------------------------------------------------------------------
    #   calculate action and other things                                                  
    #---------------------------------------------------------------------------------------------
    stot = 0.0
    ttot = 0.0
    tvtot= 0.0
    vtot = 0.0
    for j in range(n):
        xp = (x[j+1]-x[j])/a
        t  = 1.0/4.0*xp**2
        v  = (x[j]**2-f**2)**2
        tv = 2.0*x[j]**2*(x[j]**2-f**2)
        s  = a*(t+v)
        ttot  = ttot +a*t
        vtot  = vtot +a*v
        tvtot = tvtot+a*tv
        stot  = stot + s
    #--------------------------------------------------------------------------------------------
    #     populate histogram                                                     
    #--------------------------------------------------------------------------------------------
    for k in range(n):
        histogramarray(x[k], xhist_min, stxhist, nxhist, histo_x)	
    #---------------------------------------------------------------------------------------------
    #     correlation function                                                   
    #---------------------------------------------------------------------------------------------
    for ic in range(nc):
        ncor = ncor + 1 
        ip0  = int( (n-n_p)*random.random() ) 
        x0   = x[ip0] 
        for ip in range(n_p):
            x1 = x[ip0+ip]
            xcor  = x0*x1
            x2cor = xcor**2
            x3cor = xcor**3   
            xcor_sum[ip]   = xcor_sum[ip]  + xcor
            xcor2_sum[ip]  = xcor2_sum[ip] + xcor**2
            x2cor_sum[ip]  = x2cor_sum[ip] + x2cor
            x2cor2_sum[ip] = x2cor2_sum[ip]+ x2cor**2
            x3cor_sum[ip]  = x3cor_sum[ip] + x3cor
            x3cor2_sum[ip] = x3cor2_sum[ip]+ x3cor**2            	
#-------------------------------------------------------------------------------------------------
#   averages                                                               
#-------------------------------------------------------------------------------------------------
stot_sum = stot_sum + stot
stot2_sum= stot2_sum+ stot**2
vtot_sum = vtot_sum + vtot
vtot2_sum= vtot2_sum+ vtot**2
ttot_sum = ttot_sum + ttot
ttot2_sum= ttot2_sum+ ttot**2
tvir_sum = tvir_sum + tvtot
tvir2_sum= tvir2_sum+ tvtot**2
xcor_av = np.zeros(n_p)
x2cor_av= np.zeros(n_p)
x3cor_av= np.zeros(n_p)
stot_av,stot_err = disp(nconf,stot_sum,stot2_sum)
vtot_av,vtot_err = disp(nconf,vtot_sum,vtot2_sum)
ttot_av,ttot_err = disp(nconf,ttot_sum,ttot2_sum)
tvir_av,tvir_err = disp(nconf,tvir_sum,tvir2_sum)
for p in range(n_p):
    xcor_av[p]  = xcor_sum[p]/ncor
    x2cor_av[p] = x2cor_sum[p]/ncor
    x3cor_av[p] = x3cor_sum[p]/ncor
v_av  = vtot_av/tmax
v_err = vtot_err/tmax
t_av  = ttot_av/tmax
t_err = ttot_err/tmax
tv_av = tvir_av/tmax
tv_err= tvir_err/tmax

# Ground state wave function histogram
#print("---------------------------------------------")
#print("x\t\tP(x)")
#x_norm = 0
#for i in range(n_bins):
#    x_norm += histo_x[i]*delta_bin
#   
#for i in range(n_bins):
#    xx = x_hist_min + i*delta_bin
#    print("{:.4f}\t{:.4f}".format(xx, histo_x[i]/x_norm))
    		
