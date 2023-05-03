import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sympy as sym
from math import factorial
from pickle import APPEND
import random
import format_strings as fs
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#   lattice calculation in quantum mechanics                              
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#   action m/2(\dot x)^2+k(x^2-f^2)^2, units 2m=k=1.                      
#------------------------------------------------------------------------------
#   periodic b.c. x(0)=x(n-1)                               
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#   Imput:
#------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------
#   Output:
#------------------------------------------------------------------------------
#   Stot        average total action per configuration
#   Vav, Tav    average potential and kinetic energy
#   <x^n>       expectation value <x^n> (n=1,2,3,4)
#   Pi(tau)     euclidean correlation function Pi(tau)=<O(0)O(tau)>, for O=x,x^2,x^3;
#               results are given in the format: tau, Pi(tau), DeltaPi(tau), dlog(Pi)/dtau,
#               Delta[dlog(Pi)/dtau], where DeltaPi(tau) is the statistical error in Pi(tau)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#     function to include value a in histogram array hist(n)
#     def histogramarray(a, amin, st, m, hist)                
#------------------------------------------------------------------------------
#     a       value to be added to histogram array                        
#     amin    minimum value in histogram                                  
#     st      bin width                                                   
#     m       number of bins                                              
#     hist(n) histogram array                                             
#------------------------------------------------------------------------------
def histogramarray(a, amin, st, m, hist):
    j = (a - amin)/st + 1.000001
    if (j < 1):
        j = 1
    if (j > m):
        j = m
    hist[int(j)-1] += 1
    return
#------------------------------------------------------------------------------
#   Estimate average and error from xtot and x2tot
#------------------------------------------------------------------------------
#   Input:
#           n: number of measurements
#           xtot: sum of x_i
#           x2tot: sum of x**2
#   Output:
#           xav: average
#           xerr: error estimate
#------------------------------------------------------------------------------  
def disp(n, xtot, x2tot):
    if n < 1:
        raise ValueError("Number of measurements must be at least 1")
    xav = xtot / float(n)
    del2 = x2tot / float(n*n) - xav*xav / float(n)
    if del2 < 0:
        del2 = 0      
    xerr = np.sqrt(del2)  
    return xav, xerr

#------------------------------------------------------------------------------
#   set the values
#------------------------------------------------------------------------------
f   = 1.4
n   = 800
icold  = 0
a   = 0.05
nmc = 1000
neq = 100
delx= 0.5
nc  = 5
n_p  = 20
tmax = n*a
kp = 5

#------------------------------------------------------------------------------
# echo input parameters
#------------------------------------------------------------------------------
#open txt files

file16 = open('Data/qm.dat', 'w')
file17 = open('Data/config.dat', 'w')
file18 = open('Data/trajectory.dat', 'w')
file19 = open('Data/qmdist.dat', 'w')
file20 = open('Data/qmcor.dat', 'w')
file21 = open('Data/qmcor2.dat', 'w')
file22 = open('Data/qmcor3.dat', 'w')

#write on a txt file values

file16.write('lattice qm 1.0\n')
file16.write('----------\n')
file16.write(fs.f101.format(f,n,a))
file16.write(fs.f102.format(nmc,neq))
file16.write(fs.f103.format(n_p,nc))
file16.write(fs.f104.format(delx,icold))
file17.write(fs.f444.format(n, nmc/kp, n*a, f))

#------------------------------------------------------------------------------
#   set the histogram parameters
#------------------------------------------------------------------------------

nxhist    = 50
xhist_min = -1.5*f
stxhist   = 3.0*f/nxhist

#------------------------------------------------------------------------------
#     initialize                                                             
#------------------------------------------------------------------------------
random.seed(-1234)
xcor_er   = np.zeros(n_p)
x2cor_er  = np.zeros(n_p)
x3cor_er  = np.zeros(n_p)
xcor_sum  = np.zeros(neq)
x2cor_sum = np.zeros(neq)
x3cor_sum = np.zeros(neq)
xcor2_sum = np.zeros(neq)
x2cor2_sum= np.zeros(neq)
x3cor2_sum= np.zeros(neq)
histo_x   = np.zeros(neq)
x2sub_av  = np.zeros(n_p)
x2sub_er  = np.zeros(n_p)
stot_sum = 0.0
stot2_sum= 0.0
vtot_sum = 0.0
vtot2_sum= 0.0
ttot_sum = 0.0
ttot2_sum= 0.0
tvir_sum = 0.0
tvir2_sum= 0.0
x_sum = 0
x2_sum = 0
x4_sum = 0
x8_sum = 0
x = np.zeros(n)
if icold==0:
    for i in range(n):
        x[i]= -f
else:
    for i in range(n):
        x[i] = 2.0 * random.random() * f - f
        
#------------------------------------------------------------------------------
#     periodic boundary conditions                                           
#------------------------------------------------------------------------------

x[0] = x[n-1]
x    = np.append(x, x[1])

#------------------------------------------------------------------------------
#     initial action                                                       
#------------------------------------------------------------------------------

stot= 0.0         
for i in range(n):
    xp = (x[i+1]-x[i])/a
    t  = 1.0/4.0*xp**2
    v  = (x[i]**2-f**2)**2
    s  = a*(t+v)
    stot += s
    
#------------------------------------------------------------------------------
#    monte carlo sweeps                                                     
#------------------------------------------------------------------------------
nacc  = 0
nhit  = 0
nconf = 0
ncor  = 0
histo_x = np.zeros(neq)

for i in range(nmc):
    nconf += 1
    if i == neq:
        nconf = 0
        ncor  = 0
        stot_sum = 0.0
        stot2_sum= 0.0
        vtot_sum = 0.0
        vtot2_sum= 0.0
        ttot_sum = 0.0
        ttot2_sum= 0.0
        tvir_sum = 0.0
        tvir2_sum= 0.0
        x_sum = 0
        x2_sum = 0
        x4_sum = 0
        x8_sum = 0
        xcor_sum  = np.zeros(neq)
        x2cor_sum = np.zeros(neq)
        x3cor_sum = np.zeros(neq)
        histo_x   = np.zeros(neq)
    #--------------------------------------------------------------------------
    #   one sweep thorough configuration                                       
    #--------------------------------------------------------------------------
    for j in range(1,n):
        nhit += 1
        xpm   = (x[j]-x[j-1])/a
        xpp   = (x[j+1]-x[j])/a
        t     = 1.0/4.0*(xpm**2+xpp**2)
        v     = (x[j]**2-f**2)**2
        sold  = a*(t+v)
        xnew  = x[j] + delx*(2.0*random.random()-1.0)
        xpm   = (xnew-x[j-1])/a
        xpp   = (x[j+1]-xnew)/a
        t     = 1.0/4.0*(xpm**2+xpp**2)
        v     = (xnew**2-f**2)**2
        snew  = a*(t+v)
        dels  = snew-sold
        dels  = min(dels,70.0)
        dels  = max(dels,-70.0)
        if np.exp(-dels) > random.random():
            x[j]  = xnew
            nacc += 1
    x[n-1]= x[0]
    x     = np.append(x, x[1])
		
	#--------------------------------------------------------------------------
    #   calculate action and other things                                                  
    #--------------------------------------------------------------------------
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
    #write on a txt file
    file18.write(fs.f444.format(i,stot,ttot,vtot))

    #--------------------------------------------------------------------------
    #     populate histogram include in sample                                                     
    #--------------------------------------------------------------------------
    stot_sum += stot
    stot2_sum+= stot**2
    vtot_sum += vtot
    vtot2_sum+= vtot**2
    ttot_sum += ttot
    ttot2_sum+= ttot**2
    tvir_sum += tvtot
    tvir2_sum+= tvtot**2
    
    for k in range(n):
        histogramarray(x[k], xhist_min, stxhist, nxhist, histo_x)
        x_sum += x[k]
        x2_sum += x[k]**2
        x4_sum += x[k]**4
        x8_sum += x[k]**8
    #--------------------------------------------------------------------------
    #     correlation function                                                   
    #--------------------------------------------------------------------------
    for ic in range(nc):
        ncor += 1 
        ip0  = int((n-n_p)*random.random()) 
        x0   = x[ip0] 
        for ip in range(n_p):
            x1 = x[ip0+ip]
            xcor  = x0*x1
            x2cor = xcor**2
            x3cor = xcor**3   
            xcor_sum[ip]   += xcor
            xcor2_sum[ip]  += xcor**2
            x2cor_sum[ip]  += x2cor
            x2cor2_sum[ip] += x2cor**2
            x3cor_sum[ip]  += x3cor
            x3cor2_sum[ip] += x3cor**2            	
#------------------------------------------------------------------------------
#   averages                                                               
#------------------------------------------------------------------------------
xcor_av = np.zeros(n_p)
x2cor_av= np.zeros(n_p)
x3cor_av= np.zeros(n_p)
stot_av,stot_err = disp(nconf,stot_sum,stot2_sum)
vtot_av,vtot_err = disp(nconf,vtot_sum,vtot2_sum)
ttot_av,ttot_err = disp(nconf,ttot_sum,ttot2_sum)
tvir_av,tvir_err = disp(nconf,tvir_sum,tvir2_sum)
x_av,x_err       = disp(nconf*n,x_sum,x2_sum)
x2_av,x2_err     = disp(nconf*n,x2_sum,x4_sum)
x4_av,x4_err     = disp(nconf*n,x4_sum,x8_sum)
for ip in range(n_p):
    xcor_av[ip],xcor_er[ip]   = disp(ncor,xcor_sum[ip],xcor2_sum[ip])
    x2cor_av[ip],x2cor_er[ip] = disp(ncor,x2cor_sum[ip],x2cor2_sum[ip],)
    x3cor_av[ip],x3cor_er[ip] = disp(ncor,x3cor_sum[ip],x3cor2_sum[ip],)
v_av  = vtot_av/tmax
v_err = vtot_err/tmax
t_av  = ttot_av/tmax
t_err = ttot_err/tmax
tv_av = tvir_av/tmax
tv_err= tvir_err/tmax
e_av  = v_av + tv_av
e_err = np.sqrt(v_err**2 + tv_err**2)

#------------------------------------------------------------------------------
#   output                                                               
#------------------------------------------------------------------------------
file16.write('\n')
file16.write(fs.f801.format(stot_av, stot_err))
file16.write(fs.f802.format(v_av, v_err))
file16.write(fs.f803.format(t_av, t_err))
file16.write(fs.f804.format(tv_av, tv_err))
file16.write(fs.f805.format(e_av, e_err))
file16.write(fs.f806.format(x_av, x_err))
file16.write(fs.f807.format(x2_av, x2_err))
file16.write(fs.f808.format(x4_av, x4_err))
file16.write('\n')

#------------------------------------------------------------------------------
#   correlation function, log derivative
#------------------------------------------------------------------------------

file16.write("x correlation function\n")
file20.write("tau       x(tau)       dx(tau)     dlog\n")

for ip in range(n_p-1):
    dx  = (xcor_av[ip]-xcor_av[ip+1])/xcor_av[ip]/a
    dxe2 = (xcor_er[ip+1]/xcor_av[ip])**2
    + (xcor_er[ip]*xcor_av[ip+1]/xcor_av[ip]**2)**2
    dxe  = np.sqrt(dxe2)/a
    file16.write(fs.f555.format(ip*a, xcor_av[ip], xcor_er[ip], dx, dxe))
    file20.write(fs.f555.format(ip*a, xcor_av[ip], xcor_er[ip], dx, dxe))

#------------------------------------------------------------------------------
#   subtracted x^2 correlation function, log derivative                                                              
#------------------------------------------------------------------------------

xx_sub = x2cor_av[n_p-1]
xx_er  = x2cor_er[n_p-1]
for ip in range(n_p):
    x2sub_av[ip] = x2cor_av[ip]-xx_sub
    x2sub_er[ip] = np.sqrt(x2cor_er[ip]**2+xx_er**2)
    
file16.write("x2 correlation function\n")
file21.write("tau       x2(tau)       dx2(tau)     dlog\n")

for ip in range(n_p-1):
    dx  = (x2sub_av[ip]-x2sub_av[ip+1])/x2sub_av[ip]/a
    dxe2 = (x2sub_er[ip+1]/x2sub_av[ip])**2
    + (x2sub_er[ip]*x2sub_av[ip+1]/x2sub_av[ip]**2)**2
    dxe  = np.sqrt(dxe2)/a
    file16.write(fs.f555.format(ip*a, x2cor_av[ip], x2cor_er[ip], dx, dxe))
    file21.write(fs.f555.format(ip*a, x2cor_av[ip], x2cor_er[ip], dx, dxe))


#------------------------------------------------------------------------------
#   x^3 correlation function, log derivative                                                              
#------------------------------------------------------------------------------

file16.write("x3 correlation function\n")
file22.write("tau       x3(tau)       dx3(tau)     dlog\n")

for ip in range(n_p-1):
    dx  = (x3cor_av[ip]-x3cor_av[ip+1])/x3cor_av[ip]/a
    dxe2 = (x3cor_er[ip+1]/x3cor_av[ip])**2
    + (x3cor_er[ip]*x3cor_av[ip+1]/x3cor_av[ip]**2)**2
    dxe  = np.sqrt(dxe2)/a
    file16.write(fs.f555.format(ip*a, x3cor_av[ip], x3cor_er[ip], dx, dxe))
    file22.write(fs.f555.format(ip*a, x3cor_av[ip], x3cor_er[ip], dx, dxe))

#------------------------------------------------------------------------------
#   wave function                                                              
#------------------------------------------------------------------------------

file16.write('\nx distribution\n')

X = stxhist + np.arange(len(histo_x))
Y = histo_x

# plot
fig, ax = plt.subplots()

ax.step(X, Y, linewidth=2.5)

ax.set(xlim=(-20, 20))#, xticks=np.arange(1, 8),
#       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()




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

file16.close()
file17.close()
file18.close()
file19.close()
file20.close()
file21.close()
file22.close()