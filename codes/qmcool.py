import format_strings as fs
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
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
#   dx      width of Gaussian distribution used for MonteCarlo update: 
#           x_i^(n)-->x_i^(n+1)
#   np      number of points on which the correlation functions are measured: 
#           <x_i x_(i+1)>,...,<x_i x_(i+np)> (np=20)
#   nmea    number of measurement of the correlation function given MonteCarlo
#           configuration x_i
#           (nmea=5)
#   npri    number of MonteCarlo configurations between output of averaes to
#           output file (npri=100)
#   nc      number of correlator measurements in a single configuration                                
#   nst     number of MonteCarlo configurations between successive cooled
#           configurations. The number of cooled configurations is nconf/nst
#           (nst=20)
#   kp      number of sweeps between cooling                              
#   ncool   number of cooling sweeps in a single configuration (ncool=50)             
#   kp2     number of sweeps between writeout of complete configuration     
#------------------------------------------------------------------------------
#   Output:
#------------------------------------------------------------------------------
#   Pi(tau)     euclidean correlation function Pi(tau)=<O(0)O(tau)>, for O=x,
#               x^2,x^3;
#               results are given in the format: tau, Pi(tau), DeltaPi(tau),
#               dlog(Pi)/dtau,
#               Delta[dlog(Pi)/dtau], where DeltaPi(tau) is the statistical
#               error in Pi(tau)
#   N_I+A       total number of instantons extracted from number zero crossings
#               as a function of the number of cooling sweeps
#   Stot        total action vs number of cooling sweeps
#   S/N         action per instanton. S_0 is the continuum result for one 
#               instanton
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
#   discretized action for configuration x(n)                           
#------------------------------------------------------------------------------
def act(f, a, delx, n, x):
    stot = 0.0
    ttot = 0.0
    vtot = 0.0
    for j in range(1, n+1):
        xp = (x[j+1] - x[j]) / a
        t = 1.0 / 4.0 * xp**2
        v = (x[j]**2 - f**2)**2
        s = a * (t + v)
        ttot = ttot + a * t
        vtot = vtot + a * v
        stot = stot + s
    return stot, ttot, vtot

#------------------------------------------------------------------------c
#     return number and location of (anti) instantons                    c
#------------------------------------------------------------------------c
def inst(f, a, delx, n, x, xi, xa, z):
    ni = 0
    na = 0
    nin= 0
    ix = int(np.sign(1.0, x[0]))
    for i in range(1, n):
        tau = a * i
        ixp = int(np.sign(1.0, x[i]))
        if ixp > ix:
            ni  += 1
            nin += 1
            xi[ni-1] = tau
            z[nin-1] = tau
        elif ixp < ix:
            na  += 1
            nin += 1
            xa[na-1] = tau
            z[nin-1] = tau
        ix = ixp
    return ni, na

#------------------------------------------------------------------------------

file16 = open('Data/qm.dat', 'w')
file17 = open('Data/config.dat', 'w')
file18 = open('Data/trajectory.dat', 'w')
file19 = open('Data/qmdist.dat', 'w')
file20 = open('Data/coolconfig.dat', 'w')
file21 = open('Data/cor.dat', 'w')
file22 = open('Data/coolcor.dat', 'w')
file23 = open('Data/nin.dat', 'w')
file24 = open('Data/scool.dat', 'w')
file25 = open('Data/sinst.dat', 'w')
file26 = open('Data/cor2.dat', 'w')
file27 = open('Data/coolcor2.dat', 'w')
file28 = open('Data/cor3.dat', 'w')
file29 = open('Data/coolcor3.dat', 'w')
file30 = open('Data/zdist.dat', 'w')

#------------------------------------------------------------------------------
#     input                                                                  
#------------------------------------------------------------------------------

#separation of wells f (f=1.4)
f     = 1.4
#grid size n<10000 (n=100)
n     = 100
#grid spacing a (dtau=0.05)
a     = 0.05
#cold/hot start (0,1)
icold = 0
#equilibration sweeps
neq   = 100
#monte carlo sweeps
nmc   = 10000
#update x (delx)
delx  = 0.5
#number of points in correlator
n_p   = 20
#number of measurements per configuration
nc    = 5
#write every kth config
kp2   = 10
#number of sweeps between cooling
kp    = 5
#number of cooling sweeps (ncool<5000)
ncool = 2000           
tmax  = n*a

#------------------------------------------------------------------------------
#   Read input values from console
#------------------------------------------------------------------------------

while True:
    try:
        seed = int(input("Enter the random seed: ")) #change to int() if expecting int-point input
        break # Break out of the loop if input is numeric
    except ValueError:
        print("Invalid input. Please enter a number.")
     
#----------------------------------------------------------------------------
#     echo input parameters                                                  
#----------------------------------------------------------------------------

pi  = np.pi
s0  = 4.0/3.0*f**3
de  = 8*np.sqrt(2.0/pi)*f**2.5*np.exp(-s0)
de2 = de*(1.0-71.0/72.0/s0)
file16.write('lattice qm 1.1')
file16.write('--------------\n')
file16.write(fs.f201(f,n,a))
file16.write(fs.f202(nmc,neq))
file16.write(fs.f203(n_p,nc))
file16.write(fs.f204(delx,icold,ncool))
file16.write(fs.f205(s0,de,de*n*a))
file16.write(fs.f201(s0,de2,de2*n*a))
#file17.write('#',n,nmc/kp,n*a,f)
#file20.write('#',n,nmc/kp,n*a,f)

#----------------------------------------------------------------------------
#     parameters for histograms                                              
#----------------------------------------------------------------------------

nxhist = 50
xhist_min = -1.5 * f
stxhist = -2 * xhist_min / nxhist
nzhist = 40
stzhist = 4.01 / nzhist

#----------------------------------------------------------------------------c
#    clear summation arrays                                                 c
#----------------------------------------------------------------------------c

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

nin_sum    = np.zeros(ncool+1)
nin2_sum   = np.zeros(ncool+1)
scool_sum  = np.zeros(ncool+1)
scool2_sum = np.zeros(ncool+1)
ix         = np.zeros(nxhist)
iz         = np.zeros(nzhist)
xi         = np.zeros(ncool)
xa         = np.zeros(ncool)
z          = np.zeros(ncool)

#----------------------------------------------------------------------------
#     correlators <x(#0)x(t)>
#----------------------------------------------------------------------------

xcor_sum   = np.zeros(ncool)
xcor2_sum  = np.zeros(ncool)
xcor_av    = np.zeros(ncool)
xcor_er    = np.zeros(ncool)
xcool_sum  = np.zeros(ncool)
xcool2_sum = np.zeros(ncool)
xcool_av   = np.zeros(ncool)
xcool_er   = np.zeros(ncool)

#----------------------------------------------------------------------------
#     correlators <x^2(0)x^2(t)>                                             
#----------------------------------------------------------------------------

x2cor_sum     = np.zeros(ncool)
x2cor2_sum    = np.zeros(ncool)
x2cor_av      = np.zeros(ncool)
x2cor_er      = np.zeros(ncool)
x2sub_av      = np.zeros(ncool)
x2sub_er      = np.zeros(ncool)
x2cool_sum    = np.zeros(ncool)
x2cool2_sum   = np.zeros(ncool)
x2cool_av     = np.zeros(ncool)
x2cool_er     = np.zeros(ncool)
x2cool_sub_av = np.zeros(ncool)
x2cool_sub_er = np.zeros(ncool)

#----------------------------------------------------------------------------
#     correlators <x^3(0)x^3(t)>                                             
#----------------------------------------------------------------------------

x3cor_sum   = np.zeros(ncool)
x3cor2_sum  = np.zeros(ncool)
x3cor_av    = np.zeros(ncool)
x3cor_er    = np.zeros(ncool)
x3cool_sum  = np.zeros(ncool)
x3cool2_sum = np.zeros(ncool)
x3cool_av   = np.zeros(ncool)
x3cool_er   = np.zeros(ncool)

#------------------------------------------------------------------------------
#     initialize                                                             
#------------------------------------------------------------------------------
xs = np.zeros(n)
x  = np.zeros(n)
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
nacc      = 0
nhit      = 0    
nconf     = 0
ncor      = 0
ncoolconf = 0
ncoolcor  = 0

for i in tqdm(range(nmc)):
    nconf += 1
    if i == neq:
        nconf     = 0
        ncor      = 0
        ncoolconf = 0
        ncoolcor  = 0
        stot_sum  = 0.0
        stot2_sum = 0.0
        vtot_sum  = 0.0
        vtot2_sum = 0.0
        ttot_sum  = 0.0
        ttot2_sum = 0.0
        tvir_sum  = 0.0
        tvir2_sum = 0.0
        x_sum     = 0
        x2_sum    = 0
        x4_sum    = 0
        x8_sum    = 0
        xcor_sum    = np.zeros(ncool)
        xcor2_sum   = np.zeros(ncool)
        xcool_sum   = np.zeros(ncool)
        xcool2_sum  = np.zeros(ncool)
        x2cor_sum   = np.zeros(ncool)
        x2cor2_sum  = np.zeros(ncool)
        x2cool_sum  = np.zeros(ncool)
        x2cool2_sum = np.zeros(ncool)
        x3cor_sum   = np.zeros(ncool)
        x3cor2_sum  = np.zeros(ncool)
        x3cool_sum  = np.zeros(ncool)
        x3cool2_sum = np.zeros(ncool)
        nin_sum     = np.zeros(ncool+1)
        nin2_sum    = np.zeros(ncool+1)
        ix          = np.zeros(nxhist)
        iz          = np.zeros(nzhist)
        
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
        xs[j]= x[j]
        ttot  = ttot +a*t
        vtot  = vtot +a*v
        tvtot = tvtot+a*tv
        stot  = stot + s
    #write on a txt file
    if i <= 10000:
        file18.write(fs.f444.format(i,stot,ttot,vtot))
    xs[n-1]= xs[0]
    xs     = np.append(xs, xs[1])
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
        histogramarray(x[k], xhist_min, stxhist, nxhist, ix)
        x_sum  += x[k]
        x2_sum += x[k]**2
        x4_sum += x[k]**4
        x8_sum += x[k]**8
    
    #--------------------------------------------------------------------------
    #     correlation function                                                   
    #--------------------------------------------------------------------------
    ipa = np.zeros(nc)
    for ic in range(nc):
        ncor += 1 
        ip0  = int((n-n_p)*random.random())
        ipa[ic]  = ip0
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
    #----------------------------------------------------------------------------c
    #   cooling and topological charge                                         c
    #----------------------------------------------------------------------------c
    if i % kp == 0:
        ncoolconf += 1
        xs = [0.0] * (n + 1)
        ni, na = inst(f, a, delx, xs, n, xi, xa, z)
        ss, ts, vs = act(f, a, delx, n, xs)
        nin = ni + na
        nin_sum[0]   += nin
        nin2_sum[0]  += nin**2
        scool_sum[0] += ss
        scool2_sum[0]+= ss**2
    for icool in range(ncool):
        xs = [0.0] * (n + 1)
        ni, na = inst(f, a, delx, xs, n, xi, xa, z)
        ss, ts, vs = act(f, a, delx, n, xs)
        nin = ni + na
        nin_sum[0]   += nin
        nin2_sum[0]  += nin**2
        scool_sum[0] += ss
        scool2_sum[0]+= ss**2
    #----------------------------------------------------------------------------c
    #     cooled configuration: instanton distribution                           c
    #----------------------------------------------------------------------------c

         do 310 ii=1,nin-1,2
            if(ii .eq. 1) then
               zm = z(nin)-tmax
            else
               zm = z(ii-1)
            endif
            z0 = z(ii)
            zp = z(ii+1)
            zia= min(zp-z0,z0-zm)
            call lens(zia,0.0,stzhist,nzhist,iz)
 310     continue
       













