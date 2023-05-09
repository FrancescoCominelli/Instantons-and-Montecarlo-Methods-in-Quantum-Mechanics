import format_strings as fs
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
#------------------------------------------------------------------------------
#     interacting instanton calculation in quantum mechanics.                
                     
#------------------------------------------------------------------------------
#     action m/2(\dot x)^2+k(x^2-f^2)^2, units 2m=k=1.                       
#------------------------------------------------------------------------------
#     program follows units and conventions of txt file                    
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
#   tcore   range of hard interaction (tcore=0.3)
#   acore   strenght of hard core interaction (acore=3.0)
#   dz      average position update (dz=1)
#------------------------------------------------------------------------------
#   Output:
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#     sum ansatz path                                                  
#------------------------------------------------------------------------------
def xsum(nin, z, f, t):
    neven = nin - nin % 2
    xsum = -f
    for i in range(1, neven, 2):
        xsum += f * np.tanh(2.0 * f * (t - z[i])) - f * np.tanh(2.0 * f * (t - z[i+1]))
    if nin % 2 != 0:
        xsum += f * np.tanh(2.0 * f * (t - z[nin])) + f   
    return xsum
#------------------------------------------------------------------------------
#   sumansatz configuration on grid x(n)                                
#------------------------------------------------------------------------------
def xconf(n,x,nin,z,f,a):
    for j in range(1,n):  
         xx = a*j
         x[j] = xsum(nin,z,f,xx)        
    x[0] = x[n-1]
    x    = np.append(x, x[1])
    return

#------------------------------------------------------------------------------
#     discretized action for configuration x(n)                           
#------------------------------------------------------------------------------
def act(n,x,f,a):
    stot = 0.0
    ttot = 0.0
    vtot = 0.0  
    for j in range(n):
        xp = (x[j+1]-x[j])/a
        t  = 1.0/4.0*xp**2
        v  = (x[j]**2-f**2)**2
        s  = a*(t+v)
        ttot += a*t
        vtot += a*v
        stot += s  
    return stot,ttot,vtot

#-------------------------------------------------------------------------c
#     hard core                                                           c
#-------------------------------------------------------------------------c
def sshort(z,nin,tcore,score,tmax):
    shc = 0.0
    tcore2 = tcore**2
    if tcore == 0 and tcore2 == 0:
        return shc
    for i in range(1, nin+1):
        if i == 1:
            zm = z[nin-1] - tmax
        else:
            zm = z[i-2]
        dz = z[i-1] - zm
        shc = shc + score * np.exp(-dz/tcore)
    return shc

#------------------------------------------------------------------------------
#     sort array ra(n)                                                   
#------------------------------------------------------------------------------      
def sort(n, ra):
    l = n//2 + 1
    ir = n
    while True:
        if l > 1:
            l = l - 1
            rra = ra[l-1]
        else:
            rra = ra[ir-1]
            ra[ir-1] = ra[0]
            ir = ir - 1
            if ir == 1:
                ra[0] = rra
                return
        i = l
        j = l + l
        while j <= ir:
            if j < ir and ra[j-1] < ra[j]:
                j = j + 1
            if rra < ra[j-1]:
                ra[i-1] = ra[j-1]
                i = j
                j = j + j
            else:
                j = ir + 1
        ra[i-1] = rra
        
#------------------------------------------------------------------------------
#   initialize instanton configuration                               
#------------------------------------------------------------------------------
def setup(nin,z,tmax, seed):
    random.seed(seed)
    for i in range(nin):
        z[i] = random.random()*tmax
        sort(nin,z)
    return    

#------------------------------------------------------------------------c
#   save array z in zstore                                             c
#------------------------------------------------------------------------c
def store(nin,z,zstore):
    for i in range(nin):
         zstore[i] = z[i]
    return 

#------------------------------------------------------------------------c
#   restore array z from zstore                                        c
#------------------------------------------------------------------------c
def restore(nin,z,zstore):
    for i in range(nin):
         zstore[i] = z[i]
    return

#------------------------------------------------------------------------------
#   open the files
#------------------------------------------------------------------------------
#file6 = open('I dont know the name')
file16 = open('Data/iilm.dat',       'w')
file17 = open('Data/config.dat',     'w')
file18 = open('Data/trajectory.dat', 'w')
file19 = open('Data/qmdist.dat',     'w')
file20 = open('Data/icor.dat',       'w')
file21 = open('Data/icor2.dat',      'w')
file22 = open('Data/icor3.dat',      'w')
file23 = open('Data/iconf.dat',      'w')
file30 = open('Data/zdist.dat',      'w')
file31 = open('Data/sia.dat',        'w')
#------------------------------------------------------------------------------
#   inizialize the values
#------------------------------------------------------------------------------
pi  = np.pi
#grid size n<1000 (n=100)
n = 100
#grid spacing a (dtau=0.05)
a = 0.05
#file6.write('number of equilibration sweeps') 
neq = 100
#file6.write('position update dz') 
dz  = 1.0
#separation of wells f (f=1.4)
f = 1.4
#grid size n<1000 (n=800)
n = 100
#grid spacing a (dtau=0.05)
a = 0.05
#file6.write('number of instantons (even)') 
#file6.write('semiclassical result',nexp) 
#file6.write('two loop result     ',nexp2) 
nin = 2
#file6.write('number of configurations') 
nmc = 1000
#file6.write('number of points in correlator') 
n_p = 20
#file6.write('number of measurements per config') 
nc = 5
#file6.write('write every kth config') 
kp = 5
#file6.write('hard core radius rcore (tcore=rcore/f) (0.3)')
rcore = 0.3              
#file6.write('hard core strength A (score=A*s0) (3.0)')
acore = 3.0
tcore = rcore/f
tmax = n*a
s0   = 4.0/3.0*f**3
score= acore*s0
dens = 8*np.sqrt(2.0/pi)*f**2.5*np.exp(-s0)
dens2= 8*np.sqrt(2.0/pi)*f**2.5*np.exp(-s0-71.0/72.0/s0)
xnin = dens*tmax 
xnin2= dens2*tmax
nexp = int(xnin+0.5)
nexp2= int(xnin2+0.5)
pi  = np.pi
x     = np.zeros(n)
z     = np.zeros(n)
zcore = np.zeros(n)    
xcor_av    =  np.zeros(n_p)
xcor_er    =  np.zeros(n_p)
x2cor_av   =  np.zeros(n_p)
x2cor_er   =  np.zeros(n_p)
x3cor_av   =  np.zeros(n_p)
x3cor_er   =  np.zeros(n_p)
x2sub_av   =  np.zeros(n_p)
x2sub_er   =  np.zeros(n_p)       
xcor_sum   = np.zeros(n_p)
xcor2_sum  = np.zeros(n_p)
x2cor_sum  = np.zeros(n_p)
x2cor2_sum = np.zeros(n_p)
x3cor_sum  = np.zeros(n_p)
x3cor2_sum = np.zeros(n_p)     

#------------------------------------------------------------------------------
#   echo input parameters                                                  
#------------------------------------------------------------------------------
file16.write('qm iilm 1.0')   
file16.write('-----------')   
file16.write(fs.f101.format(f,n,a)) 
file16.write(fs.f1102.format(nin,nmc,neq)) 
file16.write(fs.f103.format(n_p,nc))
file16.write(fs.f1104.format(dz,tcore,score))
file16.write('\n') 
#------------------------------------------------------------------------------
#     plot S_IA                                                              
#------------------------------------------------------------------------------
ni = n // 4
for na in range(ni, ni*2+1):
    z = [ni*a, na*a]
    xconf(n, x, nin, z, f, a)
    stot, ttot, vtot = act(n, x, f, a)
    shc = sshort(z, nin, tcore, score, tmax)
    stot += shc
    file31.write(fs.f222.format((na-ni)*a, stot/s0-2.0))
    
#------------------------------------------------------------------------------
#     parameters for histograms                                              
#------------------------------------------------------------------------------

nxhist = 50
xhist_min = -1.5*f
stxhist= 3.0*f/float(nxhist)
nzhist = 40
stzhist= 4.01/float(nzhist)
ix= np.zeros(nxhist)
iz= np.zeros(nzhist)
#------------------------------------------------------------------------------
#   Read input values from console
#------------------------------------------------------------------------------

#while True:
#   try:
#       seed = int(input("Enter the random seed: ")) #change to int() if expecting int-point input
#       break # Break out of the loop if input is numeric
#   except ValueError:
#       print("Invalid input. Please enter a number.")
seed = -1234
random.seed(seed)

#----------------------------------------------------------------------------c
#   clear summation arrays                                                 c
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
nconf     = 0
ncor      = 0
nacc      = 0
nhit      = 0  
#----------------------------------------------------------------------------c
#   setup and intial action                                                c
#----------------------------------------------------------------------------c

setup(nin, z, tmax, seed)
xconf(n, x, nin, z, f, a)
stot, ttot, vtot = act(n, x, f, a)
shc = sshort(z, nin, tcore, score, tmax)
stot += shc

#----------------------------------------------------------------------------c
#   loop over configs                                                      c
#----------------------------------------------------------------------------c

for i in range(nmc):
    nconf = nconf+1
    if i == neq :
        ncor       = 0
        nconf      = 0
        stot_sum   = 0.0
        stot2_sum  = 0.0
        vtot_sum   = 0.0
        vtot2_sum  = 0.0
        ttot_sum   = 0.0
        ttot2_sum  = 0.0
        tvir_sum   = 0.0
        tvir2_sum  = 0.0
        x_sum      = 0.0
        x2_sum     = 0.0
        x4_sum     = 0.0
        x8_sum     = 0.0
        xcor_sum   = np.zeros(n_p)
        xcor2_sum  = np.zeros(n_p)
        x2cor_sum  = np.zeros(n_p)
        x2cor2_sum = np.zeros(n_p)
        x3cor_sum  = np.zeros(n_p)
        x3cor2_sum = np.zeros(n_p)
        ix         = np.zeros(nxhist)
        iz         = np.zeros(nzhist)
    #----------------------------------------------------------------------------c
    #   generate new configuration: loop over instantons                       c
    #----------------------------------------------------------------------------c
    zstore = np.zeros(nin)
    for iin in range(nin):
        nhit  += 1
        sold  = stot 
        store(nin,z,zstore)
        zold  = z[iin]
        znew  = zold + (random.random()-0.5)*dz
        if znew > tmax:
            znew=znew-tmax
        if znew < 0.0:
                znew=znew+tmax
        z[iin]= znew
        sort(nin,z)
        #----------------------------------------------------------------------------c
        #   calculate new action                                                   c
        #----------------------------------------------------------------------------c
        xconf(n,x,nin,z)
        snew, tnew, vnew = act(n,x, f, a)
        shc = sshort(z, nin, tcore, score, tmax)
        snew += shc
        #----------------------------------------------------------------------------c
        #   accept with probability exp(-delta S)                                  c
        #----------------------------------------------------------------------------c
        dels = snew-sold 
        p  = random.random()
        if np.exp(-dels) > p :
            nacc += 1
            stot = snew
        else:
            restore(nin,z,zstore)
        if i < 400 :
            file23.write(" ".join([f"{z[ipr-1]:7.4f}" for ipr in range(1, 11)]) + "\n")
    #----------------------------------------------------------------------------c
    #   new configuration: instanton distribution                              c
    #----------------------------------------------------------------------------c
    for ii in range(1, nin, 2):
        if ii == 1:
            zm = z[nin-1] - tmax
        else:
            zm = z[ii-2]
        z0 = z[ii-1]
        zp = z[ii]
        zia = min(zp - z0, z0 - zm)
        lens(zia, 0.0, stzhist, nzhist, iz)























