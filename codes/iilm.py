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
#------------------------------------------------------------------------------
#   setup and intial action                                                
#------------------------------------------------------------------------------

setup(nin, z, tmax, seed)
xconf(n, x, nin, z, f, a)
stot, ttot, vtot = act(n, x, f, a)
shc = sshort(z, nin, tcore, score, tmax)
stot += shc

#------------------------------------------------------------------------------
#   loop over configs                                                      
#------------------------------------------------------------------------------

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
    #--------------------------------------------------------------------------
    #   generate new configuration: loop over instantons                       
    #--------------------------------------------------------------------------
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
        #----------------------------------------------------------------------
        #   calculate new action
        #----------------------------------------------------------------------
        xconf(n,x,nin,z)
        snew, tnew, vnew = act(n,x, f, a)
        shc = sshort(z, nin, tcore, score, tmax)
        snew += shc
        #----------------------------------------------------------------------
        #   accept with probability exp(-delta S)                                  
        #----------------------------------------------------------------------
        dels = snew-sold  
        if np.exp(-dels) > random.random() :
            nacc += 1
            stot = snew
        else:
            restore(nin,z,zstore)
        if i < 400 :
            file23.write(" ".join([f"{z[ipr-1]:7.4f}" for ipr in range(1, 11)]) + "\n")
    #--------------------------------------------------------------------------
    #   new configuration: instanton distribution                              
    #--------------------------------------------------------------------------
    for ii in range(1, nin, 2):
        if ii == 1:
            zm = z[nin-1] - tmax
        else:
            zm = z[ii-2]
        z0 = z[ii-1]
        zp = z[ii]
        zia = min(zp - z0, z0 - zm)
        histogramarray(zia, 0.0, stzhist, nzhist, iz)
    #--------------------------------------------------------------------------
    #   action etc.                                                            
    #--------------------------------------------------------------------------
    stot = snew
    ttot = tnew
    vtot = vnew
        
    file18.write(fs.f555.format(i,stot,ttot,vtot,stot/(nin*s0)))
    if i % kp == 0:
        '''
        print("configuration   ", i, "\n",
              "acceptance rate ", float(nacc)/float(nhit), "\n",
              "action (t,v)    ", stot, ttot, vtot, \n
              "s/(n*s_0)       ", stot,ttot,vtot, \n)
        '''
        file17.write('configuration')
        file17.write(i)
        file17.write('\n')
        for k in range(n):
            file17.write(fs.f222.format(k*a,x[k]))
        
    #--------------------------------------------------------------------------
    #   include in sample                                                      
    #--------------------------------------------------------------------------
    stot_sum  += stot
    stot2_sum += stot**2
    vtot_sum  += vtot
    vtot2_sum += vtot**2
    ttot_sum  += ttot
    ttot2_sum += ttot**2
    for k in range(n):
        histogramarray(x[k],xhist_min,stxhist,nxhist,ix)
        x_sum  += x[k]
        x2_sum += x[k]**2
        x4_sum += x[k]**4
        x8_sum += x[k]**8
    #--------------------------------------------------------------------------
    #   correlation function                                                   
    #--------------------------------------------------------------------------
    for ic in range(nc):
        ncor = ncor + 1 
        ip0  = int( (n-np)*random.random() ) 
        x0   = x[ip0]
        for ip in range(n_p):
            x1    = x(ip0+ip)
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

stot_av,stot_err = disp(nconf,stot_sum,stot2_sum)
vtot_av,vtot_err = disp(nconf,vtot_sum,vtot2_sum)
ttot_av,ttot_err = disp(nconf,ttot_sum,ttot2_sum)
x_av,x_err       = disp(nconf*n,x_sum,x2_sum)
x2_av,x2_err     = disp(nconf*n,x2_sum,x4_sum)
x4_av,x4_err     = disp(nconf*n,x4_sum,x8_sum)
for ip in range(n_p): 
    xcor_av[ip],xcor_er[ip]   = disp(ncor,xcor_sum[ip],xcor2_sum[ip])
    x2cor_av[ip],x2cor_er[ip] = disp(ncor,x2cor_sum[ip],x2cor2_sum[ip])  
    x3cor_av[ip],x3cor_er[ip] = disp(ncor,x3cor_sum[ip],x3cor2_sum[ip])     
v_av  = vtot_av/tmax
v_err = vtot_err/tmax
t_av  = ttot_av/tmax
t_err = ttot_err/tmax
e_av  = v_av+t_av
e_err = np.sqrt(v_err**2+t_err**2)

#------------------------------------------------------------------------------
#   output                                                                 
#------------------------------------------------------------------------------
file16.write('\n')
file16.write(fs.f901.format(stot_av,stot_err)) 
file16.write(fs.f902.format(stot_av/float(nin),stot_err/float(nin))) 
file16.write(fs.f903.format(s0)) 
file16.write(fs.f904.format(stot_av/(nin*s0),stot_err/(nin*s0)))  
file16.write(fs.f905.format(v_av,v_err)) 
file16.write(fs.f906.format(t_av,t_err)) 
file16.write(fs.f908.format(e_av,e_err)) 
file16.write(fs.f905.format(x_av,x_err)) 
file16.write(fs.f910.format(x2_av,x2_err)) 
file16.write(fs.f911.format(x4_av,x4_err)) 
file16.write('\n') 
      
#------------------------------------------------------------------------------
#   correlation function, log derivative                                   
#------------------------------------------------------------------------------
file16.write('# x correlation function\n')
file20.write('# tau       x(tau)       dx(tau)     dlog\n')
for ip in range(n_p-1):
    dx   = (xcor_av[ip]-xcor_av[ip+1])/xcor_av[ip]/a
    dxe2 = (xcor_er[ip+1]/xcor_av[ip])**2
    +(xcor_er[ip]*xcor_av[ip+1]/xcor_av[ip]**2)**2
    dxe = np.sqrt(dxe2)/a
    file16.write(fs.f555.format(ip*a,xcor_av[ip],xcor_er[ip],dx,dxe)) 
    file20.write(fs.f555.format(ip*a,xcor_av[ip],xcor_er[ip],dx,dxe))        

#------------------------------------------------------------------------------
#   subtracted x^2 correlation function, log derivative                    
#------------------------------------------------------------------------------
xx_sub = x2cor_av(np)
xx_er  = x2cor_er(np)
for ip in range(n_p):
    x2sub_av[ip] = x2cor_av[ip]-xx_sub
    x2sub_er[ip] = np.sqrt(x2cor_er[ip]**2+xx_er**2)
file16.write('# x correlation function\n')
file21.write('# tau       x(tau)       dx(tau)     dlog\n')
for ip in range(n_p-1):
    dx  = (x2sub_av[ip]-x2sub_av[ip+1])/x2sub_av[ip]/a
    dxe2= (x2sub_er[ip+1]/x2sub_av[ip])**2
    +(x2sub_er[ip]*x2sub_av[ip+1]/x2sub_av[ip]**2)**2
    dxe = np.sqrt(dxe2)/a
    file16.write(fs.f555.format(ip*a,x2cor_av[ip],x2cor_er[ip],dx,dxe)) 
    file21.write(fs.f555.format(ip*a,x2cor_av[ip],x2cor_er[ip],dx,dxe)) 

#------------------------------------------------------------------------------
#   x^3 correlation function, log derivative                               
#------------------------------------------------------------------------------
file16.write('# x correlation function\n')
file22.write('# tau       x(tau)       dx(tau)     dlog\n')
for ip in range(n_p-1):      
    dx   = (x3cor_av[ip]-x3cor_av[ip+1])/x3cor_av[ip]/a
    dxe2 = (x3cor_er[ip+1]/x3cor_av[ip])**2
    +(x3cor_er[ip]*x3cor_av[ip+1]/x3cor_av[ip]**2)**2
    dxe = np.sqrt(dxe2)/a
    file16.write(fs.f555.format(ip*a,x3cor_av[ip],x3cor_er[ip],dx,dxe)) 
    file22.write(fs.f555.format(ip*a,x3cor_av[ip],x3cor_er[ip],dx,dxe))  

#------------------------------------------------------------------------------
#   histograms                                                             
#------------------------------------------------------------------------------
        























