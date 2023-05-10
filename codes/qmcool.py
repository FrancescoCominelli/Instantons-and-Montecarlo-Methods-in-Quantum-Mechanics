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
    for j in range(n):
        xp = (x[j+1] - x[j]) / a
        t = 1.0 / 4.0 * xp**2
        v = (x[j]**2 - f**2)**2
        s = a * (t + v)
        ttot += a * t
        vtot += a * v
        stot += s
    return stot, ttot, vtot

#------------------------------------------------------------------------------
#     return number and location of (anti) instantons                    
#------------------------------------------------------------------------------
def inst(f, a, delx, n, x, xi, xa, z):
    ni = 0
    na = 0
    nin= 0
    ix = int(np.sign(x[0]))
    for i in range(1,n):
        tau = a * i
        ixp = int(np.sign(x[i]))
        if ixp > ix:
            xi[ni] = tau
            z[nin] = tau
            ni  += 1
            nin += 1         
        elif ixp < ix:
            xa[na] = tau
            z[nin] = tau
            na  += 1
            nin += 1            
        ix = ixp
    return ni, na

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

#----------------------------------------------------------------------------c
#   log derivative                                                         c
#----------------------------------------------------------------------------c
def dl(xcor1,xcor2,a):      
    dl = (xcor1-xcor2)/(xcor1*a)
    return dl

#----------------------------------------------------------------------------c
#     log derivative, error                                                  c
#----------------------------------------------------------------------------c
def dle(xcor1,xcor2,xcor1e,xcor2e,a):      
    dle2 = (xcor2e/xcor1)**2+(xcor1e*xcor2/xcor1**2)**2
    dle  = np.sqrt(dle2)
    return dle

#------------------------------------------------------------------------------
#   plot histogram
#       Input:  amin    minimum value in histogram
#               m       number of bins 
#               ist()   histogram array
#------------------------------------------------------------------------------

def plot_histogram2(amin, m , ist):
    bins = np.linspace(amin, -amin, m+1)
    plt.hist(bins[:-1], bins, density=True ,weights=ist, histtype='step')
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.show()
 

#------------------------------------------------------------------------------
#   local cooling algorithm                                                
#------------------------------------------------------------------------------ 
def cool(f,a,delx, seed, xs, n, ncool):     
    random.seed(seed)
    nhit = 10
    delxp= 0.1*delx
    for k in range(ncool+1):
        for i in range(1,n):
            xpm = (xs[i]-xs[i-1])/a
            xpp = (xs[i+1]-xs[i])/a
            t = 1.0/4.0*(xpm**2+xpp**2)
            v = (xs[i]**2-f**2)**2
            sold = a*(t+v)
            for j in range(nhit):          
                xnew = xs[i] + delxp*(2.0*random.random()-1.0)
                xpm = (xnew-xs[i-1])/a
                xpp = (xs[i+1]-xnew)/a
                t = 1.0/4.0*(xpm**2+xpp**2)
                v = (xnew**2-f**2)**2
                snew = a*(t+v)
                if snew < sold :
                    xs[i]=xnew
    return

def plot_hist(xmin, st, n, ist):    
    bins = [xmin + st*i for i in range(n+1)]
    plt.hist(ist, bins=bins)
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()
    return

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
nmc   = 1000
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
ncool = 10          
tmax  = n*a

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
#------------------------------------------------------------------------------
#     echo input parameters                                                  
#------------------------------------------------------------------------------

pi  = np.pi
s0  = 4.0/3.0*f**3
de  = 8*np.sqrt(2.0/pi)*f**2.5*np.exp(-s0)
de2 = de*(1.0-71.0/72.0/s0)
file16.write('lattice qm 1.1')
file16.write('--------------\n')
file16.write(fs.f201.format(f,n,a))
file16.write(fs.f202.format(nmc,neq))
file16.write(fs.f203.format(n_p,nc))
file16.write(fs.f204.format(delx,icold,ncool))
file16.write(fs.f205.format(s0,de,de*n*a))
file16.write(fs.f206.format(s0,de2,de2*n*a))
#file17.write('#',n,nmc/kp,n*a,f)
#file20.write('#',n,nmc/kp,n*a,f)

#------------------------------------------------------------------------------
#     parameters for histograms                                              
#------------------------------------------------------------------------------

nxhist = 50
xhist_min = -1.5 * f
stxhist = -2 * xhist_min / float(nxhist)
nzhist = 40
stzhist = 4.01 / float(nzhist)

#------------------------------------------------------------------------------
#    clear summation arrays                                                 
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

nin_sum    = np.zeros(ncool+1)
nin2_sum   = np.zeros(ncool+1)
scool_sum  = np.zeros(ncool+1)
scool2_sum = np.zeros(ncool+1)
ix         = np.zeros(nxhist)
iz         = np.zeros(nzhist)
xi         = np.zeros(ncool)
xa         = np.zeros(ncool)

#------------------------------------------------------------------------------
#     correlators <x(#0)x(t)>
#------------------------------------------------------------------------------

xcor_sum   = np.zeros(n_p)
xcor2_sum  = np.zeros(n_p)
xcor_av    = np.zeros(n_p)
xcor_er    = np.zeros(n_p)
xcool_sum  = np.zeros(n_p)
xcool2_sum = np.zeros(n_p)
xcool_av   = np.zeros(n_p)
xcool_er   = np.zeros(n_p)

#------------------------------------------------------------------------------
#     correlators <x^2(0)x^2(t)>                                             
#------------------------------------------------------------------------------

x2cor_sum     = np.zeros(n_p)
x2cor2_sum    = np.zeros(n_p)
x2cor_av      = np.zeros(n_p)
x2cor_er      = np.zeros(n_p)
x2sub_av      = np.zeros(n_p)
x2sub_er      = np.zeros(n_p)
x2cool_sum    = np.zeros(n_p)
x2cool2_sum   = np.zeros(n_p)
x2cool_av     = np.zeros(n_p)
x2cool_er     = np.zeros(n_p)
x2cool_sub_av = np.zeros(n_p)
x2cool_sub_er = np.zeros(n_p)

#------------------------------------------------------------------------------
#     correlators <x^3(0)x^3(t)>                                             
#------------------------------------------------------------------------------

x3cor_sum   = np.zeros(n_p)
x3cor2_sum  = np.zeros(n_p)
x3cor_av    = np.zeros(n_p)
x3cor_er    = np.zeros(n_p)
x3cool_sum  = np.zeros(n_p)
x3cool2_sum = np.zeros(n_p)
x3cool_av   = np.zeros(n_p)
x3cool_er   = np.zeros(n_p)

#------------------------------------------------------------------------------
#     initialize                                                             
#------------------------------------------------------------------------------
ipa = np.zeros(nc)
xs = np.zeros(n+1)
x  = np.zeros(n)
z  = np.zeros(n)
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
        xcor_sum    = np.zeros(n_p)
        xcor2_sum   = np.zeros(n_p)
        xcool_sum   = np.zeros(n_p)
        xcool2_sum  = np.zeros(n_p)
        x2cor_sum   = np.zeros(n_p)
        x2cor2_sum  = np.zeros(n_p)
        x2cool_sum  = np.zeros(n_p)
        x2cool2_sum = np.zeros(n_p)
        x3cor_sum   = np.zeros(n_p)
        x3cor2_sum  = np.zeros(n_p)
        x3cool_sum  = np.zeros(n_p)
        x3cool2_sum = np.zeros(n_p)
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
    x[n]  = x[1]
    		
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
    xs[n]  = xs[1]
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
            
    #--------------------------------------------------------------------------
    #   cooling and topological charge                                         
    #--------------------------------------------------------------------------
    
    if i % kp == 0:
        ncoolconf += 1
        ni, na = inst(f, a, delx, n, xs, xi, xa, z)
        ss, ts, vs = act(f, a, delx, n, xs)
        nin = ni + na
        nin_sum[0]   += nin
        nin2_sum[0]  += nin**2
        scool_sum[0] += ss
        scool2_sum[0]+= ss**2
        for icool in range(1,ncool+1):
            cool(f, a, delx, seed, xs, n, ncool)
            ni, na = inst(f, a, delx, n, xs, xi, xa, z)
            ss, ts, vs = act(f, a, delx, n, xs)
            nin = ni + na
            nin_sum[icool]   += nin
            nin2_sum[icool]  += nin**2
            scool_sum[icool] += ss
            scool2_sum[icool]+= ss**2
            
        #----------------------------------------------------------------------
        #     cooled configuration: instanton distribution                            
        #----------------------------------------------------------------------
        
        for ii in range(0, nin, 2):
            if ii == 1:
                zm = z[nin-1] - tmax
            else:
                zm = z[ii-1]
            z0  = z[ii]
            zp  = z[ii+1]
            zia = min(zp-z0, z0-zm)
            histogramarray( zia, 0.0, stzhist, nzhist, iz)
        
        #----------------------------------------------------------------------
        #   cooled correlator                                                      
        #----------------------------------------------------------------------
        
        for ic in range(nc):
            ncoolcor += 1
            ip0 = int(ipa[ic])
            x0 = xs[ip0]
            for ip in range(n_p):
                x1    = xs[ip0+ip]
                xcor  = x0*x1
                x2cor = xcor**2
                x3cor = xcor**3
                xcool_sum[ip]   += xcor
                xcool2_sum[ip]  += xcor**2
                xcool_sum[ip]   += xcor
                xcool2_sum[ip]  += xcor**2
                x2cool_sum[ip]  += x2cor
                x2cool2_sum[ip] += x2cor**2
                x3cool_sum[ip]  += x3cor
                x3cool2_sum[ip] += x3cor**2
    #--------------------------------------------------------------------------
    #     write configuration                                                    
    #--------------------------------------------------------------------------
    '''
    if i % kp == 0:
        print("configuration:   ", i, "\n",
              "coupling:        ", alpha, "\n",
              "acceptance rate: ", float(nacc)/float(nhit), "\n",
              "action (T,V):    ", stot, ttot, vtot)
    '''
    file17.write('# configuration')
    file17.write(str(i))
    file20.write('# configuration')
    file20.write(str(i))
    for k in range(n):
        file17.write(fs.f222.format(k*a, x[k]))
        file20.write(fs.f222.format(k*a, x[k]))
#------------------------------------------------------------------------------
#   averages
#------------------------------------------------------------------------------
stot_av, stot_err = disp(nconf  , stot_sum, stot2_sum)
vtot_av, vtot_err = disp(nconf  , vtot_sum, vtot2_sum)
ttot_av, ttot_err = disp(nconf  , ttot_sum, ttot2_sum)
tvir_av, tvir_err = disp(nconf  , tvir_sum, tvir2_sum)
x_av   ,  x_err   = disp(nconf*n, x_sum   , x2_sum)
x2_av  ,  x2_err  = disp(nconf*n, x2_sum  , x4_sum)
x4_av  , x4_err   = disp(nconf*n, x4_sum  , x8_sum)
#------------------------------------------------------------------------------
#     correlators                                                            
#------------------------------------------------------------------------------

for ip in range(n_p):
    xcor_av[ip]  , xcor_er[ip]   = disp(ncor    , xcor_sum[ip]  , xcor2_sum[ip]) 
    x2cor_av[ip] , x2cor_er[ip]  = disp(ncor    , x2cor_sum[ip] , x2cor2_sum[ip]) 
    x3cor_av[ip] , x3cor_er[ip]  = disp(ncor    , x3cor_sum[ip] , x3cor2_sum[ip]) 
    xcool_av[ip] , xcool_er[ip]  = disp(ncoolcor, xcool_sum[ip] , xcool2_sum[ip]) 
    x2cool_av[ip], x2cool_er[ip] = disp(ncoolcor, x2cool_sum[ip], x2cool2_sum[ip]) 
    x3cool_av[ip], x3cool_er[ip] = disp(ncoolcor, x3cool_sum[ip], x3cool2_sum[ip]) 

#------------------------------------------------------------------------------
#   instanton density, cooled action                                       
#------------------------------------------------------------------------------

nin_av   = np.zeros(ncool+1)
nin_er   = np.zeros(ncool+1)
scool_av = np.zeros(ncool+1)
scool_er = np.zeros(ncool+1) 
for ic in range(ncool + 1):
    nin_av[ic], nin_er[ic] = disp(ncoolconf, nin_sum[ic]  , nin2_sum[ic]) 
    scool_av[ic], scool_er[ic] = disp(ncoolconf, scool_sum[ic], scool2_sum[ic]) 
v_av   = vtot_av/tmax
v_err  = vtot_err/tmax
t_av   = ttot_av/tmax
t_err  = ttot_err/tmax
tv_av  = tvir_av/tmax
tv_err = tvir_err/tmax
e_av   = v_av+tv_av
e_err  = np.sqrt(v_err**2+tv_err**2)

#------------------------------------------------------------------------------
#   output                                                                 
#------------------------------------------------------------------------------

file16.write('\n')
file16.write('nconf = ')
file16.write(str(nconf))
file16.write('\nncoolc= ')
file16.write(str(ncoolconf))     
file16.write(fs.f801.format(stot_av,stot_err)) 
file16.write(fs.f802.format(v_av,v_err)) 
file16.write(fs.f803.format(t_av,t_err)) 
file16.write(fs.f804.format(tv_av,tv_err)) 
file16.write(fs.f805.format(e_av,e_err)) 
file16.write(fs.f806.format(x_av,x_err)) 
file16.write(fs.f807.format(x2_av,x2_err)) 
file16.write(fs.f808.format(x4_av,x4_err)) 
file16.write('\n')

#------------------------------------------------------------------------------
#   correlators etc                                                        
#------------------------------------------------------------------------------
      
file16.write('\n')
file16.write(' <x(0)x(t)> correlation function\n') 
file21.write('#<x(0)x(t)> correlation function\n')
for ip in range(1,n_p):
    dx  = dl(xcor_av[ip-1], xcor_av[ip], a)
    dxe = dle(xcor_av[ip-1], xcor_av[ip], xcor_er[ip-1], xcor_er[ip], a)
    file16.write(fs.f555.format(ip*a, xcor_av[ip-1], xcor_er[ip-1], dx, dxe)) 
    file21.write(fs.f555.format(ip*a, xcor_av[ip-1], xcor_er[ip-1], dx, dxe)) 
file16.write('\n')
file16.write(' <x(0)x(t)> cooled correlation function\n') 
file22.write('#<x(0)x(t)> cooled correlation function\n')
for ip in range(1,n_p):
    dx  = dl(xcool_av[ip-1], xcool_av[ip], a)
    dxe = dle(xcool_av[ip-1], xcool_av[ip], xcool_er[ip-1], xcool_er[ip], a)
    file16.write(fs.f555.format(ip*a, xcool_av[ip-1], xcool_er[ip-1], dx, dxe)) 
    file22.write(fs.f555.format(ip*a, xcool_av[ip-1], xcool_er[ip-1], dx, dxe)) 

#------------------------------------------------------------------------------
#     <x^2(0)x^2(t) correlator requires subtraction                          
#------------------------------------------------------------------------------
 
xx_sub = x2cor_av[n_p-1]
xx_er  = x2cor_er[n_p-1]
for ip in range(n_p):
    x2sub_av[ip] = x2cor_av[ip]-xx_sub
    x2sub_er[ip] = np.sqrt(x2cor_er[ip]**2+xx_er**2)
#------------------------------------------------------------------------------
#     x^3(0)x^3(t) correlator                                                
#------------------------------------------------------------------------------

file16.write('\n')
file16.write(' <x^2(0)x^2(t)> correlation function\n')
file27.write('#<x^2(0)x^2(t)> correlation function\n')      
for ip in range(1,n_p):
    dx  = dl(x2sub_av[ip-1], x2sub_av[ip],a)
    dxe = dle(x2sub_av[ip-1], x2sub_av[ip], x2sub_er[ip-1],x2sub_er[ip],a)
    file16.write(fs.f555.format(ip*a, x2cor_av[ip-1], x2cor_er[ip-1], dx, dxe))
    file26.write(fs.f555.format(ip*a, x2cor_av[ip-1], x2cor_er[ip-1], dx, dxe))     
xx_sub = x2cool_av[n_p-1]
xx_er  = x2cool_er[n_p-1]
for ip in range(n_p):
    x2cool_sub_av[ip] = x2cool_av[ip]-xx_sub
    x2cool_sub_er[ip] = np.sqrt(x2cool_er[ip]**2+xx_er**2)
file16.write('\n')
file16.write(' <x^2(0)x^2(t)> cooled correlation function\n')
file27.write('#<x^2(0)x^2(t)> cooled correlation function\n')
for ip in range(1,n_p):
    dx = dl(x2cool_sub_av[ip-1], x2cool_sub_av[ip], a)
    dxe=dle(x2cool_sub_av[ip-1], x2cool_sub_av[ip], x2cool_sub_er[ip-1], x2cool_sub_er[ip], a)
    file16.write(fs.f555.format(ip*a, x2cool_av[ip-1], x2cool_er[ip-1], dx, dxe))
    file27.write(fs.f555.format(ip*a, x2cool_av[ip-1], x2cool_er[ip-1], dx, dxe))
    
#------------------------------------------------------------------------------
#     x^3(0)x^3(t) correlator                                                
#------------------------------------------------------------------------------

file16.write('\n')
file16.write(' <x^3(0)x^3(t)> correlation function\n')
file28.write('#<x^3(0)x^3(t)> correlation function\n')      
for ip in range(1,n_p):
    dx  = dl(x3cor_av[ip-1], x3cor_av[ip],a)
    dxe = dle(x3cor_av[ip-1],x3cor_av[ip], x3cor_er[ip-1],x3cor_er[ip],a)
    file16.write(fs.f555.format(ip*a,x3cor_av[ip-1],x3cor_er[ip-1],dx,dxe))
    file28.write(fs.f555.format(ip*a,x3cor_av[ip-1],x3cor_er[ip-1],dx,dxe))     
file16.write('\n')
file16.write(' <x^3(0)x^3(t)> cooled correlation function\n')
file29.write('#<x^3(0)x^3(t)> cooled correlation function\n')
for ip in range(1,n_p):
    dx  = dl(x3cool_av[ip-1],x3cool_av[ip],a)
    dxe = dle(x3cool_av[ip-1],x3cool_av[ip], x3cool_er[ip-1],x3cool_er[ip],a)
    file16.write(fs.f555.format(ip*a,x3cool_av[ip-1],x3cool_er[ip-1],dx,dxe))
    file29.write(fs.f555.format(ip*a,x3cool_av[ip-1],x3cool_er[ip-1],dx,dxe))

#------------------------------------------------------------------------------
#     instanton density                                                      
#------------------------------------------------------------------------------

file16.write('\n')
file16.write(' number of instantons\n')
file23.write('#number of instantons\n')
for ic in range(icool+1):
    file16.write(fs.f556.format(ic, nin_av[ic], nin_er[ic], de*tmax, de2*tmax)) 
    file23.write(fs.f556.format(ic, nin_av[ic], nin_er[ic], de*tmax, de2*tmax))       
file16.write('\n')
file16.write(' action vs cooling sweeps\n')
file24.write('#action vs cooling sweeps\n')
for ic in range(icool+1):    
    sin = nin_av[ic]*s0
    file16.write(fs.f556.format(ic,nin_av[ic], nin_er[ic], de*tmax, de2*tmax)) 
    file24.write(fs.f556.format(ic,nin_av[ic], nin_er[ic], de*tmax, de2*tmax))                          
file16.write('\n')
file16.write(' action per instanton, S_0 = ')
file16.write(str(4.0/3.0*f**3))
file16.write('\n')
file25.write('#action vs cooling sweeps\n')
file25.write(str(4.0/3.0*f**3))
file25.write('\n')
 
for ic in range(icool+1):
    si_av= scool_av[ic]/nin_av[ic]                    
    del2 =(scool_er[ic]/scool_av[ic])**2+(nin_er[ic]/nin_av[ic])**2
    si_er= si_av*np.sqrt(del2)
    file16.write(fs.f443.format(ic,si_av,si_er,s0))
    file25.write(fs.f443.format(ic,si_av,si_er,s0))

#------------------------------------------------------------------------------
#   histograms                                                             
#------------------------------------------------------------------------------
         
file16.write('\n')
file16.write(' x distribution \n')
#plot_histogram2(xhist_min, nxhist , ix)
plot_hist(xhist_min, stxhist, n, nxhist)
file16.write('\n')
file16.write(' z distribution \n')
#plot_histogram2(0.0,nzhist,iz)
plot_hist(0.0, stzhist, n, nzhist)
for i in range(nzhist): 
    xx = (i+0.5)*stzhist
    file30.write(fs.f222.format(xx,iz[i]))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

file16.close()
file17.close()
file18.close()
file19.close()
file20.close()
file21.close()
file22.close()
file23.close()
file24.close()
file25.close()
file26.close()
file27.close()
file28.close()
file29.close()
file30.close()








