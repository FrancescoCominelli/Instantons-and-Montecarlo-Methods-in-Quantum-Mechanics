import numpy as np
#------------------------------------------------------------------------------
#   Definition of functions needed
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#     function to include value a in histogram array hist(n)               
#------------------------------------------------------------------------------
#   Imput:
#       a       value to be added to histogram array                        
#       amin    minimum value in histogram                                  
#       st      bin width                                                   
#       m       number of bins                                              
#       hist(n) histogram array                                             
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
#       n     number of measurements
#       xtot  sum of x_i
#       x2tot sum of x**2
#   Output:
#       xav   average
#       xerr  error estimate
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
#   Compute rescaled Hermite polynomials H_i(x)/2^i/sqrt(i!) for i = 0 to n
#------------------------------------------------------------------------------
#   Imput:
#       n (int)   the highest degree of Hermite polynomials to compute
#       x (float) the value of x at which to evaluate the Hermite polynomials
#       p (list)  a list of length n+1 to store the computed values of Hermite 
#                 polynomials
#------------------------------------------------------------------------------
def hermite3(n, x, p):    
    p[0] = 1.0
    p[1] = x
    for i in range(2, n + 1):
        p[i] = (x * p[i - 1] - np.sqrt(i - 1.0) * p[i - 2] / 2.0) / np.sqrt(float(i))

#------------------------------------------------------------------------------
#   Define harmonic oscillator wave function psi(i=0,..,n)=psi_i(x)
#------------------------------------------------------------------------------
#   Imput:
#       m (float)  mass of the harmonic oscillator
#       w (float)  frequency of harmonic oscillator
#       n (int)    the highest index of the harmonic oscillator wave functions 
#                  to compute
#       x (float)  the value of x at which to evaluate the harmonic oscillator 
#                  wave functions
#       psi (list) a list of length n+1 to store the computed values of the 
#                  harmonic oscillator wave functions.
#------------------------------------------------------------------------------
def psiosc(m, w, n, x, psi):    
    h = np.zeros(n+1)  # array to store Hermite polynomials
    y = np.sqrt(m * w) * x
    hermite3(n, y, h)
    for i in range(n + 1):
        xnorm = (m * w / np.pi) ** 0.25 * 2.0 ** (i / 2.0)
        psi[i] = xnorm * h[i] * np.exp(-m * w / 2.0 * x ** 2)
    return psi

#------------------------------------------------------------------------------
#   discretized action for configuration x(n)                           
#------------------------------------------------------------------------------
#   Input:
#       f    minimum of anharmonic oscillator potential
#       a    lattice spacing
#       n    number of lattice points
#       x    field configuration
#   Output:
#       stot total action
#       ttot total time
#       vtot total potential
#------------------------------------------------------------------------------
def act(f, a, n, x):
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
#   return number and location of (anti) instantons                    
#------------------------------------------------------------------------------
#   Input:
#       f  minimum of anharmonic oscillator potential
#       a  lattice spacing
#       n  number of lattice points
#       x  field configuration
#       xi instanton location
#       xa anti-instanton location
#       z  cooled configuration
#   Output:
#       ni inatanton location
#       na anti-instanton location
#------------------------------------------------------------------------------
def inst(f, a, n, x, xi, xa, z):
    ni = 0
    na = 0
    nin= 0
    ix = int(np.sign(x[0]))
    for i in range(1,n):
        tau = a * i
        ixp = int(np.sign(x[i]))
        if ixp > ix:
            ni  += 1
            nin += 1
            xi[ni] = tau
            z[nin] = tau
                     
        elif ixp < ix:
            na  += 1
            nin += 1 
            xa[na] = tau
            z[nin] = tau                     
        ix = ixp
    return ni, na

#------------------------------------------------------------------------------
#   discretized log derivatie                                              
#------------------------------------------------------------------------------
#   Input:
#       xcor1 precedent
#       xcor2 successive
#       a     lattice spacing
#   Output:
#       dl    discretized log derivative
#------------------------------------------------------------------------------
def dl(xcor1, xcor2, a):      
    dl = (xcor1-xcor2)/(xcor1*a)
    return dl

#------------------------------------------------------------------------------
#   error of discretized log derivative                                               
#------------------------------------------------------------------------------
#   Input:
#       xcor1  precedent
#       xcor2  successive
#       xcor1e precedent error
#       xcor2e successive error
#   Output:
#       dle    discretized error of log derivative
#------------------------------------------------------------------------------
def dle(xcor1, xcor2, xcor1e, xcor2e):      
    dle2 = (xcor2e/xcor1)**2+(xcor1e*xcor2/xcor1**2)**2
    dle  = np.sqrt(dle2)
    return dle

#------------------------------------------------------------------------------
#   sum ansatz path                                                  
#------------------------------------------------------------------------------
#   Input:
#       nin  instanton number
#       z    instanton configuration
#       f    minimum of anharmonic oscillator potential
#       t    time
#   Output:
#       xsum discretized error of log derivative
#------------------------------------------------------------------------------
def xsum(nin, z, f, t):
    neven = nin - (nin % 2)
    xsum = -f
    for i in range(0, neven, 2):
        xsum += f * np.tanh(2.0 * f * (t - z[i])) - f * np.tanh(2.0 * f * (t - z[i+1]))
    if nin % 2 != 0:
        xsum += f * np.tanh(2.0 * f * (t - z[nin])) + f   
    return xsum

#------------------------------------------------------------------------------
#   sumansatz configuration on grid x(n)                                
#------------------------------------------------------------------------------
#   Imput:
#       n  number of lattice points
#       x   field configurations
#       nin  instanton number
#       z   instanton configuration
#       f  minimum of anharmonic oscillator potential
#       a  lattice spacing
#------------------------------------------------------------------------------
def xconf(n, x, nin, z, f, a):
    for j in range(n):  
         xx = a*j
         x[j] = xsum(nin,z,f,xx)        
    x[0] = x[n-1]
    x[n] = x[1]
    return

#------------------------------------------------------------------------------
#   hard core                                                           
#------------------------------------------------------------------------------
#   Input:
#       z     instanton configuration
#       nin   instanton number
#       tcore time of the core
#       score action of the core
#       tmax  maximal time
#   Output:
#       shc   hard core action
#------------------------------------------------------------------------------
def sshort(z, nin, tcore, score, tmax):
    shc = 0.0
    if tcore == 0:
        return shc
    for i in range(nin+1):
        if i == 0:
            zm = z[nin] - tmax
        else:
            zm = z[i-1]
        dz = z[i] - zm
        shc = shc + score * np.exp(-dz/tcore)
    return shc