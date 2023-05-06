# Definition of functions
import numpy as np
import matplotlib.pyplot as plt

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
#   Compute rescaled Hermite polynomials H_i(x)/2^i/sqrt(i!) for i = 0 to n.
#     Parameters:
#       n (int): The highest degree of Hermite polynomials to compute.
#       x (float): The value of x at which to evaluate the Hermite polynomials.
#       p (list): A list of length n+1 to store the computed values of Hermite polynomials.
#------------------------------------------------------------------------------

def hermite3(n, x, p):
    
    p[0] = 1.0
    p[1] = x

    for i in range(2, n + 1):
        p[i] = (x * p[i - 1] - np.sqrt(i - 1.0) * p[i - 2] / 2.0) / np.sqrt(float(i))

#------------------------------------------------------------------------------
#   Define harmonic oscillator wave function
#       Harmonic oscillator wave functions psi(i=0,..,n)=psi_i(x)
#       Parameters:
#           m (float): mass of the harmonic oscillator
#           w (float): frequency of harmonic oscillator
#           n (int): The highest index of the harmonic oscillator wave functions to compute.
#           x (float): The value of x at which to evaluate the harmonic oscillator wave functions.
#           psi (list): A list of length n+1 to store the computed values of the harmonic oscillator wave functions.
#------------------------------------------------------------------------------

def psiosc(m, w, n, x, psi):    

    h = np.zeros(n+1)  # array to store Hermite polynomials

    y = np.sqrt(m * w) * x
    hermite3(n, y, h)

    for i in range(n + 1):
        xnorm = (m * w / np.pi) ** 0.25 * 2.0 ** (i / 2.0)
        psi[i] = xnorm * h[i] * np.exp(-m * w / 2.0 * x ** 2)

    return psi
