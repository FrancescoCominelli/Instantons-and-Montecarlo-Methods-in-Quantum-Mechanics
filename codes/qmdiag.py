import numpy as np
import format_strings as fs
import functions as fn 
import re

#Direct diagonalization of quantum mechanical anharmonic oscillator

#Hamiltonian m/2(\dot x)^2+k(x^2-f^2)^2, units 2m=k=1
#Harmonic oscillator: H_0=m/2(\dot x)^2+m/2*w^2x^2
#Perturbation: H_1=A*x^4+B*x^2+C

# open the file for reading
with open('parameters.txt', 'r') as file:
    # read the contents of the file
    contents = file.read()
   
# search for the values of f and a using regular expressions
f    = re.search(r'f\s*=\s*(\d+\.\d+)', contents).group(1)
ndim = re.search(r'ndim\s*=\s*(\d+)', contents).group(1)
w0   = re.search(r'w0\s*=\s*(\d+\.\d+)', contents).group(1)

# convert the values to integers
f    = float(f)
ndim = int(ndim)
w0   = float(w0)

eps = 1.e-30
pi = np.pi

taumax = 2.5
ntau = 100
dtau = taumax / float(ntau)

xmax = 2.0 * f
nx = 100
dx = 2.0 * xmax / float(nx)

h = np.zeros((ndim, ndim), dtype=np.float32)
e = np.zeros(ndim, dtype=np.float32)
v = np.zeros((ndim, ndim), dtype=np.float32)
psi = np.zeros(ndim, dtype=np.float32)
rho = np.zeros(ndim, dtype=np.float32)
rho2 = np.zeros(ndim, dtype=np.float32)
rho3 = np.zeros(ndim, dtype=np.float32)
xcorp = np.zeros((ntau + 1, ntau + 1), dtype=np.float32)
x3corp = np.zeros((ntau + 1, ntau + 1), dtype=np.float32)
x2corp = np.zeros((ntau + 1, ntau + 1), dtype=np.float32)

# Open files with specified file names and modes
file16 = open('Data/qmdiag.dat', 'w')
file17 = open('Data/cor.dat', 'w')
file18 = open('Data/cor2.dat', 'w')
file19 = open('Data/z.dat', 'w')
file20 = open('Data/psi.dat', 'w')
file21 = open('Data/dcor.dat', 'w')

# Write information to the files

file16.write('qmdiag 1.0\n')
file16.write('----------\n')
file16.write(fs.f601.format(f, ndim))
file16.write(fs.f602.format(taumax, ntau))
file16.write(fs.f603.format(xmax, nx))

#Initialize parameters
m = 0.5
w = w0

a = 1.0
b = -2.0 * f**2 - w**2 / 4.0
c = f**4

cw = 1.0 / np.sqrt(m*w)

c22 = cw**2/2.0
c44 = cw**4/4.0

#Build up Hamiltonian matrix h

for n in range(ndim):
    # <n|h|n>
    x4 = c44 * 3.0 * ((n + 1) ** 2 + n ** 2)
    x2 = c22 * (2 * n + 1)
    e0 = w * (n + 0.5) + c
    h[n, n] = a * x4 + b * x2 + e0

    # <n|h|n+2>
    if n + 2 < ndim:
        x4 = c44 * np.sqrt((n + 1.0) * (n + 2)) * (4 * n + 6)
        x2 = c22 * np.sqrt((n + 1.0) * (n + 2))
        hh = a * x4 + b * x2
        h[n, n + 2] = hh
        h[n + 2, n] = hh

    # <n|h|n+4>
    if n + 4 < ndim:
        x4 = c44 * np.sqrt((n + 1.0) * (n + 2) * (n + 3) * (n + 4))
        hh = a * x4
        h[n, n + 4] = hh
        h[n + 4, n] = hh
    
# Diagonalize h using numpy's eigh function

e, v = np.linalg.eigh(h)

# Sort the eigenvalues and eigenvectors in ascending order and removing non physical values

sorted_indices = np.argsort(e)
e = e[sorted_indices]
v = v[:, sorted_indices]
#positive_indices = (e >= 0)
#e = e[positive_indices]
#v = v[:, positive_indices]

# Energy eigenvalues and matrix elements <0|x|n>

print('\n')
file16.write('\n')
print(fs.f901.format())
file16.write(fs.f901.format())

for n in range(ndim):
    cn = 0.0
    dn = 0.0
    en = 0.0
    for k in range(ndim):
        km3 = max(k-3,0)
        km2 = max(k-2,0)
        km1 = max(k-1,0)
        kp1 = min(k+1,ndim-1)
        kp2 = min(k+2,ndim-1)
        kp3 = min(k+3,ndim-1)
        cn += (np.sqrt(float(k))*v[km1,0]
              + np.sqrt(float(k+1))*v[kp1,0]) * v[k,n]
        dn += (np.sqrt(float(k*(k-1)))*v[km2,0]
              + (2*k+1)*v[k,0]
              + np.sqrt(float((k+1)*(k+2)))*v[kp2,0]) * v[k,n]
        en += (np.sqrt(float(k*(k-1)*(k-2)))*v[km3,0]
              + 3*k*np.sqrt(float(k))*v[km1,0]
              + 3*(k+1)*np.sqrt(float(k+1))*v[kp1,0]
              + np.sqrt(float((k+1)*(k+2)*(k+3)))*v[kp3,0]) * v[k,n] 
        
    rho[n] = cw**2 * cn**2  #dipole matrix element squared
    rho2[n] = cw**4 * dn**2 #quadrupole matrix element squared
    rho3[n] = cw**6 * en**2 #octupole matrix element squared
    
    print(fs.f551.format(n, e[n], rho[n], rho2[n], rho3[n]))
    file16.write(fs.f551.format(n, e[n], rho[n], rho2[n], rho3[n]))

# Groundstate wave function 

file16.write('\n')
file16.write(fs.f902.format()) 

xnorm = 0.0
xnorm2 = 0.0

for k in range(nx+1):
    x = -xmax + k*dx
    psix = 0.0
    fn.psiosc(m, w, ndim-1, x, psi)
    
    for j in range(ndim):
        psix += v[j,0] * psi[j]
    
    # Compare to simple model
    psip = (2.0 * f / pi)**0.25 * np.exp(-f * (x - f)**2)
    psim = (2.0 * f / pi)**0.25 * np.exp(-f * (x + f)**2)
    psi0 = 1.0 / np.sqrt(2.0) * (psip + psim)
    
    # Check normalization
    xnorm += dx * psix**2
    xnorm2 += dx * psi[0]**2
    
    file16.write(fs.f555.format(x, psix, psix**2, psi0, psi0**2))
    file20.write(fs.f555.format(x, psix, psix**2, psi0, psi0**2))

xnorm3 = 0.0
for j in range(ndim):
    xnorm3 += v[j,0]**2

# Print normalization results
print("\nnorm ", xnorm, xnorm2, xnorm3)

# Coordinate space correlator

file16.write('\n')
file16.write(fs.f903.format())
file17.write(fs.f906.format())

for k in range(ntau+1):
    tau = k * dtau
    xcor = 0.0
    for j in range(1, n):
        xcor += rho[j] * np.exp(-(e[j] - e[0]) * tau)
        xcorp[k][j] = xcor
    file16.write(fs.f555.format(tau, xcor, xcorp[k][1], xcorp[k][3], xcorp[k][5]))
    file17.write(fs.f333.format(tau, xcor, 0.01))

# Log derivative

file16.write('\n')
file16.write(fs.f904.format())

for k in range(ntau+1):
    tau = k * dtau
    xcor = xcorp[k][n-1]
    file16.write(fs.f555.format(tau, np.log(xcor + eps), np.log(xcorp[k][1] + eps), np.log(xcorp[k][3] + eps), np.log(xcorp[k][5] + eps)))

file16.write('\n')
file16.write(fs.f905.format())

for k in range(ntau):
    tau = k * dtau
    dlog = -(np.log(xcorp[k+1][n-1] + eps) - np.log(xcorp[k][n-1] + eps)) / dtau
    dlog1 = -(np.log(xcorp[k+1][1] + eps) - np.log(xcorp[k][1] + eps)) / dtau
    dlog3 = -(np.log(xcorp[k+1][3] + eps) - np.log(xcorp[k][3] + eps)) / dtau
    dlog5 = -(np.log(xcorp[k+1][5] + eps) - np.log(xcorp[k][5] + eps)) / dtau
    file16.write(fs.f555.format(tau, dlog, dlog1, dlog3, dlog5))

#x^2 and x^3 correlator

file16.write('\n')
file16.write(fs.f907.format())
file18.write(fs.f908.format())

for k in range(ntau+1):
    tau = k*dtau
    x1cor = 0.0
    x2cor = 0.0 
    x3cor = 0.0
    dx1cor = 0.0
    dx2cor = 0.0
    dx3cor = 0.0
    for j in range(n):
        x1cor += rho[j]*np.exp(-(e[j]-e[0])*tau)
        x2cor += rho2[j]*np.exp(-(e[j]-e[0])*tau)
        x3cor += rho3[j]*np.exp(-(e[j]-e[0])*tau)
        dx1cor += rho[j]*(e[j]-e[0])*np.exp(-(e[j]-e[0])*tau)
        dx2cor += rho2[j]*(e[j]-e[0])*np.exp(-(e[j]-e[0])*tau)
        dx3cor += rho3[j]*(e[j]-e[0])*np.exp(-(e[j]-e[0])*tau)
        x2corp[k][j] = x2cor
        x3corp[k][j] = x3cor
    dx1 = dx1cor/x1cor
    dx2 = dx2cor/(x2cor-rho2[0])
    dx3 = dx3cor/x3cor
    
    file16.write(fs.f555.format(tau, x2cor, x2corp[k][0], x2corp[k][2], x2corp[k][4]))
    file18.write(fs.f333.format(tau, x2cor, 0.01))
    file21.write(fs.f777.format(tau, x1cor, x2cor, x3cor, dx1, dx2, dx3))

file16.write('\n')
file18.write('\n')
file16.write(fs.f909.format())
file18.write(fs.f910.format())

for k in range(ntau+1):
    tau = k*dtau
    x3xor = x3corp[k][n-1]
    file16.write(fs.f555.format(tau, x3cor, x3corp[k][1], x3corp[k][3], x3corp[k][5]))
    file18.write(fs.f333.format(tau, x3cor, 0.01))

# Partition function

xlmax = 100.0
xlmin = 0.1
xlogmax = np.log(xlmax)
xlogmin = np.log(xlmin)
nl = 50
dlog = (xlogmax-xlogmin)/float(nl)

for il in range(nl+1):
    xlog = xlogmin+il*dlog
    xl = np.exp(xlog)
    t  = 1.0/xl
    z  = 1.0
    for i in range(1, ndim):
        z = z + np.exp(-(e[1]-e[0])*xl)
    p = t*np.log(z) - e[0]
    file19.write(fs.f333.format(t, xl, p))

# Close the files

file16.close()
file17.close()
file18.close()
file19.close()
file20.close()
file21.close()



