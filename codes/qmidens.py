import numpy as np
import format_strings as fs
import random
import functions as fn
import re

#------------------------------------------------------------------------------
#   lattice calculation in quantum mechanics.                                                                         
#------------------------------------------------------------------------------
#   calculate instanton density by adiabatically switching from gaussian
#   approximation to full potential. Have
#    d(inst)=d(gaussian)*exp(-S(non-gaussian))
#   where S(non-gaussian)=\int d\alpha <S-S(gaussian)>_\alpha. perform
#   reference calculation of fluctuations around trivial vacuum.
#------------------------------------------------------------------------------
#   instanton is placed at beta/2. anti-symmetric boundary conditions are
#   used. position is fixed during update by requiring x(beta/2)=0.
#------------------------------------------------------------------------------
#   action m/2(\dot x)^2+k(x^2-f^2)^2, units 2m=k=1. 
#------------------------------------------------------------------------------

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
#   set the values
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
neq    = re.search(r'neq\s*=\s*(\d+)', contents).group(1)
nmc    = re.search(r'nmc\s*=\s*(\d+)', contents).group(1)
delx   = re.search(r'delx\s*=\s*(\d+\.\d+)', contents).group(1)
kp     = re.search(r'kp\s*=\s*(\d+)', contents).group(1)
w0     = re.search(r'w0\s*=\s*(\d+\.\d+)', contents).group(1)
nalpha = re.search(r'nalpha\s*=\s*(\d+)', contents).group(1)

# convert the values to integers
f      = float(f)
n      = int(n)
a      = float(a)
icold  = int(icold)
neq    = int(neq)
nmc    = int(nmc)
delx   = float(delx)
kp     = int(kp)
w0     = float(w0)
nalpha = int(nalpha)


dalpha = 1.0/float(nalpha)
beta   = n*a
tau0   = beta/2.0
n0     = n/2
n0p    = int(n0+1)
n0m    = int(n0-1)
sng    = 0.0
svacng = 0.0
pi     = np.pi

#------------------------------------------------------------------------------
# echo input parameters
#------------------------------------------------------------------------------
#   open txt files

file16 = open('Data/qmidens.dat', 'w')
file17 = open('Data/idens_conf.dat', 'w')
file18 = open('Data/vac_conf.dat', 'w')

# write on a txt file the values

file16.write('lattice qm idens 1.0\n')
file16.write('--------------------\n')
file16.write(fs.f101.format(f,n,a))
file16.write(fs.f102.format(nmc,neq))
file16.write(fs.f106.format(delx,nalpha))
file16.write('\nfluctuations around instanton path\n')
file16.write('------------------------------------\n')

#------------------------------------------------------------------------------
#     initialize                                                             
#------------------------------------------------------------------------------
random.seed(seed)

stot_sum    = 0.0
stot2_sum   = 0.0
vav_sum     = 0.0
vav2_sum    = 0.0
valpha_sum  = 0.0
valpha2_sum = 0.0

x      = np.zeros(n)
w      = np.zeros(n)
x0     = np.zeros(n)
vi     = np.zeros(n)
va_av  = np.zeros(2 * nalpha + 1)
va_err = np.zeros(2 * nalpha + 1)

#------------------------------------------------------------------------------
#     initialize instanton and gaussian potential                                                            
#------------------------------------------------------------------------------

for i in range(n):
    tau   = i*a
    x[i]  = f*np.tanh(2.0*f*(tau-tau0))
    x0[i] = x[i]
    w[i]  = -4.0*(f**2-3.0*x0[i]**2)
    vi[i] = (x0[i]**2-f**2)**2
    file17.write(fs.f222.format(tau, x[i]))

#------------------------------------------------------------------------------
#     anti-periodic boundary conditions                                                            
#------------------------------------------------------------------------------

x[n-1] = -x[0]
x      = np.append(x, -x[1])

#------------------------------------------------------------------------------
#     initial actions                                                          
#------------------------------------------------------------------------------

stot = 0

for i in range(n):
    xp = (x[i+1]-x[i])/a
    t  = 1.0/4.0*xp**2    
    v  = (x[i]**2-f**2)**2
    s  = a*(t+v)
    stot += s
    
s0   = 4.0/3.0*f**3
dens = 8*np.sqrt(2.0/pi)*f**2.5*np.exp(-s0)
f0   = dens

file16.write("\n")
file16.write(fs.f301.format(f0, stot, s0))

#------------------------------------------------------------------------------
#     loop over coupling constant (up/down)                                                          
#------------------------------------------------------------------------------

for ialpha in range(2*nalpha+1):
    if ialpha <= nalpha:
        alpha = ialpha * dalpha
    else:
        alpha = 2.0 - ialpha * dalpha
    
    nacc = 0
    nhit = 0    
    nconf= 0
    ncor = 0

#------------------------------------------------------------------------------
#    monte carlo sweeps                                                     
#------------------------------------------------------------------------------
    for i in range(nmc):
        nconf += 1
        if i == neq:
            nconf = 0
            stot_sum   = 0.0
            stot2_sum  = 0.0
            vav_sum    = 0.0
            vav2_sum   = 0.0 
            valpha_sum = 0.0
            valpha2_sum= 0.0

#--------------------------------------------------------------------------
#   one sweep thorough configuration                                       
#--------------------------------------------------------------------------
    for j in range(1,n):
        nhit += 1
    
#--------------------------------------------------------------------------
#   fix instanton center                                       
#--------------------------------------------------------------------------
        if j == n0:
            break

#--------------------------------------------------------------------------
#   old action                                       
#--------------------------------------------------------------------------
        xpm   = (x[j]-x[j-1])/a
        xpp   = (x[j+1]-x[j])/a
        t     = 1.0/4.0*(xpm**2+xpp**2)
        v0    = 1.0/2.0*w[j]*(x[j]-x0[j])**2+vi[j]
        v1    = (x[j]**2-f**2)**2
        v     = alpha*(v1-v0) + v0
        sold  = a*(t+v)

#--------------------------------------------------------------------------
#   jakobian of constraint                                       
#--------------------------------------------------------------------------
        if j == n0m or j == n0p:
            vel = (x[n0p] - x[n0m]) / (2.0 * a)
            sjak = -np.log(abs(vel))
            sold = sold + sjak

#--------------------------------------------------------------------------
#   MC hit                                       
#--------------------------------------------------------------------------
        xnew = x[j] + delx*(2.0*random.random()-1.0)

#--------------------------------------------------------------------------
#   new action                                       
#--------------------------------------------------------------------------
        xpm   = (xnew-x[j-1])/a
        xpp   = (x[j+1]-xnew)/a
        t     = 1.0/4.0*(xpm**2+xpp**2)
        v0    = 1.0/2.0*w[j]*(xnew-x0[j])**2+vi[j]
        v1    = (xnew**2-f**2)**2
        v     = alpha*(v1-v0) + v0
        snew  = a*(t+v)

#--------------------------------------------------------------------------
#   jakobian                                     
#--------------------------------------------------------------------------
        if j == n0m:
            vel = (x[n0p] - xnew) / (2.0 * a)
            sjak = -np.log(abs(vel))
            snew = snew + sjak
        elif j == n0p:
            vel = (xnew - x[n0m]) / (2.0 * a)
            sjak = -np.log(abs(vel))
            snew = snew + sjak

#--------------------------------------------------------------------------
#   accept/reject                                     
#--------------------------------------------------------------------------
        dels  = snew-sold
        dels  = min(dels,70.0)
        dels  = max(dels,-70.0)
        if np.exp(-dels) > random.random():
            x[j]  = xnew
            nacc += 1
    x[n-1] = -x[0]
    x      = np.append(x, -x[1])
    
#--------------------------------------------------------------------------
#   calculate action and other things                                                  
#--------------------------------------------------------------------------
    stot = 0.0
    ttot = 0.0
    vtot = 0.0
    ptot = 0.0
    for j in range(n):
        xp = (x[j+1]-x[j])/a
        t  = 1.0/4.0*xp**2
        v0 = 1.0/2.0*w[j]*(x[j]-x0[j])**2+vi[j]
        v1 = (x[j]**2-f**2)**2
        v  = alpha*(v1-v0) + v0
        s  = a*(t+v)
        ttot += a*t
        vtot += a*v
        stot += s
        ptot += a*(v1-v0)
    
    if i % kp == 0:
        print("configuration:   ", i, "\n",
              "coupling:        ", alpha, "\n",
              "acceptance rate: ", float(nacc)/float(nhit), "\n",
              "action (T,V):    ", stot, ttot, vtot)

#--------------------------------------------------------------------------
#   include in sample                                                     
#--------------------------------------------------------------------------
    stot_sum    += stot
    stot2_sum   += stot**2
    vav_sum     += vtot/beta
    vav2_sum    += vtot**2/beta
    valpha_sum  += ptot/beta
    valpha2_sum += ptot**2/beta

#--------------------------------------------------------------------------
#   averages                                                     
#--------------------------------------------------------------------------
    stot_av, stot_err     = fn.disp(nconf, stot_sum, stot2_sum)
    v_av, v_err           = fn.disp(nconf, vav_sum, vav2_sum)
    valpha_av, valpha_err = fn.disp(nconf, valpha_sum, valpha2_sum)

    va_av[ialpha]  = valpha_av
    va_err[ialpha] = valpha_err
    
    if ialpha % (2 * nalpha) == 0:
        da = dalpha / 4.0
    else:
        da = dalpha / 2.0
    dsng = da * valpha_av
    sng += dsng

#------------------------------------------------------------------------------
#   output                                                               
#------------------------------------------------------------------------------
    file16.write('\n')
    file16.write(fs.f800.format(alpha))
    file16.write(fs.f801.format(stot_av, stot_err))
    file16.write(fs.f802.format(v_av, v_err))
    file16.write(fs.f809.format(valpha_av, valpha_err))
    file16.write(fs.f814.format(sng, dsng))
    file16.write('\n')
    for j in range(n):
        file17.write(fs.f222.format(j*a, x[j]))

#------------------------------------------------------------------------------
#   end of loop over coupling constants                                                                
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#   final estimate of integral over coupling                                                                
#------------------------------------------------------------------------------
sup_sum = 0.0
sup_err = 0.0
sup_hal = 0.0
sdw_sum = 0.0
sdw_err = 0.0
sdw_hal = 0.0

#------------------------------------------------------------------------------
#   have sum=1/2(up+down) and up = 1/2*f0+f1+...+1/2*fn, down=...                                                               
#------------------------------------------------------------------------------

for ia in range(nalpha+1):
    if ia % nalpha == 0:
        da = dalpha / 4.0
    else:
        da = dalpha / 2.0
    iap = ia + nalpha
    sup_sum += da * va_av[ia]
    sup_err += da * va_err[ia] ** 2
    sdw_sum += da * va_av[iap]
    sdw_err += da * va_err[iap] ** 2


for ia in range(0, nalpha+1, 2):
    if ia % nalpha == 0:
        da = dalpha / 2.0
    else:
        da = dalpha
    iap = ia + nalpha
    sup_hal += da * va_av[ia]
    sdw_hal += da * va_av[iap]

#------------------------------------------------------------------------------
#   uncertainties                                                                         
#------------------------------------------------------------------------------
ds     = sup_sum + sdw_sum
sng    = ds
dens_ng= dens*np.exp(-sng)
ds_err = np.sqrt(sup_err + sdw_err)
ds_hal = sup_hal + sdw_hal
ds_dif = abs(sup_sum - sdw_sum)
ds_dis = abs(ds - ds_hal)/2.0
ds_tot = np.sqrt(ds_err**2 + ds_dif**2 + ds_dis**2)
dens_er= dens_ng*ds_tot 

#------------------------------------------------------------------------------
#   output                                                                         
#------------------------------------------------------------------------------
file16.write("\n")
file16.write(fs.f815.format(sng, ds_tot))
file16.write(fs.f813.format(ds_err, ds_dif, ds_dis))
file16.write("\nreference calculation: trivial vacuum\n-------------------------------------\n\n")

#------------------------------------------------------------------------------
#   repeat calculation for fluctuations around trivial vacuum                                                                         
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#   clear summation arrays                                                                         
#------------------------------------------------------------------------------
stot_sum   = 0.0
stot2_sum  = 0.0
vav_sum    = 0.0
vav2_sum   = 0.0 
valpha_sum = 0.0
valpha2_sum= 0.0

#------------------------------------------------------------------------------
#   initialize                                                                         
#------------------------------------------------------------------------------
for i in range(n):
    tau  = i*a
    x[i] = f
    x0[i]= x[i]
    w[i] = 8.0*f**2
    file18.write(fs.f222.format(tau, x[i]))

#------------------------------------------------------------------------------
#   periodic boundary conditions                                                                         
#------------------------------------------------------------------------------
x[0] = x[n-1]
x    = np.append(x, x[1])

#------------------------------------------------------------------------------
#   loop over coupling constant (up/down)                                                                          
#------------------------------------------------------------------------------
for ialpha in range(2 * nalpha + 1):
    if ialpha <= nalpha:
        alpha = ialpha * dalpha
    else:
        alpha = 2.0 - ialpha * dalpha
    
    nacc = 0
    nhit = 0    
    nconf= 0
    ncor = 0
    
#------------------------------------------------------------------------------
#    monte carlo sweeps                                                     
#------------------------------------------------------------------------------
    for i in range(nmc):
        nconf += 1
        if i == neq:
            nconf = 0
            stot_sum   = 0.0
            stot2_sum  = 0.0
            vav_sum    = 0.0
            vav2_sum   = 0.0 
            valpha_sum = 0.0
            valpha2_sum= 0.0

#--------------------------------------------------------------------------
#   one sweep thorough configuration                                       
#--------------------------------------------------------------------------
    for j in range(1,n):
        nhit += 1

#--------------------------------------------------------------------------
#   old action                                       
#--------------------------------------------------------------------------
        xpm   = (x[j]-x[j-1])/a
        xpp   = (x[j+1]-x[j])/a
        t     = 1.0/4.0*(xpm**2+xpp**2)
        v0    = 1.0/2.0*w[j]*(x[j]-x0[j])**2
        v1    = (x[j]**2-f**2)**2
        v     = alpha*(v1-v0) + v0
        sold  = a*(t+v)

#--------------------------------------------------------------------------
#   MC hit                                       
#--------------------------------------------------------------------------
        xnew = x[j] + delx*(2.0*random.random()-1.0)

#--------------------------------------------------------------------------
#   new action                                       
#--------------------------------------------------------------------------
        xpm   = (xnew-x[j-1])/a
        xpp   = (x[j+1]-xnew)/a
        t     = 1.0/4.0*(xpm**2+xpp**2)
        v0    = 1.0/2.0*w[j]*(xnew-x0[j])**2
        v1    = (xnew**2-f**2)**2
        v     = alpha*(v1-v0) + v0
        snew  = a*(t+v)

#--------------------------------------------------------------------------
#   accept/reject                                     
#--------------------------------------------------------------------------
        dels  = snew-sold
        dels  = min(dels,70.0)
        dels  = max(dels,-70.0)
        if np.exp(-dels) > random.random():
            x[j]  = xnew
            nacc += 1
    x[n-1] = -x[0]
    x      = np.append(x, -x[1])
    
#--------------------------------------------------------------------------
#   calculate action and other things                                                  
#--------------------------------------------------------------------------
    stot = 0.0
    ttot = 0.0
    vtot = 0.0
    ptot = 0.0
    for j in range(n):
        xp = (x[j+1]-x[j])/a
        t  = 1.0/4.0*xp**2
        v0 = 1.0/2.0*w[j]*(x[j]-x0[j])**2
        v1 = (x[j]**2-f**2)**2
        v  = alpha*(v1-v0) + v0
        s  = a*(t+v)
        ttot += a*t
        vtot += a*v
        stot += s
        ptot += a*(v1-v0)
    
    if i % kp == 0:
        print("configuration:   ", i, "\n",
              "coupling:        ", alpha, "\n",
              "acceptance rate: ", float(nacc)/float(nhit), "\n",
              "action (T,V):    ", stot, ttot, vtot)

#--------------------------------------------------------------------------
#   include in sample                                                     
#--------------------------------------------------------------------------
    stot_sum    += stot
    stot2_sum   += stot**2
    vav_sum     += vtot/beta
    vav2_sum    += vtot**2/beta
    valpha_sum  += ptot/beta
    valpha2_sum += ptot**2/beta

#--------------------------------------------------------------------------
#   averages                                                     
#--------------------------------------------------------------------------
    stot_av, stot_err     = fn.disp(nconf, stot_sum, stot2_sum)
    v_av, v_err           = fn.disp(nconf, vav_sum, vav2_sum)
    valpha_av, valpha_err = fn.disp(nconf, valpha_sum, valpha2_sum)

    va_av[ialpha]  = valpha_av
    va_err[ialpha] = valpha_err
    
    if ialpha % (2 * nalpha) == 0:
        da = dalpha / 4.0
    else:
        da = dalpha / 2.0
    dsvacng = da * valpha_av
    svacng += dsvacng

#------------------------------------------------------------------------------
#   output                                                               
#------------------------------------------------------------------------------
    file16.write('\n')
    file16.write(fs.f800.format(alpha))
    file16.write(fs.f801.format(stot_av, stot_err))
    file16.write(fs.f802.format(v_av, v_err))
    file16.write(fs.f809.format(valpha_av, valpha_err))
    file16.write(fs.f814.format(svacng, dsvacng))
    file16.write('\n')
    for j in range(n):
        file18.write(fs.f222.format(j*a, x[j]))

#------------------------------------------------------------------------------
#   end of loop over coupling constants                                                                  
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#   final estimate of integral over coupling                                                                
#------------------------------------------------------------------------------
svacup_sum = 0.0
svacup_err = 0.0
svacup_hal = 0.0
svacdw_sum = 0.0
svacdw_err = 0.0
svacdw_hal = 0.0

#------------------------------------------------------------------------------
#   have sum=1/2(up+down) and up = 1/2*f0+f1+...+1/2*fn, down=...                                                               
#------------------------------------------------------------------------------

for ia in range(nalpha+1):
    if ia % nalpha == 0:
        da = dalpha / 4.0
    else:
        da = dalpha / 2.0
    iap = ia + nalpha
    svacup_sum += da * va_av[ia]
    svacup_err += da * va_err[ia] ** 2
    svacdw_sum += da * va_av[iap]
    svacdw_err += da * va_err[iap] ** 2


for ia in range(0, nalpha+1, 2):
    if ia % nalpha == 0:
        da = dalpha / 2.0
    else:
        da = dalpha
    iap = ia + nalpha
    svacup_hal += da * va_av[ia]
    svacdw_hal += da * va_av[iap]

#------------------------------------------------------------------------------
#   uncertainties                                                                         
#------------------------------------------------------------------------------
dsvac     = svacup_sum + svacdw_sum
svacng    = dsvac
fvac      = np.exp(-svacng)
dsvac_err = np.sqrt(svacup_err + svacdw_err)
dsvac_hal = svacup_hal + svacdw_hal
dsvac_dif = abs(svacup_sum - svacdw_sum)
dsvac_dis = abs(dsvac - dsvac_hal)/2.0
dsvac_tot = np.sqrt(dsvac_err**2 + dsvac_dif**2 + dsvac_dis**2)
fvac_er   = fvac*dsvac_tot 

#------------------------------------------------------------------------------
#   output                                                                         
#------------------------------------------------------------------------------
file16.write("\n")
file16.write(fs.f1814.format(svacng, dsvac_tot))
file16.write(fs.f813.format(dsvac_err, dsvac_dif, dsvac_dis))
file16.write("\n")
file16.write(fs.f1816.format(fvac, fvac_er))

#------------------------------------------------------------------------------
#   final answer                                                                          
#------------------------------------------------------------------------------
seff    = sng - svacng
seff_er = np.sqrt(ds_tot**2+dsvac_tot**2)
dens_ng = dens*np.exp(-seff)
dens_er = dens_ng*seff_er

file16.write("\n")
file16.write(fs.f817.format(seff, sng, svacng))
file16.write(fs.f818.format(seff, seff_er))
file16.write("\n")
file16.write(fs.f819.format(dens_ng, dens))
file16.write(fs.f820.format(dens_ng, dens_er))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
file16.close()
file17.close()
file18.close()












