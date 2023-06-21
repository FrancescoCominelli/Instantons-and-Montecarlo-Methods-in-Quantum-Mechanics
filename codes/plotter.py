import numpy as np
import matplotlib.pyplot as plt
import re
#------------------------------------------------------------------------------
#   Read useful values from parameters.txt
#------------------------------------------------------------------------------
with open('parameters.txt', 'r') as file:
    # read the contents of the file
    contents = file.read()
   
# search for the values of f and a using regular expressions
n      = re.search(r'n\s*=\s*(\d+)', contents).group(1)
a      = re.search(r'a\s*=\s*(\d+\.\d+)', contents).group(1)

n      = int(n)
a      = float(a)

#------------------------------------------------------------------------------
#   Fig. 1.a: Double well potential with the position of the ground state
#   and the first three excited state.
#------------------------------------------------------------------------------
def f(x):
    return (x**2 - 1.4**2)**2

with open('Data/qmdiag/qmdiag.dat', 'r') as file:
    lines = file.readlines()[7:11]

column1 = [float(line.split()[1]) for line in lines]

x = np.linspace(-2.8, 2.8, 100)
y = f(x)

plt.plot(x, y)

y = np.array(column1)
for i in range(4):
    plt.axhline(y[i], color='green',linewidth = 0.8, linestyle='--')

plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Energy levels')

plt.xlim(-2.5, 2.5)
plt.ylim(0, 10)

plt.show()  

#------------------------------------------------------------------------------
#   Fig. 1.b Spectrum of the double well potential.
#------------------------------------------------------------------------------
with open('Data/qmdiag/spectrum.dat', 'r') as file:
    lines = file.readlines()

for i in range(6):
    column  = [float(line.split()[i]) for line in lines]
    
    y = np.array(column)
    x = list(range(len(y)))
    x = [num / 10 for num in x]
    plt.plot(x, y, color = 'black', linewidth = 0.8)

plt.xlabel('f')
plt.ylabel('E')
plt.title('Energy spectrum')

plt.xlim(0, 2)
plt.ylim(0, 23)

plt.show()

#------------------------------------------------------------------------------
#   FIG. 2: Typical euclidean path obtained in a Monte Carlo simulation of the 
#   discretized euclidean action of the double well potential.
#------------------------------------------------------------------------------

with open('Data/qmcool/config.dat', 'r') as file:
    lines = file.readlines()

start_line = None
end_line   = None
for i, line in enumerate(lines):
    if line.startswith('configuration: 100'):
        start_line = i
    elif line.startswith('configuration: 150'):
        end_line = i
        break
data_lines = lines[start_line+1: end_line]

column1 = [float(line.split()[0]) for line in data_lines]
column2 = [float(line.split()[1]) for line in data_lines]
 
x     = np.array(column1)
y     = np.array(column2)


plt.plot(x, y, color = 'black',linewidth = 0.8, label = 'Monte Carlo')

with open('Data/qmcool/coolconfig.dat', 'r') as file:
    lines = file.readlines()

start_line = None
end_line   = None
for i, line in enumerate(lines):
    if line.startswith('configuration: 100'):
        start_line = i
    elif line.startswith('configuration: 150'):
        end_line = i
        break
data_lines = lines[start_line+1: end_line]

column2 = [float(line.split()[1]) for line in data_lines]
 
y     = np.array(column2)


plt.plot(x, y, color = 'green',linewidth = 0.8, label = 'Cooled')


plt.xlabel('τ')
plt.ylabel('x')
plt.legend()
plt.title('Monte Carlo vs Cooled configurations')

plt.xlim(0, 20)
plt.show()

#------------------------------------------------------------------------------
#   FIG. 3: Probability distribution in the double well potential.
#------------------------------------------------------------------------------
with open('Data/qmdiag/psi.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'red', label = 'exact')

with open('Data/qm/qmdist.dat', 'r') as file:
    lines = file.readlines()[1:51]  # read lines 1 to 51
    
column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'blue', drawstyle = 'steps', label = 'Monte Carlo' )


plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.title('Probability distribution |\u03C8|\u00B2')

plt.show()

#------------------------------------------------------------------------------
#   FIG. 4: Fig. a. Shows the correlation functions
#------------------------------------------------------------------------------

with open('Data/qmdiag/dcor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]
column4 = [float(line.split()[3]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/qm/qmcor.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/qm/qmcor2.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='o', markerfacecolor='none',
             markeredgecolor = 'red',markersize=8, capsize=5, label = '<x²(0)x²(τ)>')

with open('Data/qm/qmcor3.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='D',markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x³(0)x³(τ)>')

plt.xlabel('τ')
plt.ylabel('<xⁿ(0)xⁿ(τ)>')
plt.legend()
plt.title('Correlation functions')

plt.xlim(0, 1.5)
plt.ylim(0, 8)

plt.show()

#------------------------------------------------------------------------------
#   FIG. 4: Fig. b. Shows the logarithmic derivative of the correlators
#------------------------------------------------------------------------------
with open('Data/qmdiag/dcor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[4]) for line in lines]
column3 = [float(line.split()[5]) for line in lines]
column4 = [float(line.split()[6]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/qm/qmcor.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/qm/qmcor2.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='o', markerfacecolor='none',
             markeredgecolor = 'red',markersize=8, capsize=5, label = '<x²(0)x²(τ)>')

with open('Data/qm/qmcor3.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='D',markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x³(0)x³(τ)>')



plt.xlabel('τ')
plt.ylabel('d[log<xⁿ(0)xⁿ(τ)>]/dτ')
plt.legend()
plt.title('Log derivative of correlation functions')

plt.xlim(0, 1.5)
plt.ylim(0, 4.5)

plt.show()

#------------------------------------------------------------------------------
#   FIG. 5: Free energy F = −T log(Z) of the anharmonic oscillator as a 
#   function of the temperature t = i/b
#------------------------------------------------------------------------------
with open('Data/qmdiag/z.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.plot(x, y, color = 'black')



#data from qmswitch.py runned 6 times with different n
y     = np.array([-2.24413, -2.23835, -2.22145, -2.10050, -1.94522,       -1.74119])
y_err = np.array([ 0.02725,  0.05859,  0.03459,  0.05179,  0.03288,        0.02147])
x     = np.array([ 0.2,      0.025,    0.1,      0.5,      0.740740740741, 1.0])
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue',markersize=8, capsize=5)

plt.xlabel('T')
plt.ylabel('F')
plt.title('Free energy of anharmonic oscillator')

plt.xscale('log')
plt.xlim(0.01,2.5)
plt.ylim(-2.5, -1)

plt.show()

#------------------------------------------------------------------------------
#   FIG. 6: Fig. a. Shows the correlation functions
#------------------------------------------------------------------------------

with open('Data/qmdiag/dcor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]
column4 = [float(line.split()[3]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/qmcool/coolcor.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/qmcool/coolcor2.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='o', markerfacecolor='none',
             markeredgecolor = 'red',markersize=8, capsize=5, label = '<x²(0)x²(τ)>')

with open('Data/qmcool/coolcor3.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='D',markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x³(0)x³(τ)>')

plt.xlabel('τ')
plt.ylabel('<xⁿ(0)xⁿ(τ)>')
plt.legend()
plt.title('Cooled correlation functions')

plt.xlim(0, 1.5)
plt.ylim(0, 8)

plt.show()

#------------------------------------------------------------------------------
#   FIG. 6: Fig. b. Shows the logarithmic derivative of the correlators
#------------------------------------------------------------------------------
with open('Data/qmdiag/dcor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[4]) for line in lines]
column3 = [float(line.split()[5]) for line in lines]
column4 = [float(line.split()[6]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/qmcool/coolcor.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/qmcool/coolcor2.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='o', markerfacecolor='none',
             markeredgecolor = 'red',markersize=8, capsize=5, label = '<x²(0)x²(τ)>')

with open('Data/qmcool/coolcor3.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='D',markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x³(0)x³(τ)>')


plt.xlabel('τ')
plt.ylabel('d[log<xⁿ(0)xⁿ(τ)>]/dτ')
plt.legend()
plt.title('Cooled log derivative of correlation functions')

plt.xlim(0, 1.5)
plt.ylim(0, 4.5)

plt.show()

#------------------------------------------------------------------------------
#   FIG. 7: Fig. a. instanton density as a function of the number of cooling 
#   sweeps for different values of the parameter η
#------------------------------------------------------------------------------
with open('Data/qmcool/nin.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[3]) for line in lines]
column4 = [float(line.split()[4]) for line in lines]
x     = np.array(column1)
y     = np.array(column2)/(n*a)


plt.errorbar(x, y, fmt ='s', markerfacecolor = 'none',
             markeredgecolor = 'blue', markersize = 8, capsize = 5, label = 'η = 1.4')

y     = np.array(column3)/(n*a)
plt.plot(x, y, color = 'green', linewidth = 0.8, linestyle = '--')

y     = np.array(column4)/(n*a)
plt.plot(x, y, color = 'green', linewidth = 0.8)

plt.xlabel('n_cool')
plt.ylabel('N_top/\u03B2')
plt.title('Instanton density')

plt.xscale('log')
plt.yscale('log')
plt.xlim(1, )

plt.show()

#------------------------------------------------------------------------------
#   FIG. 7: Fig. b. instanton action as a function of the number of cooling 
#   sweeps for different values of the parameter eta
#------------------------------------------------------------------------------
with open('Data/qmcool/sinst.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]
column4 = [float(line.split()[3]) for line in lines]
x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)

plt.errorbar(x, y, yerr=y_err, fmt='s',markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = 'η = 1.4')

y     = np.array(column4)

plt.plot(x, y, color='green', linewidth = 0.8)

plt.xlabel('n_cool')
plt.ylabel('S/N_inst')
plt.title('Action per instanton')

plt.xscale('log')
plt.yscale('log')
plt.xlim(1, )

plt.show()

#------------------------------------------------------------------------------
#   FIG. 8: Instanton density as a function of the parameter f.
#------------------------------------------------------------------------------
def l1(x):
    return 8*x**(5/2)*np.sqrt(2/np.pi)*np.exp(-4/3*x**3)
def l2(x):
    return 8*x**(5/2)*np.sqrt(2/np.pi)*np.exp(-4/3*x**3-71/72*1/(4/3*x**3))
def dE(x):
    return np.sqrt((6*(4/3)*x**3)/np.pi)*4*x*np.exp(-4/3*x**3)

x = np.linspace(0.02, 2, 100)
y = l1(x)

plt.plot(x, y, color = 'green', linewidth = 0.8, linestyle = '--', label = '1-loop')

y = l2(x)
plt.plot(x, y, color = 'green', linewidth = 0.8, label = '2-loop')

with open('Data/qmdiag/spectrum.dat', 'r') as file:
    lines = file.readlines()

column1  = [float(line.split()[0]) for line in lines]
column2  = [float(line.split()[1]) for line in lines]

E0 = np.array(column1)
E1 = np.array(column2)

y = (E1 - E0)/2 
x = list(range(len(y)))
x = [num / 10 for num in x]
plt.plot(x, y, color = 'black', linewidth = 0.8, label = '\u0394E/2')

y     = np.array([4.36018,  4.0583, 3.52726, 2.82591, 2.41121, 1.10605,0.32266, 0.05903])/5
y_err = np.array([0.04225, 0.04020, 0.03971, 0.03659, 0.03607, 0.02963,0.01941, 0.00839])/5
x     = np.array([    0.5,    0.75,       1,    1.15,    1.25,    1.35,    1.5,    1.65])

plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue',markersize=8, capsize=5, label = 'cooling')

y     = np.array([0.65972, 0.32880,0.11004, 0.05115])
y_err = np.array([0.13140, 0.10810,0.03076, 0.00039])
x     = np.array([   1.25,    1.35,    1.5,    1.65])

plt.errorbar(x, y, yerr=y_err, fmt='D', markerfacecolor='none',
             markeredgecolor = 'red',markersize=8, capsize=5, label = 'Monte Carlo')

plt.xlabel('f')
plt.ylabel('N_top/\u03B2')
plt.title('Instanton density')
plt.legend()

plt.yscale('log')
plt.xlim(0, 1.9)
plt.ylim(0.01, 2)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '' if x == y[0] else f'{x:.1f}'))

plt.show()

#------------------------------------------------------------------------------
#   FIG. 9: Quantum mechanical paths which appear in a Monte-Carlo calculation
#   of the one-instanton partition function in the double well potential.
#------------------------------------------------------------------------------
for i in range(4):
    with open('Data/qmidens/idens_conf.dat', 'r') as file:
        lines = file.readlines()[i*n+1:(i+1)*n]

    column1 = [float(line.split()[0]) for line in lines]
    column2 = [float(line.split()[1]) for line in lines]

    x     = np.array(column1)
    y     = np.array(column2)
    if i==0:
        plt.plot(x, y, color = 'black', linewidth = 0.8)
    else:
        plt.plot(x, y, color = 'green', linewidth = 0.8)
for i in range(4):
    with open('Data/qmidens/vac_conf.dat', 'r') as file:
        lines = file.readlines()[i*n+1:(i+1)*n]

    column1 = [float(line.split()[0]) for line in lines]
    column2 = [float(line.split()[1]) for line in lines]

    x     = np.array(column1)
    y     = np.array(column2)
    if i==0:
        plt.plot(x, y, color = 'black', linewidth = 0.8)
    else:
        plt.plot(x, y, color = 'blue', linewidth = 0.8)

plt.xlabel('τ')
plt.ylabel('X')
plt.title('Quantum mechanical path of one-instanton partition function')

plt.xlim(0, n*a-a)

plt.show()

#------------------------------------------------------------------------------
#   FIG. 10.a Shows the correlation functions
#------------------------------------------------------------------------------

with open('Data/qmdiag/dcor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]
column4 = [float(line.split()[3]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/rilm/rcor.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/rilm/rcor2.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='o', markerfacecolor='none',
             markeredgecolor = 'red',markersize=8, capsize=5, label = '<x²(0)x²(τ)>')

with open('Data/rilm/rcor3.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='D',markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x³(0)x³(τ)>')

plt.xlabel('τ')
plt.ylabel('<xⁿ(0)xⁿ(τ)>')
plt.legend()
plt.title('Random instanton configuration correlation functions')

plt.xlim(0, 1.5)
plt.ylim(0, 8)

plt.show()

#------------------------------------------------------------------------------
#   FIG. 10.b Shows the logarithmic derivative of the correlators
#------------------------------------------------------------------------------
with open('Data/qmdiag/dcor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[4]) for line in lines]
column3 = [float(line.split()[5]) for line in lines]
column4 = [float(line.split()[6]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/rilm/rcor.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/rilm/rcor2.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='o', markerfacecolor='none',
             markeredgecolor = 'red',markersize=8, capsize=5, label = '<x²(0)x²(τ)>')

with open('Data/rilm/rcor3.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='D',markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x³(0)x³(τ)>')

plt.xlabel('τ')
plt.ylabel('d[log<xⁿ(0)xⁿ(τ)>]/dτ')
plt.legend()
plt.title('Random instanton configuration log derivative correlation functions')

plt.xlim(0, 1.5)
plt.ylim(0, 4.5)

plt.show()
#------------------------------------------------------------------------------
#   FIG. 12: Typical euclidean path obtained in a Monte Carlo simulation of the 
#   discretized euclidean action of the double well potential for  = 1.4.
#------------------------------------------------------------------------------

with open('Data/rilm_gauss/config.dat', 'r') as file:
    lines = file.readlines()

start_line = None
end_line   = None
for i, line in enumerate(lines):
    if line.startswith('configuration: 6000'):
        start_line = i
    elif line.startswith('configuration: 6050'):
        end_line = i
        break
data_lines = lines[start_line+1: end_line]

column1 = [float(line.split()[0]) for line in data_lines]
column2 = [float(line.split()[1]) for line in data_lines]
 
x     = np.array(column1)
y     = np.array(column2)


plt.plot(x, y, color = 'black',linewidth = 0.8, label = 'RILM')

with open('Data/rilm_gauss/config_gauss.dat', 'r') as file:
    lines = file.readlines()

start_line = None
end_line   = None
for i, line in enumerate(lines):
    if line.startswith('configuration: 6000'):
        start_line = i
    elif line.startswith('configuration: 6050'):
        end_line = i
        break
data_lines = lines[start_line+1: end_line]

column2 = [float(line.split()[1]) for line in data_lines]
 
y     = np.array(column2)


plt.plot(x, y, color = 'green',linewidth = 0.8, label = 'Gaussian fl')


plt.xlabel('τ')
plt.ylabel('x')
plt.legend()
plt.title('Random instanton configuration vs Gaussian fluctuations')

plt.xlim(0, 20)

plt.show()
#------------------------------------------------------------------------------
#   FIG. 13.a Shows the correlation functions in a random instanton ensamble
#   with gaussian fluctuations
#------------------------------------------------------------------------------

with open('Data/qmdiag/dcor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]
column4 = [float(line.split()[3]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/rilm_gauss/rcor_gauss.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/rilm_gauss/rcor2_gauss.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='o', markerfacecolor='none',
             markeredgecolor = 'red',markersize=8, capsize=5, label = '<x²(0)x²(τ)>')

with open('Data/rilm_gauss/rcor3_gauss.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='D',markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x³(0)x³(τ)>')

plt.xlabel('τ')
plt.ylabel('<xⁿ(0)xⁿ(τ)>')
plt.legend()
plt.title('Ran. inst. gaussian fluctuations correlation functions')

plt.xlim(0, 1.5)
plt.ylim(0, 12)

plt.show()

#------------------------------------------------------------------------------
#   FIG. 13.b Shows the logarithmic derivative of the correlators
#   in a random instanton ensamble with gaussian fluctuations
#------------------------------------------------------------------------------
with open('Data/qmdiag/dcor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[4]) for line in lines]
column3 = [float(line.split()[5]) for line in lines]
column4 = [float(line.split()[6]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/rilm_gauss/rcor_gauss.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/rilm_gauss/rcor2_gauss.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='o', markerfacecolor='none',
             markeredgecolor = 'red',markersize=8, capsize=5, label = '<x²(0)x²(τ)>')

with open('Data/rilm_gauss/rcor3_gauss.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='D',markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x³(0)x³(τ)>')

plt.xlabel('τ')
plt.ylabel('d[log<xⁿ(0)xⁿ(τ)>]/dτ')
plt.legend()
plt.title('Ran. inst. gaussian fluctuations log derivative correlation functions')

plt.xlim(0, 1.5)
plt.ylim(0, 5)

plt.show()

#------------------------------------------------------------------------------
#   Fig.16 Distribution of instanton-anti-instanton separations
#------------------------------------------------------------------------------
with open('Data/rilm/zdist.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.plot(x, y, color = 'red')

with open('Data/iilm/zdist.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.plot(x, y, color = 'black', drawstyle = 'steps')

plt.xlabel('τ_z')
plt.ylabel('n_IA(τ_z)')
plt.title('Istanton-anti-istanton separation distribution')

plt.xlim(0,3.85)
plt.ylim(0,40000)
plt.show()

#------------------------------------------------------------------------------
#   Fig.17 Typical instanton configuration in an instanton calculation
#------------------------------------------------------------------------------
with open('Data/iilm/iconf.dat', 'r') as file:
    lines = file.readlines()[:3001]

for i in range(10):
    column  = [float(line.split()[i]) for line in lines]
    
    y = np.array(column)
    x = range(len(y))
    if i % 2 == 0:
        plt.plot(x, y, color = 'blue', linewidth = 0.8)
    else:
        plt.plot(x, y, color = 'red', linewidth = 0.8)

plt.xlabel('configurations')
plt.ylabel('x')
plt.title('Instanton configuration in an interacting instanton calculation')

plt.xlim(0, 3000)

plt.show()

#------------------------------------------------------------------------------
#   FIG. 18.a Shows the correlation functions interacting
#------------------------------------------------------------------------------

with open('Data/qmdiag/dcor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]
column4 = [float(line.split()[3]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/iilm/icor.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/iilm/icor2.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='o', markerfacecolor='none',
             markeredgecolor = 'red',markersize=8, capsize=5, label = '<x²(0)x²(τ)>')

with open('Data/iilm/icor3.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='D',markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x³(0)x³(τ)>')

plt.xlabel('τ')
plt.ylabel('<xⁿ(0)xⁿ(τ)>')
plt.legend()
plt.title('Interacting instatons correlation functions')

plt.xlim(0, 1.5)
plt.ylim(0, 12)

plt.show()

#------------------------------------------------------------------------------
#   FIG. 18.b Shows the logarithmic derivative of the correlators
#------------------------------------------------------------------------------
with open('Data/qmdiag/dcor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[4]) for line in lines]
column3 = [float(line.split()[5]) for line in lines]
column4 = [float(line.split()[6]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/iilm/icor.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/iilm/icor2.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='o', markerfacecolor='none',
             markeredgecolor = 'red',markersize=8, capsize=5, label = '<x²(0)x²(τ)>')

with open('Data/iilm/icor3.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='D',markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x³(0)x³(τ)>')

plt.xlabel('τ')
plt.ylabel('d[log<xⁿ(0)xⁿ(τ)>]/dτ')
plt.legend()
plt.title('Interacting instatons log derivative correlation functions')

plt.xlim(0, 1.5)
plt.ylim(0, 5)

plt.show()


