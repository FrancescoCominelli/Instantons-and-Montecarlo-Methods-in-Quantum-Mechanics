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
#   FIG. 2: Typical euclidean path obtained in a Monte Carlo simulation of the 
#   discretized euclidean action of the double well potential for  = 1.4.
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

plt.xlim(0, 20)
plt.xlabel('τ')
plt.ylabel('x')
plt.legend()

plt.show()

#------------------------------------------------------------------------------
#   FIG. 3: Probability distribution in the double well potential for f = 1.4.
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

plt.legend()
plt.xlabel('x')
plt.ylabel('P(x)')
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

plt.xlim(0, 1.5)
plt.ylim(0, 8)
plt.legend()
plt.xlabel('τ')
plt.ylabel('<xⁿ(0)xⁿ(τ)>')
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

plt.xlim(0, 1.5)
plt.ylim(0, 4.5)
plt.legend()
plt.xlabel('τ')
plt.ylabel('d[log<xⁿ(0)xⁿ(τ)>]/dτ')

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
plt.xscale('log')  # Set x-axis scale to logarithmic
plt.xlim(0.01,2.5)
plt.ylim(-2.5, -1)
plt.xlabel('T')
plt.ylabel('F')

#data from qmswitch.py runned 6 times with different n
y     = np.array([-2.24413, -2.23835, -2.22145, -2.10050, -1.94522,       -1.74119])
y_err = np.array([ 0.02725,  0.05859,  0.03459,  0.05179,  0.03288,        0.02147])
x     = np.array([ 0.2,      0.025,    0.1,      0.5,      0.740740740741, 1.0])
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue',markersize=8, capsize=5)

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

plt.xlim(0, 1.5)
plt.ylim(0, 8)
plt.legend()
plt.xlabel('τ')
plt.ylabel('<xⁿ(0)xⁿ(τ)>')
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

plt.xlim(0, 1.5)
plt.ylim(0, 4.5)
plt.legend()
plt.xlabel('τ')
plt.ylabel('d[log<xⁿ(0)xⁿ(τ)>]/dτ')

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

plt.xscale('log')
plt.yscale('log')
plt.xlim(1, )

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
plt.xlim(0, n*a-a)

plt.show()











