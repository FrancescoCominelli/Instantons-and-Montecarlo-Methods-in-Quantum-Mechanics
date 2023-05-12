import numpy as np
import matplotlib.pyplot as plt

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
#   FIG. 2: Typical euclidean path obtained in a Monte Carlo simulation of the 
#   discretized euclidean action of the double well potential for  = 1.4.
#------------------------------------------------------------------------------

with open('Data/qm/config.dat', 'r') as file:
    lines = file.readlines()[2693765:2694565]

column1 = []
column2 = []

for line in lines:
    if "configuration: " in line:
        continue
    # split the line and append the values to the corresponding lists
    values = line.split()
    column1.append(float(values[0]))
    column2.append(float(values[1]))



x     = np.array(column1)
y     = np.array(column2)


plt.plot(x, y, color = 'black',linewidth = 0.8, label = 'Monte Carlo')

plt.xlim(0, 20)
plt.xlabel('τ')
plt.ylabel('x')

plt.show()
