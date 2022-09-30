#!/usr/bin/env python3

#=============================================================================#
#                                                                             #
# NAME:     Lab03_Q1.py                                                       #
#                                                                             #
#  PURPOSE:    calculates hermite polynomails, plots the                      #
#  first 4 stationary states of the quantum harmoic oscilator,                #
#  then plots the n=30 stationary state.Then finds <p^2>, <x^2> plots         #
#  energy and uncertainties      for n=0,1,2...15                             #
#=============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2022 Maxwell A. Fine                                          #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
#=============================================================================#

'''psuedo code

imports
numpy, plt 2 functions from handout

for q3_a, and b we need
hermite poly nomial function
* makes a hermite poly nomial given n, and for points x
* use recursive relationship from handout
* add if statements for n=0, n=1

psi
* makes stationary states of the quantum harmonic oscilator with all physical
constatns equal to 1

for q3c, we need

psi_z same as psi but with tan(z) = x substition

psi_prime:
* derivative of psi!, uses recursive relationship from hadnout

x_2 integrand:
calculates x_2 integrand for z basis

p_2 integrand for z basis

quad
* integrates with quad!
* need to adjust a,b and weighting scheme to match (done with handout functions)

if name = main
* make hermite poly plot
* make Q1A, B plots
* make Q1C plots
'''


# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from gaussxw import * # needs to be in the same dir, lab handout file


def hermite(n,x):
    '''calculates hermite polynomials

    Args:
    n: int >= 0
       degree of hermite poly to construct

    x: float or arr
       points to calculate hermite poly at

    Returns: arr or float of hermite poly calculated at x'''
    # if x is a single value return a single,
    # if array like should be ones of size of the array
    try:
        h0 = np.ones(len(x))
    except:
        h0 = 1
    h1 = 2*x

    # if statements for n=0,n=1
    if n == 0:
        return h0
    if n == 1:
        return h1
    # recursive relationship
    for i in range(1,n+1):
        hn = 2*x*h1 - 2*i*h0
        if i == n-1:
            return hn
        # update values
        h0 = h1
        h1 = hn


def psi(n,x):
    '''calculates the stationary state psi(x) for given x and any integer n ≥ 0
    for Quantum haromic oscilator, physical constants = 1 for this.

    Args:
    n: int >= 0
       the nth stationary state to produce

    x: arr or float
       positons to calculate for

    Returns: arr or float psi(x) of the nth stationary state '''

    psi = (1/ np.sqrt(2**n *factorial(n) * \
                      np.sqrt(np.pi)))*np.exp(-0.5*x**2)

    h_nx = hermite(n,x)

    return h_nx * psi


# q3c functions
# functions for psi, psi_prime, x_2_integrand, p_2_integrand, quad
# for the substition of tan(z) = x
def quad(f, N, a, b, n):
    '''Integration by Guassian quad, integrates f from a to b using N points.
    Note - this is pretty much given in the textbook.
    physical constants = 1 for this.

    Args:
    f: function
       function to integrate, note that f needs to be of the form f(n,x)

    N: int
       number of points to use for quad integral

    a: float
       lower boundry of integral

    b: float
       upper boundry of integral

    n: int >= 0
       argument for f

    Returns: float
             value of integral
        '''
    x,w = gaussxwab(N,a,b)
    s = 0.0
    for k in range(N):
        s += w[k]*f(n,x[k])
    return s


def psi_z(n,z):
    '''calculates the stationary state psi(z) for given z and any integer n ≥ 0
    for Quantum haromic oscilator, physical constants = 1 for this.

    Note - This is a
    redfined map, with tan(z) = x subsition from the psi function.

    Args:
    n: int >= 0
       the nth stationary state to produce

    z: arr or float
       positons to calculate for, domain is - pi /2 to pi /2

    Returns: arr or float
            psi(z) of the nth stationary state '''

    psi = (1/ np.sqrt(2**n *factorial(n) * \
                      np.sqrt(np.pi)))*np.exp(-0.5*np.tan(z)**2)

    return psi*hermite(n,np.tan(z))


def x2_integrand(n,z):
    '''Calculates the integrand for <x^2> calculation of the nth stationary
    state of the quantum harmonic oscilator

    Args:
    n: int >= 0
       the nth stationary state to produce  for

    z: arr or float
       positons to calculate for, domain is - pi /2 to pi /2

    Returns: arr or float
             the integrand of <x^2> for psi(z) of the nth stationary state
    '''

    return (np.tan(z)**2/np.cos(z)**2)*np.abs(psi_z(n,z))**2


def psi_prime(n,z):
    ''' Calculates the the dervative of psi forthe nth stationary
    state of the quantum harmonic oscilator

    Args:
    n: int >= 0
       the nth stationary state to produce  for

    z: arr or float
       positons to calculate for, domain is - pi /2 to pi /2

    Returns: float or array
             psi(z) prime
    '''
    psi = (1/ np.sqrt(2**n *factorial(n) * \
                      np.sqrt(np.pi)))*np.exp(-0.5*np.tan(z)**2)

    return psi*(-np.tan(z) * hermite(n,np.tan(z)) + 2*n*hermite(n-1,np.tan(z)))


def p2_integrand(n,z):
    '''Calculates the integrand for <p^2> calculation of the nth stationary
    state of the quantum harmonic oscilator

    Args:
    n: int >= 0
       the nth stationary state to produce  for

    z: arr or float
       positons to calculate for, domain is - pi /2 to pi /2

    Returns: arr or float
             the integrand of <p^2> for psi(z) of the nth stationary state
    '''

    # for n==0, the prime breaks, so we must do it another way
    if n==0:

        return x2_integrand(n,z)
        # this trick works based on our answer to Q3C

    else:
        return (1/np.cos(z)**2)* np.abs(psi_prime(n,z))**2

# main program
if __name__ == "__main__":
    fs = 15

# q3_a
# plots hermite poly nomials to confirm our function, and plots first 4
# stationary states

# plots hermite polynomial n= 0,1,2,3,4
    x = np.linspace(-4,4,50)

    # hard coded polys
    h0 = np.ones(len(x))
    h1 = 2*x
    h2 = 4*x**2 -2
    h3 = 8*x**3 - 12*x
    h4 = 16*x**4 -48*x**2 + 12

    # plot hard coded
    plt.scatter(x,h0)
    plt.scatter(x,h1)
    plt.scatter(x,h2)
    plt.scatter(x,h3)
    plt.scatter(x,h4)
    plt.ylim(-30,30)
    n = 4

    # plot function version
    for i in range(n+1):
        yi  = hermite(i,x)
        plt.plot(x,yi, label= 'H'+ str(i))
    plt.legend()
    plt.ylim(-30,30)
    plt.title('Hermite Polynomials', fontsize=fs)
    plt.xlabel('x', fontsize=fs)
    plt.ylabel('y',fontsize=fs)
    plt.savefig('q3_a2.pdf', bbox_inches='tight') # saves plot
    plt.close() # closes plot to save mem

# plot n=0,1,2,3 stationary states
    n = 3
    for i in range(n+1):
        yi = psi(i,x)
        plt.plot(x, np.abs(yi**2), linestyle='--',
                 label ='$\psi_'+str(i)+'(x)$' , )
    plt.xlim(-4,4) # set x lim
    plt.ylabel('$|\psi(x)|^2$', fontsize=fs)
    plt.xlabel(r'x',fontsize=fs)
    plt.legend(fontsize=12, bbox_to_anchor=(1,1))
    plt.title('Quantum Harmonic Oscillator \n Stationary States', fontsize =12)
    plt.savefig('q3_a.pdf', bbox_inches='tight') # saves plot
    plt.close() # closes plot to save mem


# Q3B
# plot n=30 stationary state
    n = 30
    x = np.linspace(-10,10, 500) # new x limits
    yi = psi(n,x)
    plt.plot(x, np.abs(yi**2), linestyle='--', label ='$H_{'+str(n)+'}(x)$')
    plt.xlim(-10,10) # xlim
    plt.ylabel('$|\psi(x)|^2$', fontsize=fs)
    plt.xlabel(r'x',fontsize=fs)
    plt.title('$N=30$ Stationary Sate', fontsize=fs)
    plt.savefig('q3_b.pdf', bbox_inches='tight') # saves plot
    plt.close() # closes plot to save mem


# Q3C
# find <p^2>, <x^2>, make plots of energy
    z = np.linspace(-np.pi/2 , np.pi/2, 500) # init z
    N = 100 # polints for quad
    a = z[0] # integration bounds
    b = z[-1]
    n = 15 # do n=0, to n=15 stationary states
    n_arr = np.arange(0,n+1,1)
    x_2 = np.empty(n+1) # init results
    p_2 = np.empty(n+1)
    x = np.empty(n+1)
    p = np.empty(n+1)
    e = np.empty(n+1)
    for i in range(len(n_arr)):
        n = n_arr[i]
        x_2[n] = quad(f=x2_integrand, N=N,a=a,b=b, n=n)
        p_2[n] = quad(f=p2_integrand, N=N,a=a,b=b, n=n)
    x = np.sqrt(x_2) # calculate values
    p = np.sqrt(p_2)
    e = 0.5 * ( x_2 + p_2)

    # make plots
    # energy vs n plot
    plt.scatter(n_arr, e,color='red')
    fs=14
    plt.title('Energy for the nth Stationary State', fontsize=fs)
    plt.xlabel('Stationary State', fontsize=fs)
    plt.ylabel('Energy (Dimensionless)', fontsize=fs)
    plt.savefig('q3_c.pdf', bbox_inches='tight') # saves plot
    plt.close() # closes plot to save mem

    # Uncertainty Principle plot
    plt.scatter(n_arr, x*p)
    plt.xlabel('Stationary State', fontsize=fs)
    plt.ylabel('$\sigma_x \sigma_p$', fontsize=fs)
    plt.title('Uncertainty Principle', fontsize=fs)
    plt.text(x=8, y= 4,
    s='Min of $\sigma_x \sigma_p$ = '+ str(np.min(x*p))[0:5], fontsize=fs-2)
    plt.savefig('q3_c2.pdf', bbox_inches='tight') # saves plot
    plt.close() # closes plot to save mem
