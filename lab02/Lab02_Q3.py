#!/usr/bin/env python3

#=============================================================================#
#                                                                             #
# NAME:     Lab02_q3.py                                                       #
#                                                                             #
# PURPOSE: includes a function to numerically solve for the energy emiited    #
# per area of a black body of a temperature T (in k), and estimates           #
# stephan-boltzman constant                                                   #
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
'''# pseudo code for Q3B

# import scipy, numpy
# import constants from scipy (in SI units)

# write a function to calculate integrand (using the new integrand IE x)
    # Uses given T (in kelvin) to calculate a C_1 value
    # then returns value of integrand

# numerically integrate using scipy quad
    # from 0 to np.inf
    # answer and accuracy estimate is included in quad
    # print values

'''


# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy as sp
c = sp.constants.c # speed of light in SI
h = sp.constants.h # plank constant in SI
k = sp.constants.k # boltzmann constant in SI
sigma = sp.constants.Stefan_Boltzmann # in SI


def black_body_integrand(x, T):
    ''' Calculates the integrand of blackbody radiation curve.
    IE this returns energy emitted as a function of x. x is related to
    wavenumber.

    Args:
    x : float same Units as [hv/kT]
        position to calculate for, x can range from 0 to +infty, and is equal
        to x = hv/kT

    T: float, kelvin
       temperature to calculate for

    Returns:
    value of integrant, float

    '''

    C_1 = 2 *np.pi* k**4 * T**4/ (h**3 *c **2)

    return C_1 * x**3 / ( np.exp(x)-1)


# main program
if __name__ == "__main__":
    T = 500 #kelvin this can be set to any positive value
    # coompute integral and estimate error
    W = sp.integrate.quad(black_body_integrand, a=0,
                     b=np.inf, args=(T))

    emp_sigma = W[0]*T**-4 # our value of simga
    abs_error = W[1] # absolute error of W
    error_in_sigma = (W[1]) * T**-4 # error in sigma, standard error propgation
    magnitude_err = np.abs(np.floor(np.asarray(np.log10(error_in_sigma))))
    # ^ order of magnitude of error

    # round based on error
    error_in_sigma = round(error_in_sigma, np.abs(int(magnitude_err)))
    # ^ rounded to first sig fig
    emp_sigma = round(emp_sigma, int(magnitude_err))


    print('Stefan-Boltzmann constant calculatd = ', emp_sigma,
    u"\u00B1", error_in_sigma, '[W/(m^2 K^4)]')
    print('Stefan-Boltzmann constant from scipy.constants = ',sigma,
     ' [W/(m^2 K^4)]')
