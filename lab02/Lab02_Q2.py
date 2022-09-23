#!/usr/bin/env python3

#=============================================================================#
#                                                                             #
# NAME:     Lab02_q2.py                                                       #
#                                                                             #
# PURPOSE: numerically integrates a 4/(1 +x^2)  from 0 to 1, for 4 slices and #
# then compares trapezoid simpson rule for this integrand                     #
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


''' #pseudpo code

# imports
# write function f(x)
  # calculates integrand

# integration solving functions, trap and simp
    # takes in arrays to be more generlized when you dont have a function
    # returns value of integral

# timing funcion / find big N/ little n
  # times how long ti takes to get to a set error threshold, and returns n

# if name = main
   # main program, run everything needed for Q2'''


# imports
import numpy as np
import matplotlib.pyplot as plt
from time import time


def f(x):
    '''integrand for Q2'''
    return 4/(1+x**2)


def trapzoid_rule(x_arr, y_arr,):
    '''Integrates using the trapezoid rule.

    Args:
    x_arr: array like of postions of xi

    y_arr: array like of the postions of yi

    Returns: value of integral from x[0] to x[-1]
    '''
    dx = np.abs(x_arr[1] - x_arr[0])
    n = len(x_arr)-1
    return (dx/2) * (y_arr[0] + 2* np.sum(y_arr[1:-1]) + y_arr[-1])


def simpson_rule(x_arr, y_arr):
    '''Integrates using the simpson rule.

    Args:
    x_arr: array like of postions of xi

    y_arr: array like of the postions of yi

    Returns: value of integral from x[0] to x[-1]
    '''
    dx = np.abs(x_arr[1] - x_arr[0])
    n = len(x_arr)
    y_arr_odd = y_arr[0:-1]
    y_arr_even = y_arr[0:-2]

    # index 0 is first
    odd = np.sum(y_arr_odd[1::2]) # sum over odd values
    even = np.sum(y_arr_even[0::2]) # sum over even values

    return (dx/3)* (y_arr[0] + 4*odd + 2*even + y_arr[-1])


# how many n until error is less then 10−9
def epsilon_n(func, error_max=1e-9):
    ''' finds the time needed to calculate, and both n,N such that the error of
    a given integrating method func is less then error_max using N=2**n

    Args:
    func: function
          integration method, needs to take in a x_arr, and a y_arr to integrate

    KArgs:
    error_max: float
               absolute error threashold

    Returns: none
            prints time, N, and little n along with func to screen'''
    little_n = 1
    tstart = time()
    error = 3 # dummy error value for initzilation
    while error >= error_max: # just need to think about what to set this to b

            N = 2**little_n
            x_arr = np.linspace(0, 1, N)
            intgeral= func(x_arr, f(x_arr),)
            error = np.abs(intgeral - np.pi)
            little_n += 1


            tstop = time()
    time_val = tstop-tstart
    N = N

    print(str(func))
    print('Time = ', time_val, '[s] N = ', N)
    print('n', little_n-1)
    return None

# main program
if __name__ == "__main__":

    # 4 slice integration of f(x) from 0 to 1
    n = int(5) # 5 points means 4 slices
    x_arr = np.linspace(0, 1, int(n))
    y_arr = f(x_arr)
    trap = trapzoid_rule(x_arr, y_arr,)
    simp = simpson_rule(x_arr, y_arr)
    print(trap, simp, np.pi) # comapre methods


    # compare N, n and time for the two methods
    epsilon_n(func=simpson_rule, error_max=1e-9)
    epsilon_n(func=trapzoid_rule, error_max=1e-9)
