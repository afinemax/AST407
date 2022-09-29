#!/usr/bin/env python3

#=============================================================================#
#                                                                             #
# NAME:     Lab03_Q1.py                                                       #
#                                                                             #
# PURPOSE: Implements a central and forward numerical differenation scheme,   #
# accuracy of both schemes and prints  the error compares to the step size for#
# the forward scheme                                                          #
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

'''
# psuedo code

# IDEA:  write a function for forward, and central and use them to calculate
then print error, and make plots

# imports
numpy , plt

# functions

f(x)
* test function

forward (f,x,h)
* forward diff scheme
return f'

central_dif(f,x,h)
* central diff scheme
return f'

analytic_f'
return f'

if_name=__main__
# set h, x
# calculate derivatives and error
# print error, h for forward (for writeup)

# make 2 scatter plots

'''

# imports
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    '''function f to test numerical methods on, f(x) = e^(-x^2)'''
    return np.exp(-x**2)

def for_dif(f, x, h):
    '''Forward numerical differenation schemes, returns first dervative of f

    Args:
    f: function
       function to calculate derivative on

    x: float or array
       location to calculate the derivative

    h: float or array
       step size to use in order to calculate derivative

    Returns:
    float or array of the dervative at x for stepsize h
    '''

    return (f(x + h) - f(x) )/ h


def analytic_df(x):
    'Calculats the analytical derivative of f at x'
    return -2*x*np.exp(-x**2)


def cet_dif(f, x, h):
    '''central numerical differenation schemes, returns first dervative of f

    Args:
    f: function
       function to calculate derivative on

    x: float or array
       location to calculate the derivative

    h: float or array
       step size to use in order to calculate derivative

    Returns:
    float or array of the dervative at x for stepsize h
    '''

    return (f(x + h) - f(x-h) ) / (2*h)


# main program
if __name__ == "__main__":

    # initalize h, and x
    h_q1 = np.logspace(-16,0, 17)
    x = 0.5

    # calculate forward diff, analytical diff, and error for forwrd
    df_q1 = for_dif(f, x, h_q1)
    df_true_q1 = analytic_df(x)
    error = np.abs(df_true_q1-df_q1)

    # calculate central diff,  and error for central
    df_cent = cet_dif(f, x, h_q1)
    error_cent = np.abs(df_true_q1-df_cent)

    # print error for forward and step size
    for i in range(len(error)):
        print(str(error[i]), h_q1[i])

    # plot forward vs error
    plt.scatter(h_q1, error, color='green', label='Foward Difference')
    plt.yscale('log')
    plt.ylabel('Error',fontsize = 15)
    plt.xlabel('Step Size', fontsize=15)
    plt.xscale('log')
    plt.xticks(fontsize=12,)
    plt.yticks(fontsize=12,)

    plt.title('Forward difference', fontsize=13)
    plt.savefig('q1_b.pdf', bbox_inches='tight') # saves plot
    plt.close() # closes figure to keep memory use low

    # plot forward vs central
    plt.scatter(h_q1, error, color='green', label='Foward Difference')
    plt.scatter(h_q1, error_cent, color='k', label='Central')
    plt.yscale('log')
    plt.ylabel('Error', fontsize=15)
    plt.xscale('log')
    plt.xlabel('Step Size', fontsize=12)
    plt.title('Forward vs Central Difference', fontsize=12)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12,)
    plt.yticks(fontsize=12,)
    plt.savefig('q1_c.pdf', bbox_inches='tight') # saves plot
    plt.close() # closes figure to keep memory use low
