#!/usr/bin/env python3

#=============================================================================#
#                                                                             #
# NAME:     q2.py                                                             #
#                                                                             #
# PURPOSE: Calculates & plots the orbit of Earth, taking into account         #
#          the effects of Jupiter on earth for 10 years, again but for bit    #
#           over  years, and 1000x jupiter mass. Then calculates the effect   #
#           of jupiter on an asteriod's orbit                                 #
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

# imports
import numpy as np
import matplotlib.pyplot as plt
from q1 import gr_2b_g_acc # computes accerlation due to Fg
from q1 import fg_solver as fg_2b_solver # computes orbital mechanics for
                                        #  2B problem
from q1 import plot_orbits
# set constants
M_j =  1e-3 #solar masses


# functions
def fg_3b_solver(earth_i, jupiter_dyn, M_j, dt=0.0001, int_time=1.,):
    '''Numerically solves the 3 body problem using Netonion gravity.
    For the case of a earth-like body orbiting the sun which is effected by
    a jupiter-like body (which orbits the sun uneffected by earth)
    Uses he “Euler-Cromer” method. x=y=0 is the location of the sun.

     Args:

     earth_i: list, float (in AU, and AU/yr)
              4 floats corrisponding to inital conidctions for earth-like body
              IE earth_i = [x, y, vx, vy]

     jupiter_dyn: list
                  list of jupiters-like body orbital dynamics
                  IE jupiter_dyn = [x_arr, y_arr, vx_arr, vy_arr]

    **Args:
    dt: float, Earth in years
              dt to use for numerical solution, default is 0.0001

    int_time: float, in Earth years
              integration time for solution, IE how long to solve for

    Returns:
    x_arr, y_arr, vx_arr, vy_arr

    i_arr: array-like, float
           array of the ith direction of Earth

    vi_arr: array-like, float
            array of the ith direction's velocity of Earth

    '''

    # initalize values
    n_steps = int(int_time/dt) # how many dts between 0 and end
    x_arr = np.empty((n_steps),dtype=float)
    y_arr = np.empty((n_steps),dtype=float)
    vx_arr = np.empty((n_steps),dtype=float)
    vy_arr = np.empty((n_steps),dtype=float)

    # earth to jupiter distance
    r_ejx = np.empty((n_steps),dtype=float)
    r_ejy = np.empty((n_steps),dtype=float)

    # 3 sets of accerlation, one from jupiter and one from the sun, and net
    ax_acc_sun = np.empty((n_steps),dtype=float)
    ay_acc_sun = np.empty((n_steps),dtype=float)

    ax_acc_j = np.empty((n_steps),dtype=float)
    ay_acc_j = np.empty((n_steps),dtype=float)

    ax_acc_net = np.empty((n_steps),dtype=float)
    ay_acc_net = np.empty((n_steps),dtype=float)

    # read in jupiter dynamics, and  set earth intial conditons
    Jx_arr, Jy_arr, Jvx_arr, Jvy_arr = jupiter_dyn
    xi, yi, vxi, vyi = earth_i

    # apply inital conditions
    dt = dt
    x_arr[0] = xi # AU
    y_arr[0] = yi # AU
    vx_arr[0] = vxi # AU/ Year
    vy_arr[0] = vyi # AU/ Year

    # numeric integration
    for i in range(1, n_steps):

        # Calculate acceleration from sun
        ax_acc_sun[i], ay_acc_sun[i] = gr_2b_g_acc(x_arr[i-1], y_arr[i-1],
                                           m_sun=1)

        # Calculate acceleration from jupiter
        # this is acceleration towards jupiter, not towards the sun!
        r = np.sqrt(x_arr[i-1]**2 + y_arr[i-1]**2)
        rejx = Jx_arr[i-1] - x_arr[i-1]
        rejy = Jy_arr[i-1] - y_arr[i-1]
        ax_acc_j[i], ay_acc_j[i] = gr_2b_g_acc(rejx, rejy, m_sun=M_j)

        # convert back into refrence to sun coordinats
        # acc_j should point 'toward' jupiter
        if Jx_arr[i-1] > x_arr[i-1]:
            ax_acc_j[i] = np.abs(ax_acc_j[i])
        else:
            ax_acc_j[i] = -1* np.abs(ax_acc_j[i])


        if Jy_arr[i-1] > y_arr[i-1]:
            ay_acc_j[i] = np.abs(ay_acc_j[i])
        else:
            ay_acc_j[i] = -1* np.abs(ay_acc_j[i])

        # Calculate net acc
        ax_acc_net[i] = ax_acc_sun[i] + ax_acc_j[i]
        ay_acc_net[i] = ay_acc_sun[i] + ay_acc_j[i]

        # Calculate velocity
        vx_arr[i] = vx_arr[i-1] + ax_acc_net[i]*dt
        vy_arr[i] = vy_arr[i-1] + ay_acc_net[i]*dt

        # Calculate position
        x_arr[i] = x_arr[i-1] + vx_arr[i]*dt
        y_arr[i] = y_arr[i-1] + vy_arr[i]*dt

    return x_arr, y_arr, vx_arr, vy_arr



# main program
if __name__ == "__main__":

    # inital conidtions for jupiter
    xJi = 5.2 # AU
    yJi = 0.0 # AU
    vxJi = 0.0 # AU/yr
    vyJi = 2.63 # AU/yr

    # inital conditions for Earth
    xEi = 1.0 # AU
    yEi = 0.0 # AU
    vExi = 0.0 # AU
    vEyi = 6.18 # AU/yr

    # inital conditions for asteriod
    xa = 3.3 # AU,
    ya = 0.0 # AU
    vax = 0.0 # AU/yr,
    vay = 3.46 # AU/yr

    # define simulation parmeters for first sim
    dt = 0.0001 # Earth years
    int_time = 10 # Earth years

    # run simulation
    jupiter_dyn = fg_2b_solver(xJi, yJi, vxJi, vyJi, dt=dt, int_time=int_time,)
    earth_i = [xEi, yEi, vExi, vEyi,]
    ast_i = [xa, ya, vax, vay]
    earth_dyn = fg_3b_solver(earth_i, jupiter_dyn, M_j, dt=0.0001,)
    e_x, e_y, e_vx, e_vy = earth_dyn
    Jx_arr, Jy_arr, Jvx_arr, Jvy_arr = jupiter_dyn

    # plot
    x_arrs = [e_x, Jx_arr]
    y_arrs = [e_y, Jy_arr]
    labels = ['Earth', 'Jupiter']
    colors = ['blue', 'red']
    line_styles = ['-', '--']
    title='Earth & Jupiter Orbit'
    save_name='q2_earth.pdf'
    plot_orbits(x_arrs, y_arrs, labels, colors, line_styles, title,
                    int_time, dt, save_name=save_name,
                    plot_sun=True, label_fs=15,legend_fs=15)

    # new simulation, update parameters
    int_time = 3.58 # Earth Yr
    dt = 0.0001 # Earth years
    jupiter_dyn = fg_2b_solver(xJi, yJi, vxJi, vyJi, dt=dt, int_time=int_time,)
    earth_dyn = fg_3b_solver(earth_i, jupiter_dyn, 1000*M_j, dt=dt,
                             int_time=int_time,)
    e_x, e_y, e_vx, e_vy = earth_dyn
    Jx_arr, Jy_arr, Jvx_arr, Jvy_arr = jupiter_dyn

    # plot
    x_arrs = [e_x, Jx_arr]
    y_arrs = [e_y, Jy_arr]
    labels = ['Earth', 'Jupiter']
    colors = ['blue', 'red']
    line_styles = ['-', '--']
    title='Earth Orbit With Exaggerated Jupiter Mass'
    save_name='q2_earth_mj.pdf'
    plot_orbits(x_arrs, y_arrs, labels, colors, line_styles, title,
                    int_time, dt, save_name=save_name,
                    plot_sun=True, label_fs=15,legend_fs=15)


    # new simulation, update parameters
    int_time = 20 # Earth years
    dt = 0.0001 # Earth years
    jupiter_dyn = fg_2b_solver(xJi, yJi, vxJi, vyJi, dt=dt, int_time=int_time,)
    ast_dyn = fg_3b_solver(ast_i, jupiter_dyn, M_j,dt=dt, int_time=int_time,)
    a_x, a_y, a_vx, a_vy = ast_dyn
    Jx_arr, Jy_arr, Jvx_arr, Jvy_arr = jupiter_dyn

    # plot
    x_arrs = [a_x, Jx_arr]
    y_arrs = [a_y, Jy_arr]
    labels = ['Asteroid', 'Jupiter']
    colors = ['grey', 'red']
    line_styles = [':', '--']
    title='Asteroid Orbit'
    save_name='q2_ast.pdf'
    plot_orbits(x_arrs, y_arrs, labels, colors, line_styles, title,
                    int_time, dt, save_name=save_name,
                    plot_sun=True, label_fs=15,legend_fs=15)
