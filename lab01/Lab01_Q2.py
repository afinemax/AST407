#!/usr/bin/env python3

#=============================================================================#
#                                                                             #
# NAME:     Lab01_q2.py                                                       #
#                                                                             #
# PURPOSE: Calculates & plots the orbit of Earth, taking into account         #
#          the effects of Jupiter on earth for 10 years, again but for        #
#          3.58 yrs, and 1000x jupiter mass. Then calculates the effect       #
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


'''

# Pseudocode
# general idea
# we can use functions written to solve (and plot!) the 2body problem to solve
for the orbit of jupiter
# using the orbit of jupiter, we can build a modified 2body solver function that
takes into account earth being pulled
# by the sun, and jupiter
# in if name = main, run the sets of simnulations and plots

# Imports and set constants
# IE numpy, matplotlib, jupiter mass in solar masses


# function to calculate the acceleration in a 2-body orbit
  # this is really an imported function from Q1, however it is placed in another
  file
  # takes in mass of sun-like body and x,y position relative to the sun (at 0,0)
  # sun mass has option to be set to a different value
  # returns the acceleration, calculate in AU, Year, solar mass units


# function to calculate 2-Body orbit around the sun
    # this is really an imported function from Q1, however it is placed in
    another file
    # this is really an imported function
    # takes in initial conditions xi,yi, vxi, vyi allow to change the mass of
    sun
    # takes in timestep, and integration time in earth years
    # calculate in AU, Year, solar mass units
    # use Euler-Cromer method of intergration
        # for i in time (n_time steps)
            # Calculate acceleration
            ax_arr[i], ay_arr[i] = calculate_accerlation_2b_function()
             # Calculate velocity
            vx_arr[i] = vx_arr[i-1] + ax_arr[i]*dt
            vy_arr[i] = vy_arr[i-1] + ay_arr[i]*dt
            # Calculate position
            x_arr[i] = x_arr[i-1] + vx_arr[i]*dt
            y_arr[i] = y_arr[i-1] + vy_arr[i]*dt

    # returns x, y, vx, vy arrays of the orbit


# orbit plotting function
  # this is really an imported function from Q1, however it is placed in
  another file
  # plots the orbits around the sun nicely
  # takes in n-orbit information, with associated labels and plotting parms
  # uses matplotlib.pyplot to plot a elegant plot
  # displays the sun, dt, integration time and has labels
  # dots to represent 'final' positions


# function to calculate approximated 3-body problem
  # takes in larger planets orbit (x, y, vx, vy) as calculated by 2b solver
  function
  # takes in smaller planet initial conditions
  # calculates orbit of smaller planet around the sun taking into account the
  pull from the larger planet
    # uses Euler-Cromer method just like the 2-Body solver function
    # difference is it calculates acceleration from the larger planet,
    adjusts sign to ensure acceleration points
    # points towards the larger planet (using an if statement?)
    # net acceleration is the effect of the sun + the effect of the larger
    planet on the smaller one
    # calculates other variables now just like 2-body solver but uses
    the net acceleration

  # returns x, y, vx, vy arrays of the orbit of smaller planet


# main program
# if name = main
# define the initial values for parts A, B, C and then run each set of simulations, and plot

# define inputs for A
# run functions to produce orbits for planets
# plot

# define inputs for B
# run functions to produce orbits for planet
# plot

# define inputs for c
# run functions to produce orbits for planet
# plot

'''

# imports

import numpy as np
import matplotlib.pyplot as plt
# these are now self contained as Max's lab01_q1.py was not submitted
# from lab01_Q1.py import plot_orbits
#from lab01_Q1 import gr_2b_g_acc # computes accerlation due to Fg
#from lab01_Q1 import fg_solver as fg_2b_solver # computes orbital mechanics for

# set constants
M_j =  1e-3 #solar masses
G_AU = 39.5 # for AU, solar mass and year system
m_sun = 2.0*1e30 #kg
G = 6.67*1e-11 # SI units
AU = 1.496*1e11 #m
m__merc = 3.285*10**23 # kg
year = 365*24*60*60 # seconds in a year


# functions
def gr_2b_g_acc(x, y, G=39.5, alpha=0, m_sun=1):
    '''Calculates the x, and y componets of GR corrected Newtonion gravitional
    accerlation around the sun. Helocentric model, IE x=y=0 is the location of
    the sun.

    Args:
    x: float, in AU
       positioin in the x direction

    y: float, in AU
       position in the y direction

    **KArgs
    G: float,
       G constant to use, default is in AU, Solar Mass, Earth Year system

    alpha: float, in AU^2
           value of alpha to use in correction for GR, defAUlt is 0

    m_sun: float, in solar mass units
           Mass of the sun / central body

    Returns:
    ax, ay: 2 floats
            x, and y componets of the accerlation in AU/yr^2

    '''

    # Calculate magnitude of r
    r = np.sqrt(x**2 + y**2)

    # Calculate force componets
    ax = -1*((G*m_sun) / (r**3)) * ( 1 + (alpha/r**2)) * (x)
    ay = -1*((G*m_sun) / (r**3)) * ( 1 + (alpha/r**2)) * (y)

    return ax, ay


def fg_2b_solver(xi, yi, vxi, vyi, dt=0.0001, int_time=1., alpha=0):
    '''Numerically solves the 2 body problem using the GR correction
    for Newtonion gravity, alpha=0 corresponds to Newtonion model.
    Helocentric model, IE x=y=0 is the location of the sun.
    Uses the The “Euler-Cromer” method.

     Args:
     xi: float, in AU
         inital positon in the x direction

     yi: float, in AU
         inital positon in the y direction

     vxi: float, in AU/year
          x componet of the inital velocilty

     vyi: float, in AU/year
          y componet of the inital velocity


    **Args:
    dt: float, Earth in years
              dt to use for numerical solution, default is 0.0001

    int_time: float, in Earth years
              Integration time for solution, IE how long to solve for

    alpha: float, in AU^2
           value of alpha to use in correction for GR, defaullt is 0


    Returns:
    x_arr, y_arr, vx_arr, vy_arr

    i_arr: array-like, float
           array of the ith direction

    vi_arr: array-like, float
            array of the ith direction's velocity

    '''

    # initalize values
    n_steps = int(int_time/dt) # how many dts between 0 and end
    x_arr = np.empty((n_steps),dtype=float)
    y_arr = np.empty((n_steps),dtype=float)
    vx_arr = np.empty((n_steps),dtype=float)
    vy_arr = np.empty((n_steps),dtype=float)
    ax_arr = np.empty((n_steps),dtype=float)
    ay_arr = np.empty((n_steps),dtype=float)

    # apply inital conditions
    dt = dt
    x_arr[0] = xi # AU
    y_arr[0] = yi # AU
    vx_arr[0] = vxi # AU/ Year
    vy_arr[0] = vyi # AU/ Year

    # numeric Integration
    for i in range(1, n_steps):

        # Calculate acceleration
        ax_arr[i], ay_arr[i] = gr_2b_g_acc(x_arr[i-1], y_arr[i-1],
                                           alpha=alpha, m_sun=1)

        # Calculate velocity
        vx_arr[i] = vx_arr[i-1] + dt*ax_arr[i]
        vy_arr[i] = vy_arr[i-1] + dt*ay_arr[i]
        # Calculate position
        x_arr[i] = x_arr[i-1] + dt*vx_arr[i]
        y_arr[i] = y_arr[i-1] + dt*vy_arr[i]

    return x_arr, y_arr, vx_arr, vy_arr


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



def plot_orbits(x_arrs, y_arrs, labels, colors, line_styles, title,
                int_time, dt, save_name='youforgottoset_save_name.pdf',
                plot_sun=True, label_fs=15,legend_fs=15, txt_fs=11.5):
    '''Plots orbits of n planets orbits,
       optionally plots the sun. Uses matplotlib.pyplot. Saves output.

       Args:
       x_arrs: list of arrays (arrays in AU)
               list of x_arr values corresponding to x componet of orbit

       y_arrs: list of arrays (arrays in AU)
               list of y_arr values corresponding to x componet of orbit

       labels: list of strings
               list containing the label for the plot legend corresponding to
               the same index orbit

      colors: list of strings
              list containing the color (str) for the plot corresponding to
              the same index orbit

      color_int: str
                 what color to plot the starting positon

      line_styles: list of strings
                   list containing the linestyle for the plot corresponding to
                   the same index orbit

      title: str
             title of plot

      int_time: float, in Earth years
                Integration time for solution, IE how long the plot is for


      dt: float, Earth in years
                dt to use for plotting, default is 0.0001

      **KArgs:
      save_name: str
                 argument for plt.savefig()
                 default is 'youforgottoset_save_name.pdf

      plot_sun: bool
                if True, plots the sun at the center of the figure

      label_fs: float
                fontsize to use for plot (exluding legend), default is 15

      legend_fs: float
                 fontsize to use for legend, default is 15

      txt_fs: float
              fontsize to use for plt.text(), default is 11.5

      Returns:
      None, plot is saved in the same dir as where the program is stored

     '''
    # initalize figure
    fig = plt.figure()#figsize=(6, 6), dpi=80)
    ax = fig.add_axes([0, 0, 1, 1])

    # plot sun
    if plot_sun == True:
        plt.scatter(0,0,marker ='o',color ='gold',label = 'Sun')


    # plot planets
    for i in range(len(x_arrs)):
        # get info for the ith planet
        x_arr = x_arrs[i]
        y_arr = y_arrs[i]
        label = labels[i]
        color = colors[i]
        linestyle = line_styles[i]

        # plot orbits
        plt.plot(x_arr, y_arr, color=color, label=label, linestyle=linestyle)

        # plot ending postion  condition
        plt.scatter(x_arr[-1], y_arr[-1], color=color,)

    # adjust plot parameters and add labels
    plt.xlabel("x (AU)", fontsize=label_fs)
    plt.ylabel("y (AU)",fontsize=label_fs)
    plt.axis('square')
    plt.xticks(fontsize=label_fs)
    plt.yticks(fontsize=label_fs)
    plt.legend(fontsize=legend_fs,
               bbox_to_anchor=(1., .90), fancybox=True,
               title_fontsize=legend_fs, loc='center left', )
    plt.title(title, fontsize=15, )
    # 0.61, 0.94
    #
    plt.text(0.52,0.93, 'Integration Time of ' + str(int_time)[0:4] + ' (yr)'
               +' \n$\Delta$t = ' + str(dt) + ' (yr)',
               fontsize=txt_fs, transform=ax.transAxes,)
    plt.savefig(save_name, bbox_inches='tight') # saves plot
    plt.close() # closes figure to keep memory use low
    return


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
