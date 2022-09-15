#!/usr/bin/env python3

#=============================================================================#
#                                                                             #
# NAME:     q1.py                                                             #
#                                                                             #
# PURPOSE: Calculatss Newtonion, and GR orbits for Mercury,                   #
#          produces orbital dynamic plots                                     #
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
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
# Set constants
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


def fg_solver(xi, yi, vxi, vyi, dt=0.0001, int_time=1., alpha=0):
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



def angular_momentum(x, y, vx, vy, m_p, m_sun = 2.0*1e30):
    '''Calculates the angular momentum for a given planet
       around the sun. Uses L = |v|m_p*r, returned value is in SI.

       Args:
       x: float, in AU
          postion of the planet, along the x direction

       y: float, in AU
          postion of the planet, along the y direction

       vx: float, in au/year
           the x velocity componet

       vy: float, in au/year
           the x velocity componet

       m_p: float, in kg
            mass of the planet

       **KArgs:
       m_sun: float, in kg
              mass of the sun, default is m_sun = 2.0*1e30

       returns:
       L: float
          angular momentum in SI units
       '''
    # Convert to si units
    x *= Au
    y *= AU
    vx = vx * AU/year
    vy = vy * AU/year

    # calculate r, and v vectors
    r = np.sqrt (x**2 + y**2)
    v = np.sqrt(vx**2 + vy**2)

    # Calculate L
    L = v*m_p*r # this assumes v, r are orthogonal

    return L

def velcoity_plot(vx_arrs, vy_arrs, int_time, dt,
                title = 'title', save_name='youforgottoset_save_name.pdf',
                fs=15):
    ''' Plots Velcoity componets (x,y) as a function of time.
        Uses matplotlib.pyplot. Saves output.

    Args:
    vx_arrs: list, in AU/yr
             n vx_arrs (array, float) in a list

    vx_arrs: list, in AU/yr
             n vy_arrs (array, float) in a list

    int_time: float, in Earth years
              Integration time for solution, IE how long the plot is for


    dt: float, Earth in years
              dt to use for plotting, default is 0.0001

    **KArgs:
    title: str
           title of plot

    save_name: str
               argument for plt.savefig()
               default is 'youforgottoset_save_name.pdf

    fs: int
        fontsize to use for plot, default is 15

    Returns:
    None, plot is saved in the same dir as where the program is stored

    '''
    # make time array
    time = np.linspace(0, int_time, int(int_time/dt))

    # plot componets
    for i in range(len(vx_arrs)):
        plt.plot(time, vx_arrs[i], linestyle='--', color = 'green',
                 label='$v_{x}$')
        plt.plot(time, vy_arrs[i], color= 'k', label='$v_{y}$')

    # adjust plot parameters and add labels
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlabel('Time (yr)', fontsize = fs)
    plt.ylabel('Velocity (AU/yr)', fontsize = fs)
    plt.legend(fontsize=fs, bbox_to_anchor=(1, 1))
    plt.title(title, fontsize=fs)
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight') # saves plopt
    plt.close() # closes figure to save mem


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

    # set inital condictions, and int_time, and dt
    xi = 0.47 # Au
    yi = 0 # Au
    vxi = 0 # Au/yr
    vyi = 8.17 # Au/yr
    dt = 0.0001 # yr
    int_time = 1 # yr

    # generate dynamical variables for Newtonion orbit
    dynamic_vars_class = fg_solver(xi=xi, yi=yi, vxi=vxi,
                                   vyi=vyi, dt=dt, int_time=int_time, alpha=0)
    x_arr_class, y_arr_class, vx_arr_class, vy_arr_class = dynamic_vars_class

    # generate dynamical variables for GR orbit
    dynamic_vars_gr = fg_solver(xi=xi, yi=yi, vxi=vxi,vyi=vyi, dt=dt,
                                int_time=int_time, alpha=0.01)
    x_arr_gr, y_arr_gr, vx_arr_gr, vy_arr_gr = dynamic_vars_gr

    # prep for plotting function
    x_arrs = [x_arr_class, x_arr_gr]
    y_arrs = [y_arr_class, y_arr_gr]
    labels = ['Newtonion', 'General Relativity']
    colors = ['green', 'k']
    color_int = 'k'
    line_styles = ['-', '--']

    # set titles and plot
    title= 'Mercury\'s Orbit'
    plot_orbits(x_arrs, y_arrs, labels, colors, line_styles, title,
                    int_time, dt, save_name='q1_orbit.pdf')


    title = 'Mercury\'s Newtonion Orbit'
    velcoity_plot([vx_arr_class], [vy_arr_class], int_time=int_time,
                  dt=dt, fs=15, title = title, save_name='q1_vel_newton.pdf')

    title= 'Mercury\'s Newtonion Orbit'
    velcoity_plot([vx_arr_gr], [vy_arr_gr], int_time=int_time,
                  dt=dt, fs=15, title = title, save_name='q1_vel_gr.pdf')
