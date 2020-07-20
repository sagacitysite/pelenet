import numpy as np
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter

"""
@desc: Plot 1 dimension of movement
"""
def movement1D(self, est, tgt, dim=None, ylim=None, legend=False, figsize=None, precision=20, suffix=None):
    # Set figsize if given
    if figsize is not None: plt.figure(figsize=figsize)

    # Plot lines
    plt.plot(tgt, linewidth=4.0, color=self.p.pltColor1, label='target trajectory')
    plt.plot(est, linewidth=2.0, color=self.p.pltColor2, label='network output')
    plt.plot(savgol_filter(est, 21, 1), linewidth=2.0, linestyle='dotted', color='#000000', label='smoothed output')

    # Add legend
    if legend: plt.legend()

    # Trim xlim
    plt.xlim(0, len(tgt))

    # Set ylim if given
    if ylim is not None:
        plt.yticks(getTicks(ylim, precision))
        plt.ylim(ylim)

    # Prepare suffix (add given suffix and/or dimension to file name)
    if dim is not None and suffix is not None:
        suffix = '_' + dim + '_' + suffix
    if dim is not None and suffix is None:
        suffix = '_' + dim
    if dim is None and suffix is not None:
        suffix = '_' + suffix

    # Set default value for dim if dim is not given
    # NOTE must be after suffix to avoid suffix creation in that case
    if dim is None: dim = 'distance'

    # Save and show
    plt.xlabel('time steps')
    plt.ylabel(str(dim)+' [m]')
    plt.savefig(self.plotDir + 'movement_1d'+str(suffix)+'.' + self.p.pltFileType)
    p = plt.show()

"""
@desc: Plot all 3 dimensions of movement
"""
def movement3D(self, est, tgt, view=(20, 120), xlim=None, ylim=None, zlim=None, figsize=None):
    # Set figsize if given
    if figsize is not None: plt.figure(figsize=figsize)

    # Prepare plot
    #fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    # Plot target
    ax.plot3D(tgt[0], tgt[1], tgt[2], linewidth=4.0, color=self.p.pltColor1)
    # Plot estimates
    ax.plot3D(est[0], est[1], est[2], linewidth=2.0, color=self.p.pltColor2)
    # Plot smoothed estimates
    ax.plot3D(savgol_filter(
        est[0], 21, 1), savgol_filter(est[1], 21, 1), savgol_filter(est[2], 21, 1),
        linewidth=2.0, linestyle='dotted', color='#000000'
    )

    # Set axis and limits if limits are given
    if xlim is not None:
        ax.set_xticks(getTicks(xlim))
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_yticks(getTicks(ylim))
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zticks(getTicks(zlim))
        ax.set_zlim(zlim)

    # Set view perspective
    ax.view_init(*view)

    # Set labels
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    # Save and show
    plt.savefig(self.plotDir + 'movement_3d.' + self.p.pltFileType)
    p = plt.show()

"""
@desc: Simple helper function to create ticks from plot limits with a given precision
"""
def getTicks(limits, precision=20):
    # Calc upper and bottom ticks
    b = float(np.ceil(limits[0]*precision)/precision)
    t = float(np.floor(limits[1]*precision)/precision)+0.0000001

    # Arange ticks and return
    return np.arange(b, t, float(1/precision))
