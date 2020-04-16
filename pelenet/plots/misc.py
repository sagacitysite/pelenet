import numpy as np
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import lib.anisotropic.colormap as cmap

"""
@desc: Plot presynaptic trace
"""
def preSynapticTrace(self):
    p = self.obj.reservoirTraceProbeX1[0].plot()

"""
@desc: Plot landscape
"""
def landscape(self, isInput=True, isVectors=False):
    # Define some variables
    topsize = int(np.sqrt(self.p.reservoirExSize))

    # landscape int directions
    #  4  3  2
    #  5  x  1
    #  6  7  8

    # relative directions dx/dy
    #  (-1,1)  (0,1)  (1,1)
    #  (-1,0)    x    (1,0)
    #  (-1,-1) (0,-1) (1,-1)

    # Define relative directions
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1])

    # Transform x and y dimension into relative directions
    ldx = dx[self.obj.landscape % len(dx)]
    ldy = dy[self.obj.landscape % len(dy)]

    # Transform relative directions in landscape to meshgrid
    
    x = np.arange(0, topsize, 1)
    y = np.arange(0, topsize, 1)
    xx, yy = np.meshgrid(x,y)

    # Get input mask
    inputMask = np.mean(self.obj.patchWeights, axis=1)
    inputMask[inputMask > 0] = 1
    inputMask = inputMask.astype(int).reshape((topsize,topsize))

    # Define colors and directions
    #colmap = plt.get_cmap('jet')
    colmap = cmap.virno()
    directions = ['right', 'up-right', 'up', 'up-left', 'left', 'down-left', 'down', 'down-right']
    cols = [ colmap(i) for i in np.linspace(0,1,len(directions)) ]

    # Create plot
    #fig, ax = plt.subplots(figsize=(10,10))
    fig, ax = plt.subplots(figsize=(6,6))
    # Remove grid for this plot
    ax.grid(False)
    # Show landscape, every direction gets a color
    ax.imshow(self.obj.landscape.reshape(topsize, topsize), cmap=colmap)
    # Highlights the input area
    if isInput: ax.imshow(inputMask, alpha=0.2, cmap=colors.ListedColormap(['black', 'white']))
    # Show vectors
    if isVectors: ax.quiver(xx, yy, ldx, -ldy, headwidth=2, color="black")  # note: -dy flips quiver to match imshow
    # Define some attributes of the plot and show
    title = 'Anisotropic landscape'# (perlin scale: '+ str(self.p.anisoPerlinScale) + ')'
    ax.set(aspect=1, title=title)
    # Add legend
    patches = [ mpatches.Patch(color=cols[i], label="{}".format(directions[i]) ) for i in range(len(directions)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Shift direction')
    # Save and show
    plt.savefig(self.plotDir + 'landscape.' + self.p.pltFileType)
    p = plt.show()
