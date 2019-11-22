import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

"""
@desc: Plot presynaptic trace
"""
def preSynapticTrace(self):
    p = self.obj.reservoirTraceProbeX1[0].plot()

"""
@desc: Plot landscape
"""
def landscape(self):
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
    inputMask = np.mean(self.obj.cueWeights, axis=1)
    inputMask[inputMask > 0] = 1
    inputMask = inputMask.astype(int).reshape((topsize,topsize))

    # Create plit
    fig, ax = plt.subplots(figsize=(10,10))
    # Show landscape, every direction gets a color
    ax.imshow(self.obj.landscape.reshape(topsize, topsize))
    # Highlights the input area
    ax.imshow(inputMask, alpha=0.2, cmap = colors.ListedColormap(['black', 'white']))
    # Show vectors
    ax.quiver(xx, yy, ldx, -ldy, headwidth=2, color="white")  # note: -dy flips quiver to match imshow
    # Define some attributes of the plot and show
    title = 'Quiver Plot (Perlin scale: '+ str(self.p.anisoPerlinScale) +')'
    ax.set(aspect=1, title=title)
    # Save and show
    plt.savefig(self.plotDir + 'landscape.png')
    p = plt.show()
