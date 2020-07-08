import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import warnings

"""
@desc: Plot spike train of neurons in reservoir
"""
def reservoirSpikeTrain(self, fr=0, to=None, figsize=None, colorEx=None, colorIn=None):
    # Get spikes
    exSpikes = self.obj.exSpikeTrains if self.p.isExSpikeProbe else None
    inSpikes = 2*self.obj.inSpikeTrains if self.p.isInSpikeProbe else None  # multiply by 2 to enable a different color in imshow

    # Set colors if not chosen
    if colorEx is None: colorEx = self.p.pltColor1
    if colorIn is None: colorIn = self.p.pltColor2

    # If no spike probe is in use and we can stop here
    if (not self.p.isExSpikeProbe) and (not self.p.isInSpikeProbe):
        warnings.warn("No excitatory or inhibitory spikes were probed, spike trains cannot be shown.")
        return

    # Combine ex and in spikes
    allSpikes = None
    if self.p.isExSpikeProbe and self.p.isInSpikeProbe:
        allSpikes = np.vstack((exSpikes, inSpikes))
    elif self.p.isExSpikeProbe:
        allSpikes = exSpikes
    elif self.p.isInSpikeProbe:
        allSpikes = inSpikes

    # Choose spikes ("zoom" in time)
    chosenSpikes = allSpikes[:, fr:to]

    # Define colors
    cmap = colors.ListedColormap(['#ffffff', colorEx, colorIn])

    # Plot spike train
    if figsize is not None: plt.figure(figsize=figsize)
    plt.imshow(chosenSpikes, cmap=cmap, vmin=0, vmax=2, aspect='auto')
    #plt.title('Reservoir spikes')
    plt.xlabel('time steps')
    plt.ylabel('index of neuron')
    plt.savefig(self.plotDir + 'spikes_raster.' + self.p.pltFileType)
    p = plt.show()

"""
@desc: Plot spike train of output neurons
"""
def outputSpikeTrain(self, fr=0, to=None, color=None, figsize=None):
    # Get spikes
    outSpikes = self.obj.outSpikeTrains if self.p.isOutSpikeProbe else None

    # If no spike probe is in use and we can stop here
    if (not self.p.isOutSpikeProbe):
        warnings.warn("No output spikes were probed, spike trains cannot be shown.")
        return

    # Set colors if not chosen
    if color is None: color = self.p.pltColor1
    
    # Choose spikes ("zoom" in time)
    chosenSpikes = outSpikes[:, fr:to]

    # Define colors
    cmap = colors.ListedColormap(['#ffffff', color])

    # Plot spike train
    if figsize is not None: plt.figure(figsize=figsize)
    plt.imshow(chosenSpikes, cmap=cmap, aspect='auto')
    #plt.title('Output spikes')
    plt.xlabel('time steps')
    plt.ylabel('index of neuron')
    plt.savefig(self.plotDir + 'spikes_output_raster.' + self.p.pltFileType)
    p = plt.show()

"""
@desc: Plot average firing rate of reservoir neurons
"""
def reservoirRates(self, fr=0, to=None, ylim=None, figsize=None, colorEx=None, colorIn=None, legend=True):
    # Set 'to' to total times steps if not defined
    if to is None: to = self.p.totalSteps

    # Calculate mean rate for every simulation step
    meanRateEx = np.mean(self.obj.exSpikeTrains, axis=0)[fr:to] if self.p.isExSpikeProbe else None
    meanRateIn = np.mean(self.obj.inSpikeTrains, axis=0)[fr:to] if self.p.isInSpikeProbe else None

    # If no spike probe is in use and we can stop here
    if (not self.p.isExSpikeProbe) and (not self.p.isInSpikeProbe):
        warnings.warn("No spikes were probed, reservoir rates cannot be shown.")
        return

    # Set colors if not chosen
    if colorEx is None: colorEx = self.p.pltColor1
    if colorIn is None: colorIn = self.p.pltColor2

    # Concatenate ex and in spikes
    meanRate = None
    if self.p.isExSpikeProbe and self.p.isInSpikeProbe:
        meanRate = np.concatenate((meanRateEx, meanRateIn))
    elif self.p.isExSpikeProbe:
        meanRate = meanRateEx
    elif self.p.isInSpikeProbe:
        meanRate = meanRateIn

    # Calculate mean rate for whole simulation, except cue steps
    totalMeanRate = np.round(np.mean(meanRate[self.p.inputSteps:])*1000)/1000

    # Define alpha level
    alpha = 0.7 if meanRateEx is not None and meanRateIn is not None else 1.0

    # Set figsize of given
    if figsize is not None: plt.figure(figsize=figsize)

    # Set labels
    plt.ylabel('mean firing rate')
    plt.xlabel('time steps')

    # Plot mean rates
    if meanRateIn is not None:
        plt.plot(np.arange(0,to-fr,1), meanRateIn, alpha=alpha, color=colorIn, label='Inhibitory neurons')
    if meanRateEx is not None:
        plt.plot(np.arange(0,to-fr,1), meanRateEx, alpha=alpha, color=colorEx, label='Excitatory neurons')

    # Show legend if both mean rates are plotted and legend flag is set
    if meanRateEx is not None and meanRateIn is not None and legend is True: plt.legend()

    # Set y limit if given
    plt.xlim((0, to-fr))
    if ylim is not None: plt.ylim(ylim)

    # Save and show plot
    plt.savefig(self.plotDir + 'spikes_rates.' + self.p.pltFileType)
    p = plt.show()

"""
@desc: 
"""
def outputRates(self, fr=0, to=None, ylim=None, color=None, figsize=None):
    # Set 'to' to total times steps if not defined
    if to is None: to = self.p.totalSteps

    # If no spike probe is in use and we can stop here
    if (not self.p.isOutSpikeProbe):
        warnings.warn("No output spikes were probed, spike trains cannot be shown.")
        return

    # Set color if not chosen
    if color is None: color = self.p.pltColor1
    
    # Get rate
    rate = np.mean(self.obj.outSpikeTrains, axis=0)[fr:to]

    # Plot mean rate
    if figsize is not None: plt.figure(figsize=figsize)
    plt.ylabel('mean firing rate')
    plt.xlabel('time steps')
    plt.plot(np.arange(fr,to,1), rate, color=color)

    # Set y limit if given
    if ylim is not None: plt.ylim(ylim)

    plt.savefig(self.plotDir + 'spikes_output_rates.' + self.p.pltFileType)
    p = plt.show()


"""
@desc: Plot spikes of noise neurons
"""
def noiseSpikes(self):
    plt.figure(figsize=(16, 4))
    plt.title('noise spikes')
    plt.savefig(self.plotDir + 'spikes_noise.' + self.p.pltFileType)
    p = plt.imshow(self.obj.noiseSpikes, cmap='Greys', aspect='auto')

"""
@desc: Plot autocorreltaion function
"""
def autocorrelation(spikes, numNeuronsToShow=3):
    for i in range(numNeuronsToShow):
        result = np.array([cor(spikes[i,:-t], spikes[i,t:]) for t in range(1,101)])
        plt.figure(figsize=(16, 4))
        plt.plot(np.arange(result.shape[0]), result, linestyle='-', marker='.')
        plt.title('Autocorrelation')
        plt.xlabel('$\Delta t$')
        plt.ylabel('ACF')
        plt.savefig(self.plotDir + 'spikes_autocorrelation.' + self.p.pltFileType)
        p = plt.show()

"""
@desc: Plot crosscorreltaion function
"""
def crosscorrelation(spikes):
    # Get number of neurons and define crosscorrelation array
    n = spikes.shape[0]
    crosscor = np.zeros((n,n))

    # Loop throught spikes numbers
    for i in range(n):
        for j in range(n):
            # Calculate normalized cross correlations between spike trains
            crosscor[i,j] = cor(spikes[i,:], spikes[j,:])

            # TODO: Use single network instead of mean
            # Are the lines the neurons where the cue comes in?

    # Plot cross correlation matrix
    plt.imshow(crosscor)
    plt.title('Crosscorrelation')
    plt.colorbar()
    plt.savefig(self.plotDir + 'spikes_crosscorrelation.' + self.p.pltFileType)
    p = plt.show()

"""
@desc: Plot first 2 dimensions of PCA
"""
def pca(self, spikes):
    # Perform PCA
    res = self.obj.utils.pca(spikes.T.astype(float))

    # Get components from result
    comps = res[0]

    # Plot first 2 dimensions
    plt.plot(comps[:,0], comps[:,1])
    plt.title('PCA')
    plt.savefig(self.plotDir + 'spikes_pca.png')
    p = plt.show()

"""
@desc: Plot spike missmatches
"""
def spikesMissmatch(self, trainSpikes, testSpikes, windowSize = 20):
    n = self.p.reservoirExSize
    T = self.p.simulationSteps
    numSteps = int(T/windowSize)

    spikeMissmatches = np.zeros((self.p.trainingTrials, numSteps))
    # Caclulate missmatch between test and every training trial
    for i in range(self.p.trainingTrials):
        # Calculate the missmatch for every window
        for j in range(numSteps):
            # Define start and end point of window (from/to)
            fr, to = int(j*windowSize), int((j+1)*windowSize)
            # Calculate missmatch between spiking arrays
            spikeMissmatches[i,j] = np.sum(trainSpikes != testSpikes)/(windowSize*n)

    # Plot missmatch for every training trial
    plt.figure(figsize=(16, 4))
    plt.xlabel('Time')
    plt.ylabel('% of missmatching spikes')
    plt.title('Training trials compared with test trial')
    plt.ylim((-0.05,0.55))
    for i in range(self.p.trainingTrials):
        plt.plot(np.arange(0, T, windowSize), spikeMissmatches[i,:])
    plt.savefig(self.plotDir + 'spikes_missmatch.' + self.p.pltFileType)
    p = plt.show()

"""
@desc: Plot fano factors of spike counts
"""
"""
@desc: Fano factors of spike counts
"""
def ffSpikeCounts(spikes, neuronIdx, windowSize = 50):
    numSteps = int(spikes.shape[1]/windowSize)
    fanoFactors = []

    # Loop over all windows
    for i in range(numSteps):
        # Define starting and end point of window
        fr, to = int(i*windowSize), int((i+1)*windowSize)
        # Calculate variance and mean of spike train for specific window
        var = np.var(spikes[neuronIdx,fr:to], axis=1)
        mean = np.mean(spikes[neuronIdx,fr:to], axis=1)
        # Append fano factor to list
        fanoFactors.append(var/mean)
        
    # Make numpy array
    fanoFactors =  np.array(fanoFactors)

    # Define x for plotting
    x = np.arange(0, fanoFactors.shape[0]*windowSize, windowSize)

    # Plot fano factor for every chosen neuron
    plt.figure(figsize=(16, 4))
    for i in range(fanoFactors.shape[1]):
        labelTxt = 'Neuron '+str(neuronIdx[i])
        plt.plot(x, fanoFactors[:,i], marker='.', label=labelTxt)
    plt.xlabel('Time')
    plt.ylabel('F')
    plt.ylim((0,1))
    plt.legend()
    plt.title('Fano factors for test run')
    plt.savefig(self.plotDir + 'spikes_fanofactors.' + self.p.pltFileType)
    p = plt.show()

"""
@desc: Plots the mean activity in the topology
"""
def meanTopologyActivity(self):
    # Define some variables
    topsize = int(np.sqrt(self.p.reservoirExSize))

    # Get topology of spikes and mean over time
    topoSpikes = self.obj.exSpikeTrains.reshape(-1, topsize, topsize)
    topoSpikesMean = np.mean(topoSpikes, axis=0)

    # Plot mean activity
    plt.figure(figsize=(6, 6))
    plt.title('Topological mean activity (Perlin scale: '+ str(self.p.anisoPerlinScale) + ')')
    #st_mean[st_mean < 0.06] = 0
    plt.imshow(topoSpikesMean)
    plt.savefig(self.plotDir + 'spikes_topology_activity_mean.' + self.p.pltFileType)
    p = plt.show()
