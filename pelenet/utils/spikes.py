import numpy as np
import scipy.linalg as la

"""
@desc: From activity probe, calculate spike patterns
"""
def getSpikesFromActivity(self, activityProbes):
    # Get number of probes (equals number of used cores)
    numProbes = np.shape(activityProbes)[0]
    # Concatenate all probes
    activityTrain = []
    for i in range(numProbes):
        activityTrain.extend(activityProbes[i].data)
    # Transform to numpy array
    activityTrain = np.array(activityTrain)
    # Calculate spike train from activity
    #spikeTrain = activityTrain[:,1:] - activityTrain[:,:-1]
    activityTrain[:,1:] -= activityTrain[:,:-1]
    spikeTrain = activityTrain

    return spikeTrain

"""
@desc: Calculate correlation between spike trains of two neurons
"""
def cor(self, t1, t2):
    # Calculate standard devaition of each spike train
    sd1 = np.sqrt(np.correlate(t1, t1)[0])
    sd2 = np.sqrt(np.correlate(t2, t2)[0])

    # Check if any standard deviation is zero
    if (sd1 != 0 and sd2 != 0):
        return np.correlate(t1, t2)[0]/np.multiply(sd1, sd2)
    else:
        return 0

"""
@desc: Smoothing spike train
"""
def getSmoothSpikes(self, spikesTrain):
    # Define some variables
    wd = self.p.smoothingWd  # width of smoothing, number of influenced neurons to the left and right
    var = self.p.smoothingVar  # variance of the Gaussian kernel
    
    # Define the kernel
    lin = np.linspace(-wd,wd,(wd*2)+1)
    kernel = np.exp(-(1/(2*var))*lin**2)

    # Prepare spike window
    spikeWindow = np.concatenate((spikesTrain[-wd:,:], spikesTrain, spikesTrain[:wd,:]))

    # Prepare smoothed array
    nSteps, nNeurons = spikeWindow.shape
    smoothed = np.zeros((nSteps, nNeurons))

    
    # Add smoothing to every spike
    for n in range(nNeurons):
        for t in range(wd, nSteps - wd):
            # Only add something if there is a spike, otherwise just add zeros
            add = kernel if spikeWindow[t,n] == 1 else np.zeros(2*wd+1)
            # Add values to smoothed array
            smoothed[t-wd:t+wd+1, n] += add

    # Return smoothed activity
    return smoothed[wd:-wd,:]
