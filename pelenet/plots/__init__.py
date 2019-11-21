"""
@desc: Class for plotting and visualisation of network and experiment data
"""
class Plot():

    """
    @desc: Initiates plot object, gets relation to another object for getting the data
    """
    def __init__(self, rel):
        # Store related object
        self.obj = rel
        self.p = rel.p
        self.plotDir = rel.system.datalog.dir + 'plots/'

    """
    @note: Import functions from files
    """
    # Functions to evaluate spikes
    from .spikes import (
        reservoirSpikeTrain, reservoirRates, noiseSpikes, pca,
        autocorrelation, crosscorrelation, spikesMissmatch, ffSpikeCounts
    )
    # Functions to evaluate weights
    from .weights import (
        initialExWeightDistribution, trainedExWeightDistribution,
        initialExWeightMatrix, trainedExWeightMatrix, weightsSortedBySupport
    )
    # Other functions
    from .misc import (
        preSynapticTrace
    )
