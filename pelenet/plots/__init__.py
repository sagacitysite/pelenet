from ..system import System

"""
@desc: Class for plotting and visualisation of network and experiment data
"""
class Plot():

    """
    @desc: Initiates plot object, gets relation to another object for getting the data
    """
    def __init__(self, rel):
        # Get system instance
        system = System.instance()

        # Store related object
        self.obj = rel
        self.p = rel.p
        self.plotDir = system.datalog.dir + 'plots/'

    """
    @note: Import functions from files
    """
    # Functions to evaluate spikes
    from .spikes import (
        reservoirSpikeTrain, reservoirRates, noiseSpikes, pca,
        autocorrelation, crosscorrelation, spikesMissmatch, ffSpikeCounts,
        meanTopologyActivity
    )
    # Functions to evaluate weights
    from .weights import (
        initialExWeightDistribution, trainedExWeightDistribution,
        initialExWeightMatrix, trainedExWeightMatrix, weightsSortedBySupport,
        cueWeightMean
    )
    # Other functions
    from .misc import (
        preSynapticTrace, landscape
    )
