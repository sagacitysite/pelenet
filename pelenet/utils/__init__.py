from ..parameters import Parameters
from lib.helper.singleton import Singleton
import logging

"""
@desc: Singleton for util functions, like calculating more involved evaluations of data
"""
@Singleton
class Utils():

    # Store instance of Plot class
    #_instance = None

    """
    @desc: Construct new object, but only if instance does not already exists
    """
    #def __new__(cls, *args, **kwargs):
    #    # Check if instance does already exists, if not, create an instance and store it
    #    if not cls._instance:
    #        cls._instance = object.__new__(cls, *args, **kwargs)
    #    # Return instance, either just created or from memory
    #    return cls._instance

    """
    @desc: Initiates plot object, gets relation to another object for getting the data
    """
    def __init__(self):
        # Store parameters
        self.p = Parameters()

    """
    @desc: Parameters for utils can be changed manually
    """
    def setParameters(self, parameters):
        # Manually set parameters
        self.p = parameters

    """
    @note: Import functions from files
    """
    # Functions to evaluate spikes
    from .spikes import (
        getSpikesFromActivity, cor, getSmoothSpikes
    )
    # Functions to evaluate weights
    from .weights import (
        getSpectralRadius, recombineExWeightMatrix, getSupportWeightsMask
    )
    # Other functions
    from .misc import (
        pca
    )
