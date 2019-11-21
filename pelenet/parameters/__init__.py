import numpy as np
from tabulate import tabulate

"""
@desc: Define custom error for parameter validility problems
"""
class ParamaterNotValid(Exception):
   """Raised when a parameter is not valid"""
   pass

"""
@desc: Contains all parameters and claculates derived values
"""
class Parameters():

    from .system import includeSystem
    from .experiment import includeExperiment
    from .derived import computeDerived

    """
    @desc: Initializes all parameters
    @note: Parameters defined as 'None' are computed values (function computeDerivedParameters below)
    """
    def __init__(self, includeDerived = True):
        # Include parameters
        self.includeSystem()
        self.includeExperiment()

        # If derived values shall be included, calculate them
        if includeDerived: self.computeDerived()

        # Check if all paramaters are valid
        self.vadilityCheck()
    
    """
    @desc: Transform all available parameters to string table format for printing
    """
    def __str__(self):
        # Get names of all elements in parameter module
        parNames = dir(self)
        # Initialize parameter table
        parTable = []

        # Loop through all variables names in paramater module
        for pn in parNames:
            # If variable starts with __ continue with next loop step
            if pn.startswith('__') or pn.startswith('np'):
                continue
            # Append current variable name and variable content as list to tab list
            parTable.append([pn, getattr(self, pn)])
            
        # Return list of variables in table format
        return tabulate(parTable)

    """
    @desc: Check if values make sense and raise error if not
    """
    def vadilityCheck(self):
        # If number of connections per neuron is larger than total number of neurons
        if self.numConnectionsPerNeuron > (self.reservoirExSize + self.reservoirInSize):
            raise ParamaterNotValid('Number of connections per neuron must be larger than number of neurons in the network.')

        # Check if ex neurons are square rootable
        #if self.numConnectionsPerNeuron > (self.reservoirExSize + self.reservoirInSize):
        #    raise ParamaterNotValid('Number of connections per neuron must be larger than number of neurons in the network')

        # Check if size of cue input is smaller than network site
        if self.cuePatchNeurons > self.reservoirSize:
            raise ParamaterNotValid('Cue size is too large, cannot be larger than network size.')
