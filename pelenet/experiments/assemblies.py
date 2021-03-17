# Official modules
import numpy as np

# Pelenet modules
from ._abstract import Experiment
from ..network import ReservoirNetwork

"""
@desc: Creating cell assemblies in a reservoir network
"""
class AssemblyExperiment(Experiment):

    def defineParameters(self):
        """
        Define parameters for this experiment
        """
        return {}
