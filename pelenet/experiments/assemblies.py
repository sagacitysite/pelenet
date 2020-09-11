# Official modules
import numpy as np

# Pelenet modules
from ._abstract import Experiment

"""
@desc: Creating cell assemblies in a reservoir network
"""
class AssemblyExperiment(Experiment):

    """
    @desc: Define parameters for this experiment
    """
    def defineParameters(self):
        return {
            # Experiment
            'trials': 5,
            'stepsPerTrial': 90,
            # Network
            'reservoirExSize': 2048,
            'reservoirConnProb': None,
            'reservoirConnPerNeuron': 45,
            'isLearningRule': True,
            'learningRule': '2^-2*x1*y0 - 2^-2*y1*x0 + 2^-4*x1*y1*y0 - 2^-3*y0*w*w',
            # Probes
            'isExSpikeProbe': True
        }
