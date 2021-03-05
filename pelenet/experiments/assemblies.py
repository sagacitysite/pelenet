# Official modules
import numpy as np

# Pelenet modules
from ._abstract import Experiment
from ..network import ReservoirNetwork

"""
@desc: Creating cell assemblies in a reservoir network
"""
class AssemblyExperiment(Experiment):

    """
    @desc: Define parameters for this experiment
    """
    def defineParameters(self):
        return {
            # # Experiment
            # 'seed': 1,
            # 'stepsPerTrial': 90,
            # # Network
            # 'reservoirExSize': 2048,
            # 'reservoirConnProb': None,
            # 'reservoirConnPerNeuron': 45,
            # 'isLearningRule': True,
            # 'learningRule': '2^-2*x1*y0 - 2^-2*y1*x0 + 2^-4*x1*y1*y0 - 2^-3*y0*w*w',
            # # Probes
            # 'isExSpikeProbe': True
        }
    
    """
    @desc: Overwrite build reservoir network with given mask, weights and input
    """
    def build(self, mask=None, weights=None, inputSpikes=[]):
        # Instanciate innate network
        self.net = ReservoirNetwork(self.p)
        
        # Define mask and weights
        if (mask is None and weights is None):
            # Draw anisotropic mask and weights
            self.net.drawMaskAndWeights()
        elif (mask is not None and weights is not None):
            # Set mask and weights
            self.net.initialMasks = mask
            self.net.initialWeights = weights
        else:
            # Throw an error if only one of mask/weights is defiend
            raise Exception("It is not possible to define only one of mask and weights, both must be defined or not defined.")

        # Connect ex-in reservoir
        self.net.connectReservoir()

        # Add input
        if self.p.isInput:
            self.net.addInput(inputSpikes)

        # Add background noise
        if self.p.isNoise:
            self.net.addNoiseGenerator()

        # Add Probes
        self.net.addProbes()

        # Call afterBuild
        self.afterBuild()
