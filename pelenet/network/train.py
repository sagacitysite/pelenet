import numpy as np
import nxsdk.api.n2a as nx

from .reservoir import ReservoirNetwork

"""
@desc: Reservoir network which starts with random weight matrix and learn with plasticity
"""
class TrainNet(ReservoirNetwork):

    def __init__(self, *args, **kwargs):

        self.reservoirTraceProbesX1 = []

        # Initialize parent
        super().__init__(*args, **kwargs)

    """
    @desc: Build train network structure
    """
    def build(self):
        # Enable learning and set total steps
        self.p.isLearningRule = True
        self.p.totalSteps = self.p.offset + self.p.stepsPerIteration*self.p.trials

        # Connect network
        self.addReservoirNetworkDistributed()

        # Add probes which are specific for training
        self.addTrainProbes()

        # Call build method from parent
        super().buildCommon()

    """
    @desc: Add probing specific for training phase
    """
    def addTrainProbes(self):
        probe = self.connectionChunks[0][0].probe(nx.ProbeParameter.PRE_TRACE_X1)[0]
        self.reservoirTraceProbesX1.append(probe)

        #n, m = np.shape(self.connectionChunks)
        #for i in range(n):
        #    for j in range(m):
        #        probe = self.connectionChunks[i][j].probe(nx.ProbeParameter.PRE_TRACE_X1)[0]
        #        self.reservoirTraceProbesX1.append(probe)
