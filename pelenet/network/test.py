import nxsdk.api.n2a as nx

from .reservoir import ReservoirNetwork

"""
@desc: Reservoir network which starts with trained weight matrix
"""
class TestNet(ReservoirNetwork):

    def __init__(self, *args, **kwargs):
        # Initialize parent
        super().__init__(*args, **kwargs)

    """
    @desc: Build default network structure
    """
    def build(self, reservoirMasks, reservoirWeights):
        # Disable learning
        self.p.isLearningRule = False
        self.p.totalSteps = self.p.stepsPerIteration*self.p.trials

        # Get mask and weights from trained network
        self.initialMasks = reservoirMasks
        self.initialWeights = reservoirWeights

        # Connect network
        self.addReservoirNetwork()

        # Add trace generators and connect them to the ex reservoir
        #for i in range(self.p.traceClusters):
        #    self.addTraceGenerator(i*self.p.traceSteps, cluster=i)

        # Add a cue generator and connect it to the reservoir
        #self.addCueGenerator()

        # Call build method from parent
        super().buildCommon()
