import numpy as np

"""
@desc: Some values are derived and need to be computed
@note:  - This function is necessary, since it needs to be called
          when parameter optimization is performed
        - Derived parameters are not allowed to change within one experiment
          (only inbetween experiments), for values which are more flexibel,
          use system variables
"""
def computeDerived(self):

    # Initialize some derived parameters
    self.totalSteps = None  # Number of simulation steps
    self.reservoirSize = None  # Total size of reservoir

    """
    Define some derived parameters
    """

    # Set some derived parameters directly
    self.patchNeurons = np.square(self.patchSize)  # Number of patch input neurons
    self.topologySize = int(np.sqrt(self.reservoirExSize))  # Size of one dimension of topology

    # Derived output values
    self.numOutClusters = int(self.reservoirExSize / np.square(self.partitioningClusterSize))
    self.numOutputNeurons = 2 * self.numOutClusters
    self.numOutDimSize = int(np.sqrt(self.numOutClusters))

    # 
    self.stepsPerIteration = self.traceClusters * self.traceSteps
    self.traceClusterSize = int(self.traceClusterShare * self.reservoirExSize)

    # 
    self.constSize = int(self.constSizeShare * self.reservoirExSize)
    self.offset = self.patchSteps + self.patchRelaxation

    """
    Define conditional parameters
    """

    # If reservoirInSize is not set (None), calculate it with given ex/in ratio
    if self.reservoirInSize is None:
        self.reservoirInSize = int(self.reservoirInExRatio * self.reservoirExSize)

    # If totalSteps is not set (None), calculate it with cue, cue relaxation and trial steps
    if self.totalSteps is None:
        #self.stopStart = self.patchSteps + self.patchRelaxation + self.stepsPerTrial
        self.trialSteps = self.patchSteps + self.patchRelaxation + self.stepsPerTrial
        self.breakSteps = self.stopSteps + self.stopRelaxation
        self.totalTrialSteps = self.trialSteps + self.breakSteps
        self.totalSteps = self.totalTrialSteps * self.trials

    # If patchSteps is not set (None), define cue steps as background activity
    if self.patchSteps is None:
        self.patchSteps = self.totalSteps

    # If noiseNeurons is not set (None), calculate it with given share
    if self.noiseNeurons is None:
        self.noiseNeurons = int(self.noiseNeuronsShare * self.reservoirExSize)    

    # Set datalog path for the current experiment, depending on the current time
    #self.expLogPath

    # Calculate total size of the network
    self.reservoirSize = self.reservoirInSize + self.reservoirExSize

    # Calculate connectivity
    if self.reservoirConnProb is None:
        self.reservoirConnProb = int(self.reservoirConnPerNeuron / self.reservoirSize)
    if self.reservoirConnPerNeuron is None:
        self.reservoirConnPerNeuron = int(self.reservoirConnProb * self.reservoirSize)
