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
    
    self.cuePatchNeurons = np.square(self.cuePatchSize)

    # If reservoirInSize is not set (None), calculate it with given ex/in ratio
    if self.reservoirInSize is None:
        self.reservoirInSize = int(self.reservoirInExRatio * self.reservoirExSize)

    # If totalSteps is not set (None), calculate it with cue, cue relaxation and movement trajectory steps
    if self.totalSteps is None:
        #self.stopStart = self.cueSteps + self.cueRelaxation + self.movementSteps
        self.trialSteps = self.cueSteps + self.cueRelaxation + self.movementSteps
        self.breakSteps = self.stopSteps + self.stopRelaxation
        self.totalTrialSteps = self.trialSteps + self.breakSteps
        self.totalSteps = self.totalTrialSteps * self.trials

    # If cueSteps is not set (None), define cue steps as background activity
    if self.cueSteps is None:
        self.cueSteps = self.totalSteps

    # If noiseNeurons is not set (None), calculate it with given share
    if self.noiseNeurons is None:
        self.noiseNeurons = int(self.noiseNeuronsShare * self.reservoirExSize)

    self.topologySize = int(np.sqrt(self.reservoirExSize))

    # Derived output values
    self.numOutClusters = int(self.reservoirExSize / np.square(self.partitioningClusterSize))
    self.numOutputNeurons = 2 * self.numOutClusters
    self.numOutDimSize = int(np.sqrt(self.numOutClusters))

    # Set datalog path for the current experiment, depending on the current time
    self.expLogPath

    # Calculate total size of the network
    self.reservoirSize = self.reservoirInSize + self.reservoirExSize
    self.reservoirDens = self.numConnectionsPerNeuron / (self.reservoirExSize + self.reservoirInSize)
    self.stepsPerIteration = self.traceClusters * self.traceSteps
    self.traceClusterSize = int(self.traceClusterShare * self.reservoirExSize)

    # If cueSize is not set (None), size equals size of all trace clusters together or whole ex network
    #if self.cueSize is None:
    #    #self.cueSize = self.traceClusters * self.traceClusterSize
    #    self.cueSize = self.reservoirExSize
    
    self.constSize = int(self.constSizeShare * self.reservoirExSize)
    self.offset = self.cueSteps + self.cueRelaxation
