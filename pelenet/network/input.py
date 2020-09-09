import numpy as np
import logging
import itertools
from scipy import sparse

"""
@desc: Generates a sinus signal (one 'hill') in given length of time steps
"""
def generateSinusInput(length):
    # Draw from a sin wave from 0 to 3,14 (one 'hill')
    probCoeff = 1 #1.5  # the higher the probability coefficient, the more activity is in the network
    probs = probCoeff * np.abs(np.sin((np.pi / length) * np.arange(length)))
    randoms = np.random.rand(length)

    # Get indices of spike
    spikeInd = np.where(randoms < probs)[0]

    # Return spikes
    return spikeInd

"""
@desc: Generates a simple input signal
"""
def generateUniformInput(length, prob=0.1):
    spikes = np.zeros((length, 1))
    randoms = np.random.rand(length)

    # Draw spikes
    for i in range(length):
        spikes[i] = (randoms[i] < prob)
    
    # Get indices of spike
    spikeTimes = np.where(spikes)[0]

    # Return spikes
    return spikeTimes

"""
@desc:  Create a sequence of inputs
"""
def addInputSequence(self):
    # Iterate over number of inputs
    for i in range(self.p.inputSequenceSize):
        # Create spike generator
        sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.inputNumTargetNeurons)

        # Draw spikes for input generators for current input
        inputSpikes = drawSpikesForAllGenerators(self, offset=i*self.p.inputSteps)
        self.inputSpikes.append(inputSpikes)

        # Add spikes s to generator i
        for k, s in enumerate(inputSpikes):
            sg.addSpikes(spikeInputPortNodeIds=k, spikeTimes=s.tolist())

        # Get inidices of target neurons for current input
        inputTargetNeurons = np.arange(i*self.p.inputNumTargetNeurons, (i+1)*self.p.inputNumTargetNeurons)
        self.inputTargetNeurons.append(inputTargetNeurons)

        # Connect spike generators to reservoir
        self.inputWeights = connectSpikeGenerator(self, sg, inputTargetNeurons)

    # Log that input was added
    logging.info('Input sequence was added to the network')

"""
@desc:  Adds input where n target neurons are left out every trial
        The number of target neurons may differ due to rounding in getTargetNeurons() function
        Therefore self.p.inputNumTargetNeurons cannot be used here, but instead len(self.inputTargetNeurons) must be used
        
"""
def addLeaveNOutInput(self):
    # Get inidices of target neurons if not already given
    self.inputTargetNeurons = getTargetNeurons(self)

    # Create spike generator
    sg = self.nxNet.createSpikeGenProcess(numPorts=len(self.inputTargetNeurons))

    # Define all possible combinations to leave one out
    combinations = np.array(list(itertools.combinations(np.arange(len(self.inputTargetNeurons)), self.p.inputNumLeaveOut)))

    # Iterate over target neurons to draw spikes
    for i in range(len(self.inputTargetNeurons)):
        
        spikeTimes = []
        # Iterate over trials
        for k in range(self.p.trials):
            # Add spike times only when i != k
            apply = np.all([combinations[k, m] != i for m in range(self.p.inputNumLeaveOut)])

            # If spike generator produces input for the current trial, add it to spikeTimes
            if (apply or not self.p.inputIsLeaveOut):
                offset = self.p.stepsPerTrial*k + self.p.resetOffset*(k+1)
                spks = drawSpikes(self, offset=offset)
                spikeTimes.append(spks)

        # Add spike indices to inputSpikes array and to generator
        s = list(itertools.chain(*spikeTimes))
        self.inputSpikes.append(s)
        sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=s)

    # Connect spike generators to reservoir
    self.inputWeights = connectSpikeGenerator(self, sg, self.inputTargetNeurons)

    # Log that input was added
    logging.info('Leave-one-out input was added to the network')

"""
@desc:  Create input spiking generator,
        the input is connected to the reservoir network,
        an excitatory connection type is used
@params:
        inputSpikeIndices: indices of input spikes for spike generators
        targetNeuronIndices: indices of reservoir neurons to connect input to, default is None (indices are just taken successively)
"""
def addInput(self, inputSpikeIndices=[], targetNeuronIndices=[]):
    # Create spike generator
    sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.inputNumTargetNeurons)

    # Draw spikes for input generators if not already given
    self.inputSpikes = inputSpikeIndices if len(inputSpikeIndices) else drawSpikesForAllGenerators(self)

    # Add spikes s to generator i
    for i, s in enumerate(self.inputSpikes):
        sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=s.tolist())

    # Get inidices of target neurons if not already given
    self.inputTargetNeurons = targetNeuronIndices if len(targetNeuronIndices) else getTargetNeurons(self)

    # Connect spike generators to reservoir
    self.inputWeights = connectSpikeGenerator(self, sg, self.inputTargetNeurons)

    # Log that input was added
    logging.info('Input was added to the network')

"""
@desc: Draw spikes for ALL spike generators
"""
def drawSpikesForAllGenerators(self, offset=0):
    # Initialize array for spike indices
    inputSpikes = []

    # Number of spike generators
    numGens = self.p.inputNumTargetNeurons

    # Draw spikes for every spike generator
    for i in range(numGens):
        drawSpikes(self, offset)
        # Multiply spikes to all trials
        trialsRange = range(0, self.p.totalSteps, self.p.totalTrialSteps)
        sAll = np.ndarray.flatten(np.array([s + i for i in trialsRange]))
        # Store input spikes in array
        inputSpikes.append(sAll)

    return inputSpikes

"""
@desc: Draw spikes for ONE spike generator
"""
def drawSpikes(self, offset):
    s = []
    # Generate spikes, depending on input type
    if self.p.inputType == 'uniform':
        s = self.p.inputOffset + offset + generateUniformInput(self.p.inputSteps, prob=self.p.inputGenSpikeProb)
    if self.p.inputType == 'sinus':
        s = self.p.inputOffset + offset + generateSinusInput(self.p.inputSteps)

    return s

"""
@desc: Define target neurons in reservoir to connect generators with
"""
def getTargetNeurons(self):
    # Initialize array for target neurons
    targetNeurons = []

    # If topology should NOT be considered, just take first n neurons as target
    if not self.p.inputIsTopology:
        targetNeurons = np.arange(self.p.inputNumTargetNeurons)

    # If topology should be considered, define square input target area
    if self.p.inputIsTopology:
        # Define size and 
        targetNeuronsEdge = int(np.sqrt(self.p.inputNumTargetNeurons))
        exNeuronsEdge = int(np.sqrt(self.p.reservoirExSize))

        # Get shifts for the input area of the target neurons
        sX = self.p.inputShiftX
        sY = self.p.inputShiftY

        # Define input region in network topology and store their indices
        topology = np.zeros((exNeuronsEdge,exNeuronsEdge))
        topology[sY:sY+targetNeuronsEdge,sX:sX+targetNeuronsEdge] = 1
        targetNeurons = np.where(topology.flatten())[0]

    return targetNeurons
    
"""
@desc:  Connect spike generators with target neurons of reservoir
        Every spike generator is connected to one neuron
        Finally draws uniformly distributed weights
@params:
        spikeGenerators: nxsdk spike generators
        inputTargetNeurons: indices of reservoir target neurons
"""
def connectSpikeGenerator(self, spikeGenerators, inputTargetNeurons):
    # Creates empty mask matrix
    inputMask = np.zeros((self.p.reservoirExSize, len(inputTargetNeurons)))

    # Every generator is connected to one 
    for i, idx in enumerate(inputTargetNeurons):
        inputMask[idx,i:i+1] = 1

    # Transform to sparse matrix
    inputMask = sparse.csr_matrix(inputMask)

    # Draw weights
    inputWeights = self.drawSparseWeightMatrix(inputMask, distribution='uniform')

    # Connect generator to the reservoir network
    for i in range(len(self.exReservoirChunks)):
        fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
        ma = inputMask[fr:to, :].toarray()
        we = inputWeights[fr:to, :].toarray()
        spikeGenerators.connect(self.exReservoirChunks[i], prototype=self.genConnProto, connectionMask=ma, weight=we)

    return inputWeights

"""
@desc:  Create input spiking generator to add a patch signal,
        the input is connected to the reservoir network,
        an excitatory connection prototype is used
@params:
        connectToIndices: indices of reservoir neurons to connect input to, default is None (indices are just taken successively)
"""
def addInput2D(self, connectToIndices=None):
    idc = connectToIndices

    patchGens = int(self.p.patchGensPerNeuron*self.p.patchNeurons)

    # Create spike generator
    sg = self.nxNet.createSpikeGenProcess(numPorts=patchGens)

    combinations = np.array(list(itertools.combinations(np.arange(self.p.patchNeurons), self.p.patchMissingNeurons)))

    # Initialize counter
    cnt = 0
    # Iterate over patch neurons
    for i in range(self.p.patchNeurons):
        
        spikeTimes = []
        # Iterate over trials
        for k in range(self.p.trials):
            # Add spike times only when i != k
            apply = np.all([combinations[k, m] != i for m in range(self.p.patchMissingNeurons)])
            # If patch neuron indices are given, add spikes to all patch neurons
            if idc is not None:
                apply = True

            if (apply):
                spks = np.arange(self.p.inputSteps) + self.p.stepsPerTrial*k + self.p.resetOffset*(k+1)
                spikeTimes.append(spks)
    
        spikeTimes = list(itertools.chain(*spikeTimes))
        
        # Iterate over generators
        for j in range(self.p.patchGensPerNeuron):
            # Add spikes indices to current spike generator
            sg.addSpikes(spikeInputPortNodeIds=cnt, spikeTimes=spikeTimes)
            # Increase counter
            cnt += 1

        # Add spike indices to patchSpikes array
        self.patchSpikes.append(spikeTimes)

    if idc is None:
        patchSize = int(np.sqrt(self.p.patchNeurons))
        exNeuronsTopSize = int(np.sqrt(self.p.reservoirExSize))

        #patchMask = np.zeros((patchSize, patchSize))
        #patchMask[self.p.patchSize:, :] = 0  # set all mas values behind last neuron of patch input to zero

        # Set all values zero which are not part of the patch
        shiftX = self.p.patchNeuronsShiftX
        shiftY = self.p.patchNeuronsShiftY
        #shiftX = 44
        #shiftY = 24
        topology = np.zeros((exNeuronsTopSize,exNeuronsTopSize))
        topology[shiftY:shiftY+patchSize,shiftX:shiftX+patchSize] = 1
        #topology[0,0] = 1
        idc = np.where(topology.flatten())[0]

        # In every trial remove another 
        #self.idc = idc
    
    # Store patch neurons
    self.patchNeurons = idc

    # Define mask for connections
    patchMask = np.zeros((self.p.reservoirExSize, patchGens))
    for i, idx in enumerate(idc):
        fr, to = i*self.p.patchGensPerNeuron, (i+1)*self.p.patchGensPerNeuron
        patchMask[idx,fr:to] = 1

    # Define weights
    self.patchWeights = patchMask*self.p.patchMaxWeight

    # Connect generator to the reservoir network
    for i in range(len(self.exReservoirChunks)):
        fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
        ma = patchMask[fr:to, :]
        we = self.patchWeights[fr:to, :]
        sg.connect(self.exReservoirChunks[i], prototype=self.exConnProto, connectionMask=ma, weight=we)

    # Log that patch generator was added
    logging.info('Cue generator was added to the network')

"""
@desc: Create input spiking generator to add a trace signal,
        the input is connected to a share of the reservoir network,
        an excitatory connection prototype is used
"""
def addTraceGenerator(self, clusterIdx):
    start = clusterIdx*self.p.traceSteps
    # Create spike generator
    sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.traceGens)

    traceSpikes = []
    for i in range(self.p.traceGens):
        # Generate spikes for one training step
        #traceSpikesInd = generateSinSignal(self.p.traceSteps, start)
        traceSpikesInd = generateInputSignal(self.p.traceSteps, prob=self.p.traceSpikeProb, start=start)

        # Multiply spikes to all training steps
        spikeRange = range(0, self.p.totalSteps, self.p.totalTrialSteps)
        traceSpikesInds = np.ndarray.flatten(np.array([traceSpikesInd + i for i in spikeRange]))
        
        # Add all spikes to spike generator
        sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=traceSpikesInds.tolist())

        # Store trace input in object
        traceSpikes.append(traceSpikesInds)
    
    self.traceSpikes.append(np.array(traceSpikes))

    # Define start and end neuron
    startNeuron = clusterIdx*self.p.traceClusterSize
    endNeuron = startNeuron+(self.p.traceClusterSize-1)

    # Define mask matrix
    traceMask = np.zeros((self.p.reservoirExSize, self.p.traceGens)).astype(int)
    traceMask[startNeuron:endNeuron, :] = 1
    traceMask = sparse.csr_matrix(traceMask)

    # Draw weigh matrix based on mask matrix for trace input
    #traceWeights = self.p.traceMaxWeight*np.random.rand(self.p.reservoirExSize, self.p.traceGens)
    traceWeights = self.drawSparseWeightMatrix(traceMask)

    # Connect generator to the excitatory reservoir network
    for i in range(len(self.exReservoirChunks)):
        fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
        ma = traceMask[fr:to, :].toarray()
        we = traceWeights[fr:to, :].toarray()
        sg.connect(self.exReservoirChunks[i], prototype=self.exConnProto, connectionMask=ma, weight=we)

    self.traceMasks.append(traceMask)
    #self.traceWeights.append(self.getMaskedWeights(traceWeights, traceMask))
    self.traceWeights.append(traceWeights)