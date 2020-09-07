import numpy as np
import logging
import itertools
from scipy import sparse

"""
TODO: Shift to utils/misc.py
@desc: Generates a sinus signal (one 'hill') in given length of time steps
"""
def generateSinusInput(length, start=0):
    # Draw from a sin wave from 0 to 3,14 (one 'hill')
    probCoeff = 1 #1.5  # the higher the probability coefficient, the more activity is in the network
    probs = probCoeff * np.abs(np.sin((np.pi / length) * np.arange(length)))
    randoms = np.random.rand(length)

    # Get indices of spike
    spikeInd = np.where(randoms < probs)[0]

    # Shift sin signal by 'start' and return spikes
    return (spikeInd + start)

"""
@desc: Generates a simple input signal
"""
def generateUniformInput(length, prob=0.1, start=0):
    spikes = np.zeros((length, 1))
    randoms = np.random.rand(length)

    # Draw spikes
    for i in range(length):
        spikes[i] = (randoms[i] < prob)
    
    # Get indices of spike
    spikeTimes = np.where(spikes)[0]

    # Shift sin signal by 'start' and return spikes
    return (spikeTimes + start)

"""
@desc:  Create input spiking generator to add a patch signal,
        the input is connected to the reservoir network,
        an excitatory connection prototype is used
@params:
        targetNeuronIndices: indices of reservoir neurons to connect input to, default is None (indices are just taken successively)
"""
def addInput(self, inputSpikeIndices=[], targetNeuronIndices=[]):
    # Create spike generator
    sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.inputNumGenerators)

    # Draw spikes for input generators if not already given
    self.inputSpikes = drawSpikes(self) if not inputSpikeIndices else inputSpikeIndices

    # Add spikes s to generator i
    for i, s in enumerate(self.inputSpikes):
        sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=s.tolist())

    # Get inidices of target neurons if not already given
    self.inputTargetNeurons = getTargetNeurons(self) if not targetNeuronIndices else targetNeuronIndices

    # Connect spike generators to reservoir
    self.inputWeights = connectSpikeGenerator(self, sg)

    # Log that input was added
    logging.info('Input was added to the network')

"""
@desc: Draw spikes for spike generators
"""
def drawSpikes(self):
    # Initialize array for spike indices
    inputSpikes = []

    # Draw spikes for every spike generator
    for i in range(self.p.inputNumGenerators):
        s = []
        # Generate spikes, depending on input type
        if self.p.inputType == 'uniform':
            s = generateUniformInput(self.p.inputSteps, prob=self.p.inputGenSpikeProb, start=self.p.inputOffset)
        if self.p.inputType == 'sinus':
            s = generateSinusInput(self.p.inputSteps, start=self.p.inputOffset)
        # Store input spikes in array
        inputSpikes.append(s)

    return inputSpikes

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
        sX = self.p.inputTopologyShiftX
        sY = self.p.inputTopologyShiftY

        # Define input region in network topology and store their indices
        topology = np.zeros((exNeuronsEdge,exNeuronsEdge))
        topology[sY:sY+targetNeuronsEdge,sX:sX+targetNeuronsEdge] = 1
        targetNeurons = np.where(topology.flatten())[0]

    return targetNeurons
    
"""
@desc: Connect spike generators with target neurons of reservoir
       Draws mask and weights
"""
def connectSpikeGenerator(self, spikeGenerators):
    # Draw mask connections from generators to neurons
    inputMask = self.drawSparseMaskMatrix(p=self.p.inputMaskConnProb, nrows=self.p.reservoirExSize, ncols=self.p.inputNumGenerators, avoidSelf=False)

    # Set all non-targeted neurons to zero
    inputMask[np.delete(np.arange(self.p.inputNumGenerators), self.inputTargetNeurons),:] = 0

    # Draw weights
    #inputWeights = inputMask*self.p.patchMaxWeight
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
                spks = np.arange(self.p.inputSteps) + self.p.trialSteps*k + self.p.resetOffset*(k+1)
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
def addInputSequence(self, clusterIdx):
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