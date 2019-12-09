import numpy as np
import nxsdk.api.n2a as nx
import matplotlib.pyplot as plt
from scipy import sparse
from types import SimpleNamespace
import logging
import itertools

# Importing own modules
from ...utils import Utils
from ...plots import Plot
from ...parameters import Parameters

# Importing additional class from this module
from .._basic import BasicNetwork

"""
@desc: Reservoir network
"""
class ReservoirNetwork(BasicNetwork):

    """
    @desc: Initiates the innate network
    """
    def __init__(self, parameters=None):
        # Get parameters
        self.p = Parameters() if parameters is None else parameters

        # Call super init
        super().__init__()

        """
        Network objects
        """
        # Weights
        self.initialMasks = SimpleNamespace(**{
            'exex': None, 'inin': None, 'inex': None, 'exin': None
        })

        self.initialWeights = SimpleNamespace(**{
            'exex': None, 'inin': None, 'inex': None, 'exin': None
        })
        self.trainedWeightsExex = None

        # NxSDK compartment group chunks
        self.exReservoirChunks = []
        self.inReservoirChunks = []
        self.connectionChunks = []

        # Probes
        self.exSpikeProbes = []
        self.inSpikeProbes = []
        self.exVoltageProbes = []
        self.inVoltageProbes = []
        self.exCurrentProbes = []
        self.inCurrentProbes = []
        self.weightProbes = []

        # Spikes
        self.exSpikeTrains = []
        self.inSpikeTrains = []

        # Trace input
        self.traceSpikes = []
        self.traceMasks = []
        self.traceWeights = []

        # Cue input
        self.cueSpikes = []
        self.cueWeights = None

        # Noise input spikes
        self.noiseSpikes = None
        self.noiseWeights = None

        # Instantiate utils and plot
        self.utils = Utils.instance()
        self.plot = Plot(self)

    """
    @desc: Run the network
    """
    def run(self):
        super().run()

        # Post processing of probes
        self.postProcessing()

        #if self.log:
        #    self.plot.reservoirSpikeTrain()  # Plot spike train of reservoir network
        #    self.plot.reservoirRates()  # Plot avergae rates of reservoir network
        #    self.plot.noiseSpikes()  # Plot spikes of noise neurons

    """
    @desc: Build default network structure
    """
    def build(self):
        # Add probes
        self.addProbes()

        # Add spike receiver
        #self.

    """
    @desc: Post processing of probes
    """
    def postProcessing(self):
        # Calculate spikes from probes activities
        #self.exSpikeTrains = self.utils.getSpikesFromActivity(self.exActivityProbes)
        #self.inSpikeTrains = self.utils.getSpikesFromActivity(self.inActivityProbes)

        # Combine spike probes from all chunks together for excitatory neurons
        if self.p.isExSpikeProbe:
            spks = []
            for i in range(len(self.exSpikeProbes)):
                spks.append(self.exSpikeProbes[i].data)
            self.exSpikeTrains = np.reshape(spks, (self.p.reservoirExSize, self.p.totalSteps))

        # Combine spike probes from all chunks together for inhibitory neurons
        if self.p.isInSpikeProbe:
            spks = []
            for i in range(len(self.inSpikeProbes)):
                spks.append(self.inSpikeProbes[i].data)
            self.inSpikeTrains = np.reshape(spks, (self.p.reservoirInSize, self.p.totalSteps))

        # Recombine all weights from probe chunks together to a matrix again
        if self.p.weightProbe:
            self.trainedWeightsExex = self.utils.recombineExWeightMatrix(self.initialWeights.exex, self.weightProbes)

        # Log that post processing has finished
        logging.info('Post processing succesfully completed')

    """
    @desc: Adds a generator which produces random spikes and
           connects it to the excitatory reservoir neurons
    """
    def addNoiseGenerator(self):
        # Create spike generator
        sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.noiseNeurons)

        # Create random spikes
        noise = np.random.rand(self.p.noiseNeurons, self.p.totalSteps)
        noise[noise < (1-self.p.noiseSpikeprob)] = 0
        noise[noise >= (1-self.p.noiseSpikeprob)] = 1

        # Store spikes in object
        self.noiseSpikes = noise.astype(int)

        # Add spike times to spike generator
        for i in range(self.p.noiseNeurons):
            spikes = np.where(self.noiseSpikes[i,:])[0].tolist()
            sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=spikes)

        # Create mask for noise/reservoir connections
        noiseMask = self.drawSparseMaskMatrix(self.p.noiseDens, self.p.reservoirExSize, self.p.noiseNeurons)

        # Create weights for noise/reservoir connections between -noiseMaxWeight and +noiseMaxWeight
        randoms = ((np.random.rand(self.p.reservoirExSize, self.p.noiseNeurons)*2*self.p.noiseMaxWeight) - self.p.noiseMaxWeight)
        self.noiseWeights = sparse.csr_matrix(np.round(self.p.noiseMaxWeight*randoms).astype(int))
        #sign = np.random.rand(self.p.reservoirExSize, self.p.noiseNeurons)
        #sign[sign < 0.5] = -1
        #sign[sign >= 0.5] = 1
        #self.noiseWeights = self.drawSparseWeightMatrix(noiseMask).multiply(sign).tocsr()

        # Connect noise network to the reservoir network
        for i in range(len(self.exReservoirChunks)):
            fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
            ma = noiseMask[fr:to, :].toarray()
            we = self.noiseWeights[fr:to, :].toarray()
            sg.connect(self.exReservoirChunks[i], prototype=self.mixedConnProto, connectionMask=ma, weight=we)
        
        # Log that background noise was added
        logging.info('Background noise was added to the network')

    """
    TODO: Shift to utils/misc.py
    @desc: Generates a sinus signal (one 'hill') in given length of time steps
    """
    def generateSinSignal(self, length, start=0):
        # Draw from a sin wave from 0 to 3,14 (one 'hill')
        probCoeff = 1 #1.5  # the higher the probability coefficient, the more activity is in the network
        probs = probCoeff * np.abs(np.sin((np.pi / length) * np.arange(length)))
        randoms = np.random.rand(length)

        # Get indices of spike
        spikeInd = np.where(randoms < probs)[0]
        # Shift sin signal by 'start' and return spikes
        return (spikeInd + start)

    """
    TODO: Shift to utils/misc.py
    @desc: Generates a simple input signal
           Respects tEpoch of STDP learning rule
    """
    def generateInputSignal(self, length, prob=0.1, start=0):
        spikes = np.zeros((length, 1))
        refCtr = 0  # initialize refractory value
        randoms = np.random.rand(length)

        for i in range(length):
            spikes[i] = (randoms[i] < prob) and refCtr <= 0
            # After spike, set refractory value
            #if spikes[i]:
            #    refCtr = self.p.learningEpoch + 1
            # Reduce refractory value by one
            #refCtr -= 1

        # Get indices of spike
        spikeTimes = np.where(spikes)[0]
        # Shift sin signal by 'start' and return spikes
        return (spikeTimes + start)

    """
    @desc: Adds a very strong inhibitory input to kill all network activity at a given time step
    """
    def addStopGenerator(self,):
        sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.stopGens)

        for i in range(self.p.stopGens):
            # Generate spikes and add them to spike generator
            stopSpikesInd = np.where(np.random.rand(self.p.cueSteps) < self.p.stopSpikeProb)[0] + self.p.stopStart
            sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=stopSpikesInd.tolist())
        
        # Define mask
        stopMask = np.ones((self.p.reservoirExSize, self.p.stopGens))

        # Define weights
        stopWeights = -1*np.ones((self.p.reservoirExSize, self.p.stopGens))*255

        for i in range(len(self.exReservoirChunks)):
            fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
            ma = stopMask[fr:to, :]
            we = stopWeights[fr:to, :]
            sg.connect(self.exReservoirChunks[i], prototype=self.inConnProto, connectionMask=ma, weight=we)
        
        # Log that stop generator was added
        logging.info('Stop generator was added to the network')

    """
    @desc: Adds a very strong inhibitory input to kill all network activity at a given time step
    """
    def addRepeatedStopGenerator(self,):
        sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.stopGens)

        for i in range(self.p.stopGens):
            # Generate spikes 
            spikes = (np.random.rand(self.p.cueSteps) < self.p.stopSpikeProb)
            # Get indices from spikes
            stopSpikesInd = [ np.where(spikes)[0] + self.p.stopStart + self.p.trialSteps*j for j in range(self.p.trials) ]
            # Add spikes indices to spike generator
            sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=list(itertools.chain(*stopSpikesInd)))
        
        # Define mask
        stopMask = np.ones((self.p.reservoirExSize, self.p.stopGens))

        # Define weights
        stopWeights = -1*np.ones((self.p.reservoirExSize, self.p.stopGens))*255

        for i in range(len(self.exReservoirChunks)):
            fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
            ma = stopMask[fr:to, :]
            we = stopWeights[fr:to, :]
            sg.connect(self.exReservoirChunks[i], prototype=self.inConnProto, connectionMask=ma, weight=we)

        # Log that stop generator was added
        logging.info('Stop generator was added to the network')

    """
    @desc: Create input spiking generator to add a cue signal,
           the input is connected to the reservoir network,
           an excitatory connection prototype is used
    """
    def addCueGenerator(self, inputSpikes = None):
        # Create spike generator
        sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.cueGens)

        cueSpikes = []
        for i in range(self.p.cueGens):
            # Generate spikes and add them to spike generator
            cueSpikesInd = None
            # If input spikes are not given
            if inputSpikes is None:
                cueSpikesInd = self.generateInputSignal(self.p.cueSteps, prob=self.p.cueSpikeProb) #self.generateSinSignal(self.p.cueSteps)
                # Store cue input in object
                cueSpikes.append(cueSpikesInd)
            # If input spikes are given
            else:
                cueSpikesInd = inputSpikes[i]
            
            sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=cueSpikesInd.tolist())

        if len(self.cueSpikes) == 0:
            self.cueSpikes = np.array(cueSpikes)  # If train, store generated spikes

        # Define mask
        cueMask = self.drawSparseMaskMatrix(self.p.cueDens, self.p.reservoirExSize, self.p.cueGens)

        cueSize = int(np.sqrt(self.p.cuePatchNeurons))
        exNeuronsTopSize = int(np.sqrt(self.p.reservoirExSize))

        #cueMask = np.zeros((cueSize, cueSize))
        #cueMask[self.p.cueSize:, :] = 0  # set all mas values behind last neuron of cue input to zero

        # Set all values zero which are not part of the patch
        shift = 20
        topology = np.ones((exNeuronsTopSize,exNeuronsTopSize))
        topology[shift:shift+cueSize,shift:shift+cueSize] = 0
        #topology[0,0] = 1
        idc = np.where(topology.flatten())[0]
        cueMask[idc,:] = 0

        # Define weights
        cueWeights = self.p.cueMaxWeight*np.random.rand(self.p.reservoirExSize, self.p.cueGens)

        # Connect generator to the reservoir network
        for i in range(len(self.exReservoirChunks)):
            fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
            ma = cueMask[fr:to, :]
            we = cueWeights[fr:to, :]
            sg.connect(self.exReservoirChunks[i], prototype=self.exConnProto, connectionMask=ma, weight=we)
        
        self.cueWeights = self.getMaskedWeights(cueWeights, cueMask)

        # Log that cue generator was added
        logging.info('Cue generator was added to the network')

    """
    @desc: Create input spiking generator to add a cue signal,
           the input is connected to the reservoir network,
           an excitatory connection prototype is used
    """
    def addRepeatedCueGenerator(self):
        # Create spike generator
        sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.cueGens)

        for i in range(self.p.cueGens):
            # Generate spikes for spike current generator
            spikes = (np.random.rand(self.p.cueSteps) < self.p.cueSpikeProb)
            # Get indices from spikes
            cueSpikesInd = []
            for j in range(self.p.trials):
                # Draw neurons to flip with probability flipProb
                flips = (np.random.rand(self.p.cueSteps) < self.p.flipProb)
                # Apply flips to cue input
                noisedSpikes = np.logical_xor(spikes, flips)
                # Transform to event indices
                noisedIndices = np.where(noisedSpikes)[0] + self.p.trialSteps*j
                cueSpikesInd.append(noisedIndices)

            self.cueSpikes.append(list(itertools.chain(*cueSpikesInd)))
                
            # Add spikes indices to current spike generator
            sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=list(itertools.chain(*cueSpikesInd)))

        # Define mask
        cueMask = self.drawSparseMaskMatrix(self.p.cueDens, self.p.reservoirExSize, self.p.cueGens)

        cueSize = int(np.sqrt(self.p.cuePatchNeurons))
        exNeuronsTopSize = int(np.sqrt(self.p.reservoirExSize))

        #cueMask = np.zeros((cueSize, cueSize))
        #cueMask[self.p.cueSize:, :] = 0  # set all mas values behind last neuron of cue input to zero

        # Set all values zero which are not part of the patch
        shift = 0
        topology = np.ones((exNeuronsTopSize,exNeuronsTopSize))
        topology[shift:shift+cueSize,shift:shift+cueSize] = 0
        #topology[0,0] = 1
        idc = np.where(topology.flatten())[0]
        cueMask[idc,:] = 0

        # Define weights
        cueWeights = self.p.cueMaxWeight*np.random.rand(self.p.reservoirExSize, self.p.cueGens)

        # Connect generator to the reservoir network
        for i in range(len(self.exReservoirChunks)):
            fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
            ma = cueMask[fr:to, :]
            we = cueWeights[fr:to, :]
            sg.connect(self.exReservoirChunks[i], prototype=self.exConnProto, connectionMask=ma, weight=we)
        
        self.cueWeights = self.getMaskedWeights(cueWeights, cueMask)

        # Log that cue generator was added
        logging.info('Cue generator was added to the network')

    """
    @desc: Create input spiking generator to add a trace signal,
           the input is connected to a share of the reservoir network,
           an excitatory connection prototype is used
    """
    def addTraceGenerator(self, clusterIdx):
        start = self.p.offset + clusterIdx*self.p.traceSteps
        # Create spike generator
        sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.traceGens)

        traceSpikes = []
        for i in range(self.p.traceGens):
            # Generate spikes for one training step
            #traceSpikesInd = self.generateSinSignal(self.p.traceSteps, start)
            traceSpikesInd = self.generateInputSignal(self.p.traceSteps, prob=self.p.traceSpikeProb, start=start)

            # Multiply spikes to all training steps
            spikeRange = range(0, self.p.totalSteps, self.p.stepsPerIteration)
            traceSpikesInds = np.ndarray.flatten(np.array([traceSpikesInd + i for i in spikeRange]))
            
            # Add all spikes to spike generator
            sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=traceSpikesInds.tolist())

            # Store trace input in object
            traceSpikes.append(traceSpikesInds)
        
        self.traceSpikes.append(np.array(traceSpikes))

        # Connect generator to the reservoir network
        startNeuron = clusterIdx*self.p.traceClusterSize + self.p.constSize
        endNeuron = startNeuron+(self.p.traceClusterSize-1)

        #traceMask = np.zeros((self.p.reservoirExSize, self.p.traceGens)).astype(int)
        #traceMask[startNeuron:endNeuron, :] = 1
        #traceMask = sparse.csr_matrix(traceMask)
        traceMask = self.drawSparseMaskMatrix(self.p.traceDens, self.p.reservoirExSize, self.p.traceGens)
        traceMask[endNeuron:, :] = 0  # set all mas values behind last neuron of cluster to zero

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

    """
    @desc: Create input spiking generator to add a constant signal,
           the input is connected to a share of the reservoir network,
           an excitatory connection prototype is used
    """
    def addConstantGenerator(self):
        # Create spike generator
        sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.constGens)

        constSpikes = []
        for i in range(self.p.constGens):
            # Generate spikes for one training step
            constSpikesInd = self.generateInputSignal(self.p.totalSteps, prob=self.p.constSpikeProb, start=0)
            
            # Add all spikes to spike generator
            sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=constSpikesInd)

            # Store const input in object
            constSpikes.append(constSpikesInd)
        
        self.constSpikes.append(np.array(constSpikes))

        # Connect generator to the reservoir network
        startNeuron = 0
        endNeuron = (self.p.constSize-1)

        # Sample mask for constant input
        constMask = self.drawSparseMaskMatrix(self.p.constDens, self.p.reservoirExSize, self.p.constGens)
        constMask[endNeuron:, :] = 0  # set all mask values behind last neuron of cluster to zero

        # Sample weights for constant input
        constWeights = self.drawSparseWeightMatrix(constMask)

        # Connect generator to the excitatory reservoir network
        for i in range(len(self.exReservoirChunks)):
            fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
            ma = constMask[fr:to, :].toarray()
            we = constWeights[fr:to, :].toarray()
            sg.connect(self.exReservoirChunks[i], prototype=self.exConnProto, connectionMask=ma, weight=we)

        #self.constMasks.append(constMask)
        #self.constWeights.append(constWeights)

    """
    @desc: Add probing
    """
    def addProbes(self):
        # Add voltage probe for excitatory network
        if self.p.isExVoltageProbe:
            for idx, net in enumerate(self.exReservoirChunks):
                self.exVoltageProbes.append(net.probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE ])[0])

        # Add voltage probe for inhibitory network
        if self.p.isInVoltageProbe:
            for idx, net in enumerate(self.inReservoirChunks):
                self.inVoltageProbes.append(net.probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE ])[0])

        # Add current probe for excitatory network
        if self.p.isExCurrentProbe:
            for idx, net in enumerate(self.exReservoirChunks):
                self.exCurrentProbes.append(net.probe([nx.ProbeParameter.COMPARTMENT_CURRENT ])[0])

        # Add current probe for inhibitory network
        if self.p.isInCurrentProbe:
            for idx, net in enumerate(self.inReservoirChunks):
                self.inCurrentProbes.append(net.probe([nx.ProbeParameter.COMPARTMENT_CURRENT ])[0])
        
        # Probe weights
        if self.p.weightProbe:
            probeCond = nx.IntervalProbeCondition(tStart=self.p.totalSteps-1, dt=self.p.totalSteps)
            #probeCond = nx.IntervalProbeCondition(tStart=self.p.stepsPerIteration-1, dt=self.p.stepsPerIteration)
            n, m = np.shape(self.connectionChunks)
            for i in range(n):
                tmp = []
                for j in range(m):
                    tmp.append(self.connectionChunks[i][j].probe([nx.ProbeParameter.SYNAPSE_WEIGHT], probeConditions=[probeCond]))
                self.weightProbes.append(tmp)

        # Add spike probe for excitatory network
        if self.p.isExSpikeProbe:
            #probeCond = nx.SpikeProbeCondition(tStart=1, dt=5)
            for net in self.exReservoirChunks:
                self.exSpikeProbes.append(net.probe([nx.ProbeParameter.SPIKE])[0])#, probeConditions=[probeCond])[0])

        # Add spike probe for excitatory network
        if self.p.isInSpikeProbe:
            #probeCond = nx.SpikeProbeCondition(tStart=self.p.cueSteps, dt=5)
            for net in self.inReservoirChunks:
                self.inSpikeProbes.append(net.probe([nx.ProbeParameter.SPIKE])[0])#, probeConditions=[probeCond])[0])

        # Log that probes were added to network
        logging.info('Probes added to Loihi network')

    """
    @desc: Connects reservoir neurons
    """
    def addReservoirNetworkDistributed(self):
        # Predefine some helper variables
        nEx = self.p.reservoirExSize
        nIn = self.p.reservoirInSize

        nExCores = int(np.ceil(nEx / self.p.neuronsPerCore))
        nLastExCore = nEx % self.p.neuronsPerCore
        nInCores = int(np.ceil(nIn / self.p.neuronsPerCore))
        nLastInCore = nIn % self.p.neuronsPerCore
        nAllCores = nExCores + nInCores

        exConnProto = None
        # Create learning rule
        if self.p.isLearningRule:
            # Define learning rule
            lr = self.nxNet.createLearningRule(dw=self.p.learningRule, tEpoch=self.p.learningEpoch,
                                               x1Impulse=self.p.learningImpulse, x1TimeConstant=self.p.learningTimeConstant,
                                               y1Impulse=self.p.learningImpulse, y1TimeConstant=self.p.learningTimeConstant)
            # Define connection prototype with learning rule
            exConnProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                 enableLearning=1, learningRule=lr)
        else:
            # Define connection prototype from basic network
            exConnProto = self.exConnProto

        # Define variable to enable/disable spike backprop for learning
        isBackprop = int(self.p.isLearningRule)
        # Define compartment prototypes and compartmen groups
        for i in range(nAllCores):
            if i < nExCores:
                # Excitatory compartment prototype
                exCompProto = nx.CompartmentPrototype(compartmentVoltageDecay=self.p.compartmentVoltageDecay,
                                                      refractoryDelay=self.p.refractoryDelay, logicalCoreId=i,
                                                      enableSpikeBackprop=isBackprop, enableSpikeBackpropFromSelf=isBackprop)#,
                                                      #enableHomeostasis=1, minActivity=0, maxActivity=127,
                                                      #homeostasisGain=0, activityImpulse=1, activityTimeConstant=1000000)
                # Calculate size of compartment: if last core has remainder, use remainder as size
                size = nLastExCore if (i == (nExCores-1) and nLastExCore > 0) else self.p.neuronsPerCore
                # Excitatory compartment group
                self.exReservoirChunks.append(self.nxNet.createCompartmentGroup(size=size, prototype=exCompProto))
            elif i >= nExCores:
                # Inhibitory compartment prototype
                inCompProto = nx.CompartmentPrototype(compartmentVoltageDecay=self.p.compartmentVoltageDecay,
                                                      refractoryDelay=self.p.refractoryDelay, logicalCoreId=i)#,
                                                      #enableHomeostasis=1, minActivity=0, maxActivity=127,
                                                      #homeostasisGain=0, activityImpulse=1, activityTimeConstant=1000000)
                # Calculate size of compartment: if last core has remainder, use remainder as size
                size = nLastInCore if (i == (nAllCores-1) and nLastInCore > 0) else self.p.neuronsPerCore
                # Inhibitory compartment prototype
                self.inReservoirChunks.append(self.nxNet.createCompartmentGroup(size=size, prototype=inCompProto))

        # Interconnect excitatory and inhibitory network chunks
        self.connectNetworkChunks(fromChunks=self.exReservoirChunks, toChunks=self.exReservoirChunks, mask=self.initialMasks.exex, weights=self.initialWeights.exex, prototype=exConnProto) #store=True, prototype=exConnProto)
        self.connectNetworkChunks(fromChunks=self.inReservoirChunks, toChunks=self.inReservoirChunks, mask=self.initialMasks.inin, weights=self.initialWeights.inin, prototype=self.inConnProto)
        self.connectNetworkChunks(fromChunks=self.exReservoirChunks, toChunks=self.inReservoirChunks, mask=self.initialMasks.exin, weights=self.initialWeights.exin, prototype=self.exConnProto)
        self.connectNetworkChunks(fromChunks=self.inReservoirChunks, toChunks=self.exReservoirChunks, mask=self.initialMasks.inex, weights=self.initialWeights.inex, prototype=self.inConnProto)

        # Log that all cores are interconnected
        logging.info('All cores are sucessfully interconnected')

    """
    @desc: Interconnect all network chunks (basically interconnects cores)
    """
    def connectNetworkChunks(self, fromChunks, toChunks, mask, weights, store=False, **connectionArgs):
        nCoresFrom = len(fromChunks)
        nCoresTo = len(toChunks)
        nPerCore = self.p.neuronsPerCore

        for i in range(nCoresFrom):
            # Get indices for chunk from outer loop
            ifr, ito = i*nPerCore, (i+1)*nPerCore

            tmp = []
            for j in range(nCoresTo):
                # Get indices for chunk from inner loop
                jfr, jto = j*nPerCore, (j+1)*nPerCore

                # Define chunk from sparse matrix and transform to numpy array
                ma = mask[jfr:jto, ifr:ito].toarray()
                we = weights[jfr:jto, ifr:ito].toarray()

                # Connect network chunks
                conn = fromChunks[i].connect(toChunks[j], connectionMask=ma, weight=we, **connectionArgs)
                if store:
                    tmp.append(conn)
            
            if store:
                self.connectionChunks.append(tmp)

    """
    TODO: Shifted to utils/weights.py
    @desc: Get mask of support weights for every cluster in the assembly
    @return: Mask of the bottom-left area of the matrix
    """
    # def getSupportWeightsMask(self):
    #     nCs = self.p.traceClusterSize
    #     nEx = self.p.reservoirExSize
    #     nC = self.p.traceClusters
    #     matrix = self.initialWeights.exex

    #     # Get areas in matrix
    #     left = matrix[:,:nC*nCs]  # left
    #     top = matrix[:nC*nCs,:]  # top
    #     bottom = matrix[nC*nCs:,:]  # bottom
    #     bottomLeft = matrix[nC*nCs:,:nC*nCs]  # bottom-left

    #     # Get single cluster colums in bottom-left area (candidates for support weights)
    #     cols = np.array([ bottomLeft[:,i*nCs:(i+1)*nCs] for i in range(nC)])

    #     # Calculate means for every column in bottom-left
    #     col_rowmeans = np.array([np.mean(cols[i,...], axis=1) for i in range(nC)])

    #     # Condition 1: Get only rows their mean is greater than total mean
    #     greaterMeanIndices = col_rowmeans > np.mean(bottomLeft)

    #     # Condition 2: Get for every row the column which has max value
    #     col_argmax = np.argmax(col_rowmeans, axis=0)
    #     maxRowIndices = np.array(col_argmax[:,None] == range(nC)).T

    #     # Get final mask in combining both conditions
    #     return np.logical_and(greaterMeanIndices, maxRowIndices)
