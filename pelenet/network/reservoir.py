"""
@desc: Run the network
"""
def run(self):
    self.nxNet.run(self.p.totalSteps)
    self.nxNet.disconnect()

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
    #pass
    # Add probes
    self.addProbes()

    # Add spike receiver
    #self.

"""
@desc: Adds a very strong inhibitory input to kill all network activity at a given time step
"""
def addStopGenerator(self):
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
