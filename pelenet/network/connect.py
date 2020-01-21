import nxsdk.api.n2a as nx
import numpy as np
from scipy import sparse
import logging

"""
@desc: Connects reservoir neurons
"""
def addReservoirNetworkDistributed(self):
    # Predefine some helper variables
    nEx = self.p.reservoirExSize
    nIn = self.p.reservoirInSize
    nOut = self.p.numOutputNeurons

    nExCores = int(np.ceil(nEx / self.p.neuronsPerCore))
    nLastExCore = nEx % self.p.neuronsPerCore  # number of excitatory neurons in last core
    nInCores = int(np.ceil(nIn / self.p.neuronsPerCore))
    nLastInCore = nIn % self.p.neuronsPerCore  # number of inhibitory neurons in last core
    nOutCores = int(np.ceil(nOut / self.p.neuronsPerCore))
    nLastOutCore = nOut % self.p.neuronsPerCore  # number of ouput neurons in last core
    nAllCores = nExCores + nInCores + nOutCores

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
                                                    enableSpikeBackprop=isBackprop, enableSpikeBackpropFromSelf=isBackprop)
            # Calculate size of compartment: if last core has remainder, use remainder as size
            size = nLastExCore if (i == (nExCores-1) and nLastExCore > 0) else self.p.neuronsPerCore
            # Excitatory compartment group
            self.exReservoirChunks.append(self.nxNet.createCompartmentGroup(size=size, prototype=exCompProto))
        elif i >= nExCores and i < nExCores+nInCores:
            # Inhibitory compartment prototype
            inCompProto = nx.CompartmentPrototype(compartmentVoltageDecay=self.p.compartmentVoltageDecay,
                                                    refractoryDelay=self.p.refractoryDelay, logicalCoreId=i)
            # Calculate size of compartment: if last core has remainder, use remainder as size
            size = nLastInCore if (i == (nExCores+nInCores-1) and nLastInCore > 0) else self.p.neuronsPerCore
            # Inhibitory compartment prototype
            self.inReservoirChunks.append(self.nxNet.createCompartmentGroup(size=size, prototype=inCompProto))
        elif i >= nExCores+nInCores:
            # Output compartment prototype
            outCompProto = nx.CompartmentPrototype(compartmentVoltageDecay=self.p.compartmentVoltageDecay,
                                                    refractoryDelay=self.p.refractoryDelay, logicalCoreId=i)
            # Calculate size of compartment: if last core has remainder, use remainder as size
            size = nLastOutCore if (i == (nAllCores-1) and nLastOutCore > 0) else self.p.neuronsPerCore
            # Inhibitory compartment prototype
            self.outputLayerChunks.append(self.nxNet.createCompartmentGroup(size=size, prototype=outCompProto))

    # Interconnect excitatory and inhibitory network chunks
    connectNetworkChunks(self, fromChunks=self.exReservoirChunks, toChunks=self.exReservoirChunks, mask=self.initialMasks.exex, weights=self.initialWeights.exex, prototype=exConnProto) #store=True, prototype=exConnProto)
    connectNetworkChunks(self, fromChunks=self.inReservoirChunks, toChunks=self.inReservoirChunks, mask=self.initialMasks.inin, weights=self.initialWeights.inin, prototype=self.inConnProto)
    connectNetworkChunks(self, fromChunks=self.exReservoirChunks, toChunks=self.inReservoirChunks, mask=self.initialMasks.exin, weights=self.initialWeights.exin, prototype=self.exConnProto)
    connectNetworkChunks(self, fromChunks=self.inReservoirChunks, toChunks=self.exReservoirChunks, mask=self.initialMasks.inex, weights=self.initialWeights.inex, prototype=self.inConnProto)

    # Connect excitatory neurons to output layer
    #connectNetworkChunks(self, fromChunks=self.exReservoirChunks, toChunks=self.outputLayerChunks, mask=self.outputMask, weights=self.outputWeights, prototype=self.exConnProto)

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
