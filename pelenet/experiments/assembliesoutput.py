# Official modules
import numpy as np
from sklearn.linear_model import RidgeClassifier

# Pelenet modules
from ._abstract import Experiment
from ..network import ReservoirNetwork

"""
@desc: Train output on support neuron activity from assemblies
"""
class AssemblyOutputExperiment(Experiment):

    def defineParameters(self):
        """
        Define parameters for this experiment
        """
        return {}
    
    
    def build(
            self,
            supportMask=None,
            mask=None, weights=None,
            assemblyIndex=0,
            inputSpikeIndices=[],
            targetNeuronIndices=[]
        ):
        """
        Overwrite build reservoir network with given mask, weights and input

        Parameters
        ----------
        supportMask :
            The last support mask of the assembly organization
        mask :
        weights :
        assemblyIndex :
            The index of the assembly where 0: A, 1: B, etc.
        inputSpikeIndices :
        targetNeuronIndices : 
        """
        # Define variables for the experiment
        self.assemblyIndex = assemblyIndex
        self.supportMask = supportMask

        # Instanciate innate network
        self.net = ReservoirNetwork(self.p)
        
        # Define mask and weights
        if (mask is not None and weights is not None):
            # Set mask and weights
            self.net.initialMasks = mask
            self.net.initialWeights = weights
        else:
            # Throw an error if only one of mask/weights is defiend
            raise Exception("It is not possible to define only one of mask and weights, both must be defined.")

        # Connect ex-in reservoir
        self.net.connectReservoir()

        # Add input
        if len(targetNeuronIndices) == 0:
            targetNeuronIndices = np.arange(self.p.inputNumTargetNeurons) + assemblyIndex*self.p.inputNumTargetNeurons
        self.net.addInput(inputSpikeIndices=inputSpikeIndices, targetNeuronIndices=targetNeuronIndices)

        # Add background noise
        if self.p.isNoise:
            self.net.addNoiseGenerator()

        # Add Probes
        self.net.addProbes()

        # Call afterBuild
        self.afterBuild()

    def getInputIdcs(self, dataset):
        """
        Get indices of spikes for input generators
        """
        spikeIdcs = []
        # Iterate over target neurons
        for i in range(dataset.shape[1]):

            spikeTimes = []
            # Iterate over trials
            for k in range(dataset.shape[0]):
                off = self.p.inputOffset + self.p.stepsPerTrial*k + self.p.resetOffset*(k+1)
                spks = off + np.where(dataset[k,i,:])[0]
                spikeTimes.extend(spks.tolist())

            spikeIdcs.append(spikeTimes)
        return spikeIdcs

    def loadYinYangAsInputs(self, nAssemblies=2, nTrain=500, nTest=100):
        """
        Loads the Yin Yang dataset and transforms it to an input for the reservoir network
        """

        # TODO add dot class

        # Define variables
        self.nAssemblies = nAssemblies
        self.nTrain = nTrain
        self.nTest = nTest

        # Load raw data
        yin_train = np.load('data/yinyang/inputs_yin_train.npy')  # 1000
        yang_train = np.load('data/yinyang/inputs_yang_train.npy')  # 1000
        dots_train = np.load('data/yinyang/inputs_dots_train.npy')  # 1000
        yin_test = np.load('data/yinyang/inputs_yin_test.npy')  # 200
        yang_test = np.load('data/yinyang/inputs_yang_test.npy')  # 200
        dots_test = np.load('data/yinyang/inputs_dots_test.npy')  # 200

        # Check if requested data is avialable
        if (nTrain > yin_train.shape[0]) or (nTrain > yang_train.shape[0]):
            raise Exception('The training dataset has fewer samples than requested.')
        if (nTest > yin_test.shape[0]) or (nTest > yang_test.shape[0]):
            raise Exception('The test dataset has fewer samples than requested.')

        # Check if number of classes is available
        if (nAssemblies > 3):
            raise Exception('This dataset has maximum 3 classes available')

        # Compute total length
        nTrials = nAssemblies * (nTrain + nTest)

        # Concatenate datasets
        inputs = np.concatenate((
            yin_train[:nTrain],
            #yang_train[:nTrain],
            dots_train[:nTrain],
            yin_test[:nTest],
            #yang_test[:nTest],
            dots_test[:nTest]
        ), axis=0)
        
        # Transform to input for reservoir and return
        return self.getInputIdcs(inputs)


    def getDatasetFromSpikes(self):
        """
        Get Dataset from raw spiking data
        """
        
        # Get data from spikes
        data = []
        nIdx = self.p.inputNumTargetNeurons
        for i in range(self.p.trials):
            fr = i*self.p.totalTrialSteps + self.p.resetOffset + self.p.inputOffset
            to = (i+1)*self.p.totalTrialSteps
            # Get number of input neurons (all assemblies)
            nInput = self.p.inputNumTargetNeurons*len(self.p.inputVaryProbs)
            tmp = self.net.exSpikeTrains[nInput:,fr:to]
            data.append(tmp)

        # Convert to numpy array
        data = np.array(data)

        # Get support indices
        supportIndices = np.where(self.supportMask[self.assemblyIndex])[0]

        # Separate between train and test data
        dataTrain = data[:self.nAssemblies*self.nTrain,supportIndices,:]
        dataTest = data[self.nAssemblies*self.nTrain:,supportIndices,:]

        return dataTrain, dataTest

    def afterRun(self):
        """
        A lifecycle function called after the simulation has successfully finished

        Prepares the dataset
        """

        # Get dataset from spikes
        dataTrain, dataTest = self.getDatasetFromSpikes()
        self.dataTrain = dataTrain
        self.dataTest = dataTest

        # Store 2D spikes dataset
        sTra = self.dataTrain.shape
        self.spikesTrain = self.dataTrain.reshape((sTra[0], sTra[1]*sTra[2]))
        sTst = self.dataTest.shape
        self.spikesTest = self.dataTest.reshape((sTst[0], sTst[1]*sTst[2]))

        # Store rates dataset
        self.ratesTrain = np.sum(self.dataTrain, axis=1)
        self.ratesTest = np.sum(self.dataTest, axis=1)

        # Store frequencies dataset
        self.freqTrain = np.sum(self.dataTrain, axis=2)
        self.freqTest = np.sum(self.dataTest, axis=2)

        # Store labels
        self.labelsTrain = np.concatenate((-np.ones(self.nTrain), np.ones(self.nTrain)))
        self.labelsTest = np.concatenate((-np.ones(self.nTest), np.ones(self.nTest)))

    def fitRidgeClassifier(self):
        """
        Fit ridge classifier and show scores
        """
        self.clfSpikes = RidgeClassifier().fit(self.spikesTrain, self.labelsTrain)
        print('(spikes) score:', self.clfSpikes.score(self.spikesTest, self.labelsTest))

        self.clfRates = RidgeClassifier().fit(self.ratesTrain, self.labelsTrain)
        print('(rates) score:', self.clfRates.score(self.ratesTest, self.labelsTest))

        self.clfFreq = RidgeClassifier().fit(self.freqTrain, self.labelsTrain)
        print('(frequency) score:', self.clfFreq.score(self.freqTest, self.labelsTest))
