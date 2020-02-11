"""
@desc: Include parameters of the experiment
"""
def includeExperiment(self):

    # Optimization
    self.optimizeParameters = 'reservoirExSize'  # Parameter to optimize (TODO later extend to a list)

    # Experiment
    self.seed = 2  # Seed of the simulation, 'None' is random (no seed)
    self.totalSteps = None #1000 #None  # Number of simulation steps, if 'None', value is derived
    self.trials = 25  # number of trials
    self.movementSteps = 200  # Number of steps for movement

    # Ansitotropic
    self.connectionProb = 0.05 #0.04 #0.05  # percentage of established connections (range 0.05 - 0.1), FIXME bring together with "self.reservoirDens"
    self.anisoStdE = 12 #6 #10 #12  # space constant, std of gaussian for excitatory neurons
    self.anisoStdI = 9 #4.5 #8 #9  # space constant, std of gaussian for inhibitory neurons (range 9 - 11)
    self.anisoShift = 1  # intensity of the shift of the connectivity distribution for a neuron
    self.anisoPerlinScale = 4 #8 # 4-12  # perlin noise scale, high value => dense valleys, low value => broad valleys
    self.weightExCoefficient = 12 #16 #8 #16 # 8 #8 #16 #8 #4  # coefficient for excitatory anisotropic weight
    self.weightInCoefficient = 48 #64 #32 #64 # 28 #32 #64 #28 sieht gut aus!! #32 #22  # coefficient for inhibitory anisotropic weight, Perlin scale 4: 25-30 ok, 25-28 good

    # Neuron
    self.compartmentVoltageDecay = 200 #20  # voltage decay
    #self.compartmentVoltageTimeConstant = 
    self.refractoryDelay = 2  # refractory period for a neuron

    # Readout training
    #self.flipProb = 0.05 #0.05  # percentage of neuron flips in every trial
    self.smoothingWd = 3  # number of neurons to the left and right which are influenced
    self.smoothingVar = 7  # variance of the Gaussian kernel

    # Output layer
    self.partitioningClusterSize = 10  # size of clusters connected to an output neuron
    self.outputWeightValue = 8  # weight for output neurons

    # Plasticity
    self.isLearningRule = False  # defines if learning rule is active or not (true/false)
    self.learningImpulse = 5  # impulse for learning rule for x1 and y1
    self.learningTimeConstant = 10 #10  # time constant for learning rule for x1 and y1
    #self.learningRule = '2^-2*x1*y0 - 2^-2*y1*x0'  # applied learning rule
    #self.learningRule = '2^-2*x1*y0 - 2^-2*y1*x0'  # applied learning rule
    #self.learningRule = '2^-2*x1*y0 - 2^-2*y1*x0 + 2^-1*x0*y0 - 2^-3*y0*y0*w'  # applied learning rule
    self.learningEpoch = 4  # learning epoch for learnin rule
    self.homeostasisStatus = False # defines if homesostatic plasticity is active or not

    # Network size and connections
    self.reservoirExSize = 3600 #50 #400 #3600  # size of the excitatory network
    self.reservoirInSize = None  # size of the inhibitory network
    self.reservoirInExRatio = 1./4.  # number excitatory/inhibitory neurons ratio
    self.reservoirDens = None  # connection density of the network
    self.numConnectionsPerNeuron = 50 #50 #45 #100  # average number of connections per neuron

    # Trace input
    self.traceClusters = 3  # number of trace clusters
    self.traceGens = 50  # number of trace generators per input cluster
    self.traceSteps = 20 #50 #20  # number of steps the trace input should drive the network
    self.traceDens = 0.2  # percent of connections to reservoir from input
    self.traceSpikeProb = 0.2  # probability of spike for the generator
    #self.traceMaxWeight = 255  # maximum weight a trace connection can have
    self.traceClusterShare = 0.1  # percentage of excitatory neurons a cluster should be connected with
    self.traceClusterSize = None  # the number of neurons a trace cluster has
    self.stepsPerIteration = None  # overall steps for whole trace sequence

    # Constant input
    self.constGens = 50 #10  # number of constant signal generators
    self.constDens = 0.1 #0.2  # percent of connections to reservoir from input
    self.constSpikeProb = 0.2  # probability of spike for the generator
    self.constSizeShare = 0.1  # share of neurons for the constant input in relation to excitatory network size
    self.constSize = None  # number of neurons for the constant input

    # Cue input
    #self.cueGens = 50 #50 #10  # number of cue generators
    self.cueSteps = 1 #5 #2 #200 #100  # number of steps the cue should drive the network, if None, cue is background activity over all steps
    self.cueRelaxation = 0 #23  # time to wait for relaxation of the network activity after cue
    #self.cueDens = 0.2 # 0.1  # percent of connections to reservoir from input
    #self.cueSpikeProb = 0.5  # probability of spike for the generator, needs to be 0.5 if flips are applied (to remain equal input strength)
    self.cueMaxWeight = 255 #200 #100  # maximum weight a cue connection can have
    self.cuePatchNeuronsShift = 0 #20  # shift in x and y direction of cue input
    self.cuePatchNeurons = 25  # number of neurons for the cue input (needs to be root squarable)
    self.cueGensPerNeuron = 16  # number of generators per input neuron

    # Stop input
    self.stopStart = None  # point in time where the stop signal should start
    #self.stopGens = 20  # number of stop generators
    self.stopSteps = 30  # number of time steps the stop generator should be active
    self.stopRelaxation = 0  # number time steps to relax after the stop signal 
    #self.stopSpikeProb = 0.5  # probability of spike for the generator

    # Noise
    self.noiseNeuronsShare = 0.1 # share of noise neurons in relation to excitatory neurons
    self.noiseNeurons = None  # number of noise neurons (calculated if None)
    self.noiseSpikeprob = 0.05  # in average, every 1/x th time step a noise neuron fires
    self.noiseDens = 0.05 # 0.005 # connectivity between noise neurons and reservoir
    self.noiseMaxWeight = 10 #2  # maximum weight a noise connection can have

    # Probes
    self.isExSpikeProbe = False  # activate/deactivate spike probes for excitatory neurons
    self.isInSpikeProbe = False  # activate/deactivate spike probes for inhibitory neurons
    self.isOutSpikeProbe = True  # activate/deactivate spike probes for output neurons
    self.weightProbe = False  # read weights at the end of the simulation
    self.isExVoltageProbe = False  # activate/deactivate voltage probes for excitatory neurons
    self.isInVoltageProbe = False  # activate/deactivate voltage probes for inhibitory neurons
    self.isOutVoltageProbe = False  # activate/deactivate voltage probes for output neurons
    self.isExCurrentProbe = False  # activate/deactivate current probes for excitatory neurons
    self.isInCurrentProbe = False  # activate/deactivate current probes for inhibitory neurons
