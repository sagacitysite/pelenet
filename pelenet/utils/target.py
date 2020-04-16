import numpy as np
import statsmodels.api as sm

"""
@desc: Load robotic movement target function
"""
def loadTarget(self):
    # Target range
    fr = self.p.targetOffset
    to = None if fr is None else (fr + self.p.stepsPerTrial-self.p.binWindowSize)

    # Define file path
    filePath = self.p.targetPath + self.p.targetFilename

    # Load data and return
    return np.loadtxt(filePath)[fr:to,0:3].T

"""
@desc:
        Prepares dataset for estimation
@pars:
        data: has dimensions B (number of trials) x N (number of neurons) x T (number of time steps per trial)
        target: n-dim target function
        binSize: bin works as sliding window to smooth spikes with binSize into the past
        trainTrials: binary array for choosing trials for training (default: all but last trial)
        testTrial: integer which gives the trial number to test with (default: last trial)
@return:
        x: prepared train data
        xe: prepared test data
        y: prepared n-dim target function
"""
def prepareDataset(self, data, target, binSize=None, trainTrials=None, testTrial=None):
    # Set binSize to default parameter value if not set
    if binSize is None: binSize = self.p.binWindowSize
    # If not train trials are given, take all except the last one
    if trainTrials is None: trainTrials = np.append(np.repeat(True, self.p.trials-1), False)
    # If not test trial is given, take the last one
    if testTrial is None: testTrial = self.p.trials-1

    # Select train data
    train = np.array([np.mean(data[trainTrials,:,i:i+binSize], axis=2) for i in range(self.p.stepsPerTrial-binSize)])
    trainSpikes = np.moveaxis(train, 0, 2)

    # Concatenate train data
    x = np.hstack(tuple( trainSpikes[i,:,:] for i in range(self.p.trials-1) ))
    x = np.insert(x, 0, 1.0, axis=0)  # Add intercept

    # Select test data
    test = np.array([np.mean(data[testTrial,:,i:i+binSize], axis=1) for i in range(self.p.stepsPerTrial-binSize)])
    testSpikes = np.moveaxis(test, 0, 1)
    xe = np.insert(testSpikes, 0, 1.0, axis=0)  # Add intercept
    
    # Select target
    y = np.tile(target[:,:self.p.stepsPerTrial], self.p.trials-1)

    # Return dataset
    return (x, xe, y)

"""
@desc: Estimates a one dimensional function with given train and test data
@pars:
        x: train data
        xe: test data
        y: one dimensional target function
@return:
        ye: estimated target function
"""
def estimateMovement(self, x, xe, y):
    # Fit
    model = sm.OLS(y, x.T)
    params = model.fit_regularized(alpha=0.0, L1_wt=0.0).params
    
    # Predict
    ye = np.dot(xe.T, params)
    
    return ye
