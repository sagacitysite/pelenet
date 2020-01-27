import os
import logging
from nxsdk.logutils.nxlogging import LoggingLevel

"""
@desc: Include parameters regarding the system (e.g. error/warning thresholds)
"""
def includeSystem(self):

    # Data log
    self.loihiLoggingLevel = LoggingLevel.INFO
    self.systemLoggingLevel = logging.INFO
    self.dataLogPath = os.path.join(os.getcwd(), 'datalog/')  # Define datalog path
    self.expLogPath = None  # The explicit path within dataLogPath, which is set when parameter instance is created, it depends on datetime

    # Loihi chip
    self.numChips = 2  # number of cores available in current setting
    self.numCoresPerChip = 128  # number of cores per chip in current setting
    self.neuronsPerCore = 20 #1024 #10 #20 #55  # numer of neurons distributed to each Loihi core
    self.bufferFactor = 20  # number of timesteps the buffer should collect spike data from the Loihi cores

    # Validity check
    self.maxSparseToDenseLimit = 6000  # When e.g. doing an imshow plot a normal dense matrix is necessary, which only can be transformed from a sparse matrix, when memory is sufficient
