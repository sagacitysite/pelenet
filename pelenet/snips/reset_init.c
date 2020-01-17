#include <stdlib.h>
#include <string.h>
#include "time.h"
#include "reset_init.h"

// Values necessary for reset
int numCores;
int resetInterval;
int enableReset;

// Name of the channel
char channelName[] = "initreset";

void initialize_reset(runState *s) {

    // Get channel id from channel name
    int channelID = getChannelID(channelName);

    // Check if channel id is valid
    if (channelID == -1) {
      printf("Invalid channelName %s\n", channelName);
      return;
    }

    // Read values from channel buffer
    readChannel(channelID, &numCores, 1);
    readChannel(channelID, &resetInterval, 1);
    readChannel(channelID, &enableReset, 1);

    // Log results
    printf("Transfered values %d, %d, %d \n", numCores, resetInterval, enableReset);
}
