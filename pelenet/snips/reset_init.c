#include <stdlib.h>
#include <string.h>
#include "time.h"
#include "reset_init.h"

// Values necessary for reset
int numCores;
int resetInterval;
int enableReset;

int channelID;

// Name of the channel
//char channelName[] = "initreset";
//char *channelName[2];
//channelName[0] = "initreset";
//channelName[1] = "initreset2";
char channelName1[] = "initreset";
char channelName2[] = "initreset2";


void initialize_reset(runState *s) {

    // Get channel id from channel name
    int channelID1 = getChannelID(channelName1);
    int channelID2 = getChannelID(channelName2);

    if (channelID1 != -1) {
      channelID = channelID1;
    }

    if (channelID2 != -1) {
      channelID = channelID2;
    }

    // Check if channel id is valid
    if (channelID1 == -1 && channelID2 == -1) {
      printf("Invalid channelName %s %s\n", channelName1, channelName2);
      return;
    }

    // Read values from channel buffer
    readChannel(channelID, &numCores, 1);
    readChannel(channelID, &resetInterval, 1);
    readChannel(channelID, &enableReset, 1);

    // Log results
    printf("Transfered values %d, %d, %d \n", numCores, resetInterval, enableReset);
}
