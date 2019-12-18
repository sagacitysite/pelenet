/*
INTEL CONFIDENTIAL

Copyright © 2018 Intel Corporation.

This software and the related documents are Intel copyrighted
materials, and your use of them is governed by the express
license under which they were provided to you (License). Unless
the License provides otherwise, you may not use, modify, copy,
publish, distribute, disclose or transmit  this software or the
related documents without Intel's prior written permission.

This software and the related documents are provided as is, with
no express or implied warranties, other than those that are
expressly stated in the License.
*/

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
