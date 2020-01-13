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
#include "reset.h"

static int numNeuronsPerCore = 1024; // 1024
//static int NUM_Y_TILES = 5;

int tImgStart = 0;
int tImgEnd = 0;

extern int numCores;
extern int resetInterval;
extern int enableReset;

/*void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId) {
    physicalId->p = logicalId % 4;
    physicalId->x = logicalId/(4*(NUM_Y_TILES-1));
    physicalId->y = (logicalId - physicalId->x*4*(NUM_Y_TILES-1))/4 + 1;
}*/

int do_reset(runState *RunState) {
    bool apply = false;

    for(int i=0; i<10; i++) {
        apply = apply || (RunState->time_step - i) % resetInterval == 0;
    }

    //if (enableReset && (RunState->time_step == 1 || (RunState->time_step - 1) % resetInterval == 0)) {
    //if (enableReset && (RunState->time_step == 1 || (RunState->time_step - 5) % resetInterval == 0 || (RunState->time_step - 4) % resetInterval == 0 || (RunState->time_step - 3) % resetInterval == 0 || (RunState->time_step - 2) % resetInterval == 0 || (RunState->time_step - 1) % resetInterval == 0)) {
    if (enableReset && (RunState->time_step == 1 || apply)) {
        return 1;
    } else {
        return 0;
    }
}

void reset(runState *RunState) {

    CxState cxs = (CxState) {.U=0, .V=0};
    for(int i=0; i<numCores; i++) {
        NeuronCore* nc = NEURON_PTR(nx_nth_coreid(i));
        // Sets all 1024 registers on the neurocore
        nx_fast_init64(nc->cx_state, numNeuronsPerCore, *(uint64_t*)&cxs);
        //nx_fast_init64(nc->cx_state, numNeuronsPerCore, 0);

        nx_flush_core(nx_nth_coreid(i));
    }

    // Flush spikes
    //SPIKE_COUNT[RunState->time_step][0x20] = 0;
}


