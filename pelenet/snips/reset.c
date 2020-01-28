#include <stdlib.h>
#include <string.h>
#include "time.h"
#include "reset.h"

static int numCores = 128;

extern int neuronsPerCore;
extern int resetInterval;
extern int stopSteps;

int do_reset(runState *RunState) {
    bool apply = false;

    // Prepare boolean variable, which is true for 10 time steps
    for(int i = 1; i <= stopSteps; i++) {
        apply = apply || (RunState->time_step - i) % resetInterval == 0;
    }


    if (RunState->time_step == 1 || apply) {
        return 1;
    } else {
        return 0;
    }
}

void reset(runState *RunState) {

    CxState cxs = (CxState) {.U=0, .V=0};

    // Iterate over all cores
    for(int i = 0; i < numCores; i++) {
        NeuronCore* nc = NEURON_PTR(nx_nth_coreid(i));
        // Sets all 1024 registers on the neurocore
        nx_fast_init64(nc->cx_state, neuronsPerCore, *(uint64_t*)&cxs);
        //nx_fast_init64(nc->cx_state, neuronsPerCore, 0);

        nx_flush_core(nx_nth_coreid(i));
    }

    // Flush spikes
    //SPIKE_COUNT[RunState->time_step][0x20] = 0;
}


