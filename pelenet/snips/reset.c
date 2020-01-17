#include <stdlib.h>
#include <string.h>
#include "time.h"
#include "reset.h"

static int numNeuronsPerCore = 1024;

int tImgStart = 0;
int tImgEnd = 0;

extern int numCores;
extern int resetInterval;
extern int enableReset;

int do_reset(runState *RunState) {
    bool apply = false;

    for(int i=0; i<10; i++) {
        apply = apply || (RunState->time_step - i) % resetInterval == 0;
    }

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


