#include <GPUCreate.h>

#include "GPUGainProcessor.h"

std::unique_ptr<GainInterface> createGpuProcessor(uint32_t nchannels, uint32_t nsamples_per_channel, bool double_buffering) {
    if (double_buffering) {
        return std::make_unique<GPUGainProcessor<ExecutionMode::eAsync>>(nchannels, nsamples_per_channel);
    }
    else {
        return std::make_unique<GPUGainProcessor<ExecutionMode::eSync>>(nchannels, nsamples_per_channel);
    }
}
