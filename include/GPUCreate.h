#pragma once

#include "GainInterface.h"

#include <cstdint>
#include <memory>

/**
 * @brief Create an instance of the GPUGainProcessor.
 * @param nchannels [in] number of channels of the audio data to process
 * @param nsamples_per_channel [in] capacity of the processing-buffer per channel
 * @param double_buffering [in] enable or disable double buffering, i.e., async execution
 * @return GainInterface pointer to the created GPUGainProcessor instance
 */
std::unique_ptr<GainInterface> createGpuProcessor(uint32_t nchannels = 2u, uint32_t nsamples_per_channel = 256u, bool double_buffering = false);
