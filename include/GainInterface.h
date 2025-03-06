#ifndef GPU_GAIN_INTERFACE_H
#define GPU_GAIN_INTERFACE_H

#include <gain_processor/GainSpecification.h>

#include <cstdint>

class GainInterface {
public:
    virtual ~GainInterface() {};

    /**
     * @brief Process samples provided in input and write them output buffers.
     * @param input [in] pointer to pointers to the channels of the input audio data
     * @param output [out] pointer to pointers to the channels of the output audio data
     */
    virtual void process(float const* const* input, float* const* output, const int nsamples) = 0;

    /**
     * @brief Get the client library ready for processing with the current configuration
     */
    virtual void arm() = 0;

    /**
     * @brief Clean up and get ready for destruction or re-configuration
     */
    virtual void disarm() = 0;

    /**
     * @brief Set the gain value used in supsequent process calls
     * @param gain [in] gain value to apply during processing
     */
    virtual void set_gain(float gain) = 0;

    /**
     * @brief Turn automatic buffer growth on or off.
     * @param enable [in] true to enable, false to disable
     */
    virtual void enable_buffer_growth(bool enable) = 0;

    /**
     * @brief Get the current latency introduced by double buffering
     * @return latency in number of samples
     */
    virtual uint32_t get_latency() = 0;
};

#endif // GPU_GAIN_INTERFACE_H
