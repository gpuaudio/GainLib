#ifndef GPUA_GPU_GAIN_PROCESSOR_H
#define GPUA_GPU_GAIN_PROCESSOR_H

#include <GainInterface.h>

#include <engine_api/GraphLauncher.h>
#include <engine_api/Module.h>
#include <engine_api/ProcessingGraph.h>
#include <engine_api/Processor.h>

#include <gpu_audio_client/ProcessExecutorSync.h>
#include <gpu_audio_client/ProcessExecutorAsync.h>

#include <array>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

template <ExecutionMode EXEC_MODE>
class GPUGainProcessor : public GainInterface {
public:
    GPUGainProcessor(uint32_t nchannels, uint32_t nsamples_per_channel);
    virtual ~GPUGainProcessor();

    ////////////////////////////////
    // GainInterface methods
    virtual void arm() override;
    virtual void disarm() override;
    virtual void process(float const* const* in_buffer, float* const* out_buffer, int nsamples) override;

    virtual void set_gain(float gain) override;

    virtual void enable_buffer_growth(bool enable) override;
    virtual uint32_t get_latency() override;
    // GainInterface methods
    ////////////////////////////////

private:
    std::mutex m_armed_mutex;
    bool m_armed {false};

    uint32_t const m_nchannels;
    static constexpr uint32_t MaxSampleCount {4096u};

    GPUA::engine::v2::GraphLauncher* m_launcher {nullptr};
    GPUA::engine::v2::ProcessingGraph* m_graph {nullptr};
    GPUA::engine::v2::Module* m_module {nullptr};

    GainConfig::Specification m_processor_spec;

    GPUA::engine::v2::Processor* m_processor {nullptr};

    bool m_buffer_growth_enabled {true};

    ProcessExecutorConfig m_executor_config;
    ProcessExecutor<EXEC_MODE>* m_process_executor {nullptr};

    void renewExecutor();
};

#endif // GPUA_GPU_GAIN_PROCESSOR_H
