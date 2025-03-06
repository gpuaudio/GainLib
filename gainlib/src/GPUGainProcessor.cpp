#include "GPUGainProcessor.h"

#include <gpu_audio_client/GpuAudioManager.h>

#include <engine_api/DeviceInfoProvider.h>
#include <engine_api/LauncherSpecification.h>
#include <engine_api/ModuleInfo.h>

#define _USE_MATH_DEFINES
#include <algorithm>
#include <array>
#include <cassert>
#include <math.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

template <ExecutionMode EXEC_MODE>
GPUGainProcessor<EXEC_MODE>::GPUGainProcessor(uint32_t nchannels, uint32_t nsamples_per_channel) :
    m_nchannels {nchannels} {
    // buffer settings and double buffering configuration (see `gpu_audio_client` for details)
    m_executor_config = {
        .retain_threshold = 0.625,
        .launch_threshold = 0.7275,
        .nchannels_in = m_nchannels,
        .nchannels_out = m_nchannels,
        .max_samples_per_channel = nsamples_per_channel};

    // initial gain value set to 1.0, i.e., input == output
    m_processor_spec.params.gain_value = 1.0f;

    // create gpu_audio engine and make sure a supported GPU is installed
    const auto& gpu_audio = GpuAudioManager::GetGpuAudio();
    const auto& device_info_provider = gpu_audio->GetDeviceInfoProvider();
    const auto dev_idx = GpuAudioManager::GetDeviceIndex();
    if (dev_idx >= device_info_provider.GetDeviceCount()) {
        throw std::runtime_error("No supported device found");
    }
    // get all the information about the GPU device required to create a launcher
    GPUA::engine::v2::LauncherSpecification launcher_spec = {};
    if ((device_info_provider.GetDeviceInfo(dev_idx, launcher_spec.device_info) != GPUA::engine::v2::ErrorCode::eSuccess) || !launcher_spec.device_info) {
        throw std::runtime_error("Failed to get device info");
    }
    // create a launcher for the specified GPU device
    if ((gpu_audio->CreateLauncher(launcher_spec, m_launcher) != GPUA::engine::v2::ErrorCode::eSuccess) || !m_launcher) {
        throw std::runtime_error("Failed to create launcher");
    }
    // create a processing graph, which will hold the gain processor
    if ((m_launcher->CreateProcessingGraph(m_graph) != GPUA::engine::v2::ErrorCode::eSuccess) || !m_graph) {
        gpu_audio->DeleteLauncher(m_launcher);
        throw std::runtime_error("Failed to create processing graph");
    }

    // iterate over all available modules to find the gain processor
    auto& module_provider = m_launcher->GetModuleProvider();
    const auto module_count = module_provider.GetModulesCount();
    GPUA::engine::v2::ModuleInfo info {};
    bool processor_module_found = false;
    for (uint32_t i = 0; i < module_count; ++i) {
        if ((module_provider.GetModuleInfo(i, info) == GPUA::engine::v2::ErrorCode::eSuccess) && info.id && (std::wcscmp(info.id, L"gain") == 0)) {
            processor_module_found = true;
            break;
        }
    }
    // clean up if we did not find the gain processor in the search paths
    if (!processor_module_found) {
        m_launcher->DeleteProcessingGraph(m_graph);
        gpu_audio->DeleteLauncher(m_launcher);
        throw std::runtime_error("Failed to find required processor module");
    }

    // load the gain processor module
    if ((module_provider.GetModule(info, m_module) != GPUA::engine::v2::ErrorCode::eSuccess) || !m_module) {
        m_launcher->DeleteProcessingGraph(m_graph);
        gpu_audio->DeleteLauncher(m_launcher);
        throw std::runtime_error("Failed to load required processor module");
    }
};

template <ExecutionMode EXEC_MODE>
GPUGainProcessor<EXEC_MODE>::~GPUGainProcessor() {
    // delete executor and processor
    disarm();
    // delete the processing graph
    m_launcher->DeleteProcessingGraph(m_graph);
    // delete the launcher
    GpuAudioManager::GetGpuAudio()->DeleteLauncher(m_launcher);
}

template <ExecutionMode EXEC_MODE>
void GPUGainProcessor<EXEC_MODE>::arm() {
    std::lock_guard<std::mutex> lock(m_armed_mutex);

    if (!m_armed) {
        // use the module to create a processor, following the processor specification, in the graph
        if (m_module->CreateProcessor(m_graph, &m_processor_spec, sizeof(m_processor_spec), m_processor) != GPUA::engine::v2::ErrorCode::eSuccess || !m_processor) {
            throw std::runtime_error("Failed to create processor");
        }
        // create an executor that manages input and output buffers and performs the actual launches
        m_process_executor = new ProcessExecutor<EXEC_MODE>(m_launcher, m_graph, 1u, &m_processor, m_executor_config);
        m_armed = true;
    }
}

template <ExecutionMode EXEC_MODE>
void GPUGainProcessor<EXEC_MODE>::disarm() {
    std::lock_guard<std::mutex> lock(m_armed_mutex);

    if (m_armed) {
        // delete the executor. ensures that all launches have finished before destroying itself.
        if (m_process_executor) {
            delete m_process_executor;
            m_process_executor = nullptr;
        }

        // as no more launches are active, it's safe to destroy the processor
        if (m_processor) {
            m_module->DeleteProcessor(m_processor);
            m_processor = nullptr;
        }
        m_armed = false;
    }
}

template <ExecutionMode EXEC_MODE>
void GPUGainProcessor<EXEC_MODE>::process(float const* const* in_buffer, float* const* out_buffer, int nsamples) {
    // If the processor was not armed ahead of time, arm it on the first process call.
    if (!m_armed) {
        arm();
    }

    // If we get more samples to process than the buffers can currently hold, we increase their size if buffer growth is enabled.
    // The current content of the buffers is lost - only relevant if double buffering is enabled, as we then get a full buffer of
    // zeros after the buffer realloc.
    if (m_buffer_growth_enabled && nsamples > m_executor_config.max_samples_per_channel && m_executor_config.max_samples_per_channel < MaxSampleCount) {
        do {
            m_executor_config.max_samples_per_channel = std::min(m_executor_config.max_samples_per_channel * 2, MaxSampleCount);
        } while (nsamples > m_executor_config.max_samples_per_channel && m_executor_config.max_samples_per_channel < MaxSampleCount);
        renewExecutor();
    }

    thread_local std::vector<float const*> input_ptrs;
    thread_local std::vector<float*> output_ptrs;
    // If we have to perform multiple launches, we have to copy the channel pointers s.t. we can modify them.
    if (nsamples > m_executor_config.max_samples_per_channel) {
        input_ptrs.assign(in_buffer, in_buffer + m_nchannels);
        output_ptrs.assign(out_buffer, out_buffer + m_nchannels);
    }

    uint32_t remaining_samples = nsamples;
    while (remaining_samples != 0) {
        // number of samples for this launch
        uint32_t this_launch_samples = std::min(m_executor_config.max_samples_per_channel, remaining_samples);

        if constexpr (EXEC_MODE == ExecutionMode::eAsync) {
            // Async execution, i.e., double buffering enabled
            // order processing of samples [i, i + this_launch_samples)
            m_process_executor->template ExecuteAsync<AudioDataLayout::eChannelsIndividual>(this_launch_samples, in_buffer);
            // retrieve samples from earlier launch(es) [i - m_executor_config.max_samples_per_channel, i - m_executor_config.max_samples_per_channel + this_launch_samples)
            m_process_executor->template RetrieveOutput<AudioDataLayout::eChannelsIndividual>(this_launch_samples, out_buffer);
        }
        else {
            // Synchronous execution, i.e., double buffering disabled
            // process samples [i, i + this_launch_samples)
            m_process_executor->template Execute<AudioDataLayout::eChannelsIndividual>(this_launch_samples, in_buffer, out_buffer);
        }

        // advance channel pointers for the next iteration if required
        remaining_samples -= this_launch_samples;
        if (remaining_samples != 0) {
            for (auto& ptr : input_ptrs)
                ptr += this_launch_samples;
            in_buffer = input_ptrs.data();

            for (auto& ptr : output_ptrs)
                ptr += this_launch_samples;
            out_buffer = output_ptrs.data();
        }
    }
}

template <ExecutionMode EXEC_MODE>
void GPUGainProcessor<EXEC_MODE>::set_gain(float gain) {
    std::lock_guard<std::mutex> lock(m_armed_mutex);

    if (!m_armed) {
        // update specification - will take effect when the processor is created
        m_processor_spec.params.gain_value = gain;
    }
    else {
        throw std::runtime_error("GPUGainProcessor::set_gain can only be called while disarmed\n");
    }
}

template <ExecutionMode EXEC_MODE>
void GPUGainProcessor<EXEC_MODE>::enable_buffer_growth(bool enable) {
    m_buffer_growth_enabled = enable;
}

template <ExecutionMode EXEC_MODE>
uint32_t GPUGainProcessor<EXEC_MODE>::get_latency() {
    if constexpr (EXEC_MODE == ExecutionMode::eAsync) {
        // distance between first sample to process and first processed sample retrieved in a single process call
        return m_executor_config.max_samples_per_channel;
    }
    return 0u;
}

template <ExecutionMode EXEC_MODE>
void GPUGainProcessor<EXEC_MODE>::renewExecutor() {
    std::lock_guard<std::mutex> lock(m_armed_mutex);
    // delete executor after all launches finished
    if (m_process_executor) {
        delete m_process_executor;
        m_process_executor = nullptr;
    }
    // create new executor to apply changes in the executor config or in the processor
    m_process_executor = new ProcessExecutor<EXEC_MODE>(m_launcher, m_graph, 1u, &m_processor, m_executor_config);
}

template class GPUGainProcessor<ExecutionMode::eSync>;
template class GPUGainProcessor<ExecutionMode::eAsync>;
