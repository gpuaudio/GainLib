#include <gtest/gtest.h>

#include "TestCommon.h"

#include <GPUCreate.h>
#include "../include/gain_processor/GainSpecification.h"

namespace {
void apply_gain(TestData& data, float gain) {
    for (uint32_t ch {0u}; ch < data.m_nchannels; ++ch) {
        for (uint32_t s {0u}; s < data.m_nsamples; ++s) {
            data.at(ch, s) *= gain;
        }
    }
}
} // namespace

TEST(GainLib, CreateDestroy) {
    std::unique_ptr<GainInterface> gainlib = createGpuProcessor(2u, 256u);
    ASSERT_NE(gainlib, nullptr);
}

TEST(GainLib, CreateArmDisarmDestroy) {
    std::unique_ptr<GainInterface> gainlib = createGpuProcessor(2u, 256u);
    ASSERT_NE(gainlib, nullptr);

    TestData input(2, 256, 1.0f, TestData::DataMode::Sin);
    TestData output(2, 256, 0.0f, TestData::DataMode::Constant);

    gainlib->arm();
    gainlib->disarm();
}

TEST(GainLib, CreateProcessDestroy) {
    std::unique_ptr<GainInterface> gainlib = createGpuProcessor(2u, 256u);
    ASSERT_NE(gainlib, nullptr);

    TestData input(2, 256, 1.0f, TestData::DataMode::Sin);
    TestData output(2, 256, 0.0f, TestData::DataMode::Constant);

    gainlib->process(input.m_data, output.m_data, 256);
}

TEST(GainLib, ProcessValidate) {
    std::unique_ptr<GainInterface> gainlib = createGpuProcessor(2u, 256u);
    ASSERT_NE(gainlib, nullptr);

    TestData input(2, 256, 1.0f, TestData::DataMode::Sin);
    TestData output(2, 256, 0.0f, TestData::DataMode::Constant);

    gainlib->process(input.m_data, output.m_data, 256);
    ASSERT_TRUE(input == output);
}

TEST(GainLib, SetGainProcessValidate) {
    std::unique_ptr<GainInterface> gainlib = createGpuProcessor(2u, 256u);
    ASSERT_NE(gainlib, nullptr);

    TestData input(2, 256, 1.0f, TestData::DataMode::Sin);
    TestData output(2, 256, 0.0f, TestData::DataMode::Constant);

    gainlib->set_gain(2.0f);
    gainlib->process(input.m_data, output.m_data, 256);
    ASSERT_FALSE(input == output);

    apply_gain(input, 2.0f);
    ASSERT_TRUE(input == output);
}

void process_test(uint32_t nchannels, uint32_t buffer_size, uint32_t nlaunches, uint32_t min_nsamples, uint32_t max_nsamples, bool double_buffering = false) {
    std::unique_ptr<GainInterface> gainlib = createGpuProcessor(nchannels, buffer_size, double_buffering);
    ASSERT_NE(gainlib, nullptr);

    // turn off buffer growth for simpler validation
    if (double_buffering) {
        gainlib->enable_buffer_growth(false);
    }

    std::vector<uint32_t> nsamples_per_launch(nlaunches);
    std::random_device dev;
    std::mt19937 rne(dev());
    std::uniform_int_distribution<uint32_t> nsamples_dist(min_nsamples, max_nsamples);
    uint32_t nsamples_total {0u};
    for (auto& ns : nsamples_per_launch) {
        ns = nsamples_dist(rne);
        nsamples_total += ns;
    }

    TestData input(nchannels, nsamples_total, 1.0f, TestData::DataMode::Sin);
    std::vector<float*> in_ptr(nchannels, nullptr);
    TestData output(nchannels, nsamples_total, 0.0f, TestData::DataMode::Constant);
    std::vector<float*> out_ptr(nchannels, nullptr);

    float const gain {3.14f};
    gainlib->set_gain(gain);
    uint32_t sample_off {0u};
    for (auto& ns : nsamples_per_launch) {
        for (uint32_t ch {0u}; ch < nchannels; ++ch) {
            in_ptr[ch] = input.getChannel(ch) + sample_off;
            out_ptr[ch] = output.getChannel(ch) + sample_off;
        }
        gainlib->process(in_ptr.data(), out_ptr.data(), ns);
        sample_off += ns;
    }
    ASSERT_FALSE(CompareBuffers(input, 0u, output, double_buffering ? buffer_size : 0u, 1e-4f));

    apply_gain(input, gain);
    ASSERT_TRUE(CompareBuffers(input, 0u, output, double_buffering ? buffer_size : 0u, 1e-4f));
}

TEST(GainLib, Process2ChPartiallyFilledValidate) {
    uint32_t nchannels {2u}, buffer_size {256u}, nlaunches {50u};
    uint32_t min_nsamples {1u}, max_nsamples {buffer_size};
    process_test(nchannels, buffer_size, nlaunches, min_nsamples, max_nsamples);
}

TEST(GainLib, Process2ChPartiallyFilledDoubleBufferedValidate) {
    uint32_t nchannels {2u}, buffer_size {256u}, nlaunches {50u};
    uint32_t min_nsamples {1u}, max_nsamples {buffer_size};
    process_test(nchannels, buffer_size, nlaunches, min_nsamples, max_nsamples, true);
}

TEST(GainLib, Process32ChPartiallyFilledValidate) {
    uint32_t nchannels {32u}, buffer_size {256u}, nlaunches {50u};
    uint32_t min_nsamples {1u}, max_nsamples {buffer_size};
    process_test(nchannels, buffer_size, nlaunches, min_nsamples, max_nsamples);
}

TEST(GainLib, Process32ChPartiallyFilledDoubleBufferedValidate) {
    uint32_t nchannels {32u}, buffer_size {256u}, nlaunches {50u};
    uint32_t min_nsamples {1u}, max_nsamples {buffer_size};
    process_test(nchannels, buffer_size, nlaunches, min_nsamples, max_nsamples, true);
}

TEST(GainLib, Process2ChPartiallyFilledAndDeviceIterateValidate) {
    uint32_t nchannels {2u}, buffer_size {256u}, nlaunches {50u};
    uint32_t min_nsamples {1u}, max_nsamples {4u * buffer_size};
    process_test(nchannels, buffer_size, nlaunches, min_nsamples, max_nsamples);
}

TEST(GainLib, Process2ChPartiallyFilledAndDeviceIterateDoubleBufferedValidate) {
    uint32_t nchannels {2u}, buffer_size {256u}, nlaunches {50u};
    uint32_t min_nsamples {1u}, max_nsamples {4u * buffer_size};
    process_test(nchannels, buffer_size, nlaunches, min_nsamples, max_nsamples, true);
}

TEST(GainLib, Process32ChPartiallyFilledAndDeviceIterateValidate) {
    uint32_t nchannels {32u}, buffer_size {256u}, nlaunches {50u};
    uint32_t min_nsamples {1u}, max_nsamples {4u * buffer_size};
    process_test(nchannels, buffer_size, nlaunches, min_nsamples, max_nsamples);
}

TEST(GainLib, Process32ChPartiallyFilledAndDeviceIterateDoubleBufferedValidate) {
    uint32_t nchannels {32u}, buffer_size {256u}, nlaunches {50u};
    uint32_t min_nsamples {1u}, max_nsamples {4u * buffer_size};
    process_test(nchannels, buffer_size, nlaunches, min_nsamples, max_nsamples, true);
}
