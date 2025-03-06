/*
 * Copyright (c) 2024 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#ifndef GAIN_GAIN_SPECIFICATION_H
#define GAIN_GAIN_SPECIFICATION_H

#include <cstdint>
#include <stddef.h>

namespace GainConfig {

struct Parameters {
    static constexpr uint32_t GainMessage = 0xDE2F52AD;
    uint32_t ThisMessage {GainMessage};

    float gain_value {};
};

struct Specification {
    static constexpr uint32_t GainConstructionType = 0xDE2F52AC;
    uint32_t ThisType {GainConstructionType};

    Parameters params {};
};

} // namespace GainConfig

#endif // GAIN_GAIN_SPECIFICATION_H
